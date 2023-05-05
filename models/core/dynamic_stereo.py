# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


from dynamic_stereo.models.core.update import (
    BasicUpdateBlock,
    SequenceUpdateBlock3D,
    TimeAttnBlock,
)
from dynamic_stereo.models.core.extractor import BasicEncoder
from dynamic_stereo.models.core.corr import CorrBlock1D

from dynamic_stereo.models.core.attention import (
    PositionEncodingSine,
    LocalFeatureTransformer,
)
from dynamic_stereo.models.core.utils.utils import InputPadder, interp

autocast = torch.cuda.amp.autocast


class DynamicStereo(nn.Module):
    def __init__(
        self,
        max_disp: int = 192,
        mixed_precision: bool = False,
        num_frames: int = 5,
        attention_type: str = None,
        use_3d_update_block: bool = False,
        different_update_blocks: bool = False,
    ):
        super(DynamicStereo, self).__init__()

        self.max_flow = max_disp
        self.mixed_precision = mixed_precision

        self.hidden_dim = 128
        self.context_dim = 128
        dim = 256
        self.dim = dim
        self.dropout = 0
        self.use_3d_update_block = use_3d_update_block
        self.fnet = BasicEncoder(
            output_dim=dim, norm_fn="instance", dropout=self.dropout
        )
        self.different_update_blocks = different_update_blocks
        cor_planes = 4 * 9
        self.depth = 4
        self.attention_type = attention_type
        # attention_type is a combination of the following attention types:
        # self_stereo, temporal, update_time, update_space
        # for example, self_stereo_temporal_update_time_update_space

        if self.use_3d_update_block:
            if self.different_update_blocks:
                self.update_block08 = SequenceUpdateBlock3D(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4
                )
                self.update_block16 = SequenceUpdateBlock3D(
                    hidden_dim=self.hidden_dim,
                    cor_planes=cor_planes,
                    mask_size=4,
                    attention_type=attention_type,
                )
                self.update_block04 = SequenceUpdateBlock3D(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4
                )
            else:
                self.update_block = SequenceUpdateBlock3D(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4
                )
        else:
            if self.different_update_blocks:
                self.update_block08 = BasicUpdateBlock(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4
                )
                self.update_block16 = BasicUpdateBlock(
                    hidden_dim=self.hidden_dim,
                    cor_planes=cor_planes,
                    mask_size=4,
                    attention_type=attention_type,
                )
                self.update_block04 = BasicUpdateBlock(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4
                )
            else:
                self.update_block = BasicUpdateBlock(
                    hidden_dim=self.hidden_dim, cor_planes=cor_planes, mask_size=4
                )

        if attention_type is not None:
            if ("update_time" in attention_type) or ("temporal" in attention_type):
                self.time_embed = nn.Parameter(torch.zeros(1, num_frames, dim))
            if "temporal" in attention_type:
                self.time_attn_blocks = nn.ModuleList(
                    [TimeAttnBlock(dim=dim, num_heads=8) for _ in range(self.depth)]
                )

            if "self_stereo" in attention_type:
                self.self_attn_blocks = nn.ModuleList(
                    [
                        LocalFeatureTransformer(
                            d_model=dim,
                            nhead=8,
                            layer_names=["self"] * 1,
                            attention="linear",
                        )
                        for _ in range(self.depth)
                    ]
                )

                self.cross_attn_blocks = nn.ModuleList(
                    [
                        LocalFeatureTransformer(
                            d_model=dim,
                            nhead=8,
                            layer_names=["cross"] * 1,
                            attention="linear",
                        )
                        for _ in range(self.depth)
                    ]
                )

        self.num_frames = num_frames

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"time_embed"}

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def convex_upsample(self, flow: torch.Tensor, mask: torch.Tensor, rate: int = 4):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def zero_init(self, fmap: torch.Tensor):
        N, _, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        _y = torch.zeros([N, 1, H, W], dtype=torch.float32)
        zero_flow = torch.cat((_x, _y), dim=1).to(fmap.device)
        return zero_flow

    def forward_batch_test(
        self, batch_dict: Dict, kernel_size: int = 14, iters: int = 20
    ):
        stride = kernel_size // 2
        predictions = defaultdict(list)

        disp_preds = []
        video = batch_dict["stereo_video"]
        num_ims = len(video)
        print("video", video.shape)

        for i in range(0, num_ims, stride):
            left_ims = video[i : min(i + kernel_size, num_ims), 0]
            padder = InputPadder(left_ims.shape, divis_by=32)

            right_ims = video[i : min(i + kernel_size, num_ims), 1]
            left_ims, right_ims = padder.pad(left_ims, right_ims)

            with autocast(enabled=self.mixed_precision):
                disparities_forw = self.forward(
                    left_ims[None].cuda(),
                    right_ims[None].cuda(),
                    iters=iters,
                    test_mode=True,
                )

            disparities_forw = padder.unpad(disparities_forw[:, 0])[:, None].cpu()

            if len(disp_preds) > 0 and len(disparities_forw) >= stride:

                if len(disparities_forw) < kernel_size:
                    disp_preds.append(disparities_forw[stride // 2 :])
                else:
                    disp_preds.append(disparities_forw[stride // 2 : -stride // 2])

            elif len(disp_preds) == 0:
                disp_preds.append(disparities_forw[: -stride // 2])

        predictions["disparity"] = (torch.cat(disp_preds).squeeze(1).abs())[:, :1]
        print(predictions["disparity"].shape)

        return predictions

    def forward_sst_block(
        self, fmap1_dw16: torch.Tensor, fmap2_dw16: torch.Tensor, T: int
    ):
        *_, h, w = fmap1_dw16.shape

        # positional encoding and self-attention
        pos_encoding_fn_small = PositionEncodingSine(d_model=self.dim, max_shape=(h, w))
        # 'n c h w -> n (h w) c'
        fmap1_dw16 = pos_encoding_fn_small(fmap1_dw16)
        # 'n c h w -> n (h w) c'
        fmap2_dw16 = pos_encoding_fn_small(fmap2_dw16)

        if self.attention_type is not None:
            # add time embeddings
            if (
                "temporal" in self.attention_type
                or "update_time" in self.attention_type
            ):
                fmap1_dw16 = rearrange(
                    fmap1_dw16, "(b t) m h w -> (b h w) t m", t=T, h=h, w=w
                )
                fmap2_dw16 = rearrange(
                    fmap2_dw16, "(b t) m h w -> (b h w) t m", t=T, h=h, w=w
                )

                # interpolate if video length doesn't match
                if T != self.num_frames:
                    time_embed = self.time_embed.transpose(1, 2)
                    new_time_embed = F.interpolate(time_embed, size=(T), mode="nearest")
                    new_time_embed = new_time_embed.transpose(1, 2).contiguous()
                else:
                    new_time_embed = self.time_embed

                fmap1_dw16 = fmap1_dw16 + new_time_embed
                fmap2_dw16 = fmap2_dw16 + new_time_embed

                fmap1_dw16 = rearrange(
                    fmap1_dw16, "(b h w) t m -> (b t) m h w", t=T, h=h, w=w
                )
                fmap2_dw16 = rearrange(
                    fmap2_dw16, "(b h w) t m -> (b t) m h w", t=T, h=h, w=w
                )

            if ("self_stereo" in self.attention_type) or (
                "temporal" in self.attention_type
            ):
                for att_ind in range(self.depth):
                    if "self_stereo" in self.attention_type:
                        fmap1_dw16 = rearrange(
                            fmap1_dw16, "(b t) m h w -> (b t) (h w) m", t=T, h=h, w=w
                        )
                        fmap2_dw16 = rearrange(
                            fmap2_dw16, "(b t) m h w -> (b t) (h w) m", t=T, h=h, w=w
                        )

                        fmap1_dw16, fmap2_dw16 = self.self_attn_blocks[att_ind](
                            fmap1_dw16, fmap2_dw16
                        )
                        fmap1_dw16, fmap2_dw16 = self.cross_attn_blocks[att_ind](
                            fmap1_dw16, fmap2_dw16
                        )

                        fmap1_dw16 = rearrange(
                            fmap1_dw16, "(b t) (h w) m -> (b t) m h w ", t=T, h=h, w=w
                        )
                        fmap2_dw16 = rearrange(
                            fmap2_dw16, "(b t) (h w) m -> (b t) m h w ", t=T, h=h, w=w
                        )

                    if "temporal" in self.attention_type:
                        fmap1_dw16 = self.time_attn_blocks[att_ind](fmap1_dw16, T=T)
                        fmap2_dw16 = self.time_attn_blocks[att_ind](fmap2_dw16, T=T)
        return fmap1_dw16, fmap2_dw16

    def forward_update_block(
        self,
        update_block: nn.Module,
        corr_fn: CorrBlock1D,
        flow: torch.Tensor,
        net: torch.Tensor,
        inp: torch.Tensor,
        predictions: List,
        iters: int,
        interp_scale: float,
        t: int,
    ):
        for _ in range(iters):
            flow = flow.detach()
            out_corrs = corr_fn(flow)
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = update_block(net, inp, out_corrs, flow, t=t)

            flow = flow + delta_flow
            flow_up = flow_out = self.convex_upsample(flow, up_mask, rate=4)
            if interp_scale > 1:
                flow_up = interp_scale * interp(
                    flow_out,
                    (
                        interp_scale * flow_out.shape[2],
                        interp_scale * flow_out.shape[3],
                    ),
                )
            flow_up = flow_up[:, :1]
            predictions.append(flow_up)
        return flow_out, net

    def forward(self, image1, image2, flow_init=None, iters=10, test_mode=False):
        """Estimate optical flow between pair of frames"""
        # if input is list,
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        b, T, *_ = image1.shape

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim

        image1 = rearrange(image1, "b t c h w -> (b t) c h w")
        image2 = rearrange(image2, "b t c h w -> (b t) c h w")

        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

            net, inp = torch.split(fmap1, [hdim, hdim], dim=1)
            net = torch.tanh(net)
            inp = F.relu(inp)
            *_, h, w = fmap1.shape
            # 1/4 -> 1/16
            # feature
            fmap1_dw16 = F.avg_pool2d(fmap1, 4, stride=4)
            fmap2_dw16 = F.avg_pool2d(fmap2, 4, stride=4)

            fmap1_dw16, fmap2_dw16 = self.forward_sst_block(fmap1_dw16, fmap2_dw16, T=T)

            net_dw16, inp_dw16 = torch.split(fmap1_dw16, [hdim, hdim], dim=1)
            net_dw16 = torch.tanh(net_dw16)
            inp_dw16 = F.relu(inp_dw16)

            fmap1_dw8 = (
                F.avg_pool2d(fmap1, 2, stride=2) + interp(fmap1_dw16, (h // 2, w // 2))
            ) / 2.0
            fmap2_dw8 = (
                F.avg_pool2d(fmap2, 2, stride=2) + interp(fmap2_dw16, (h // 2, w // 2))
            ) / 2.0

            net_dw8, inp_dw8 = torch.split(fmap1_dw8, [hdim, hdim], dim=1)
            net_dw8 = torch.tanh(net_dw8)
            inp_dw8 = F.relu(inp_dw8)
        # Cascaded refinement (1/16 + 1/8 + 1/4)
        predictions = []
        flow = None
        flow_up = None
        if flow_init is not None:
            scale = h / flow_init.shape[2]
            flow = -scale * interp(flow_init, (h, w))
        else:
            # zero initialization
            flow_dw16 = self.zero_init(fmap1_dw16)

            # Recurrent Update Module
            # Update 1/16
            update_block = (
                self.update_block16
                if self.different_update_blocks
                else self.update_block
            )

            corr_fn_att_dw16 = CorrBlock1D(fmap1_dw16, fmap2_dw16)
            flow, net_dw16 = self.forward_update_block(
                update_block=update_block,
                corr_fn=corr_fn_att_dw16,
                flow=flow_dw16,
                net=net_dw16,
                inp=inp_dw16,
                predictions=predictions,
                iters=iters // 2,
                interp_scale=4,
                t=T,
            )

            scale = fmap1_dw8.shape[2] / flow.shape[2]
            flow_dw8 = -scale * interp(flow, (fmap1_dw8.shape[2], fmap1_dw8.shape[3]))

            net_dw8 = (
                net_dw8
                + interp(net_dw16, (2 * net_dw16.shape[2], 2 * net_dw16.shape[3]))
            ) / 2.0
            # Update 1/8

            update_block = (
                self.update_block08
                if self.different_update_blocks
                else self.update_block
            )

            corr_fn_dw8 = CorrBlock1D(fmap1_dw8, fmap2_dw8)
            flow, net_dw8 = self.forward_update_block(
                update_block=update_block,
                corr_fn=corr_fn_dw8,
                flow=flow_dw8,
                net=net_dw8,
                inp=inp_dw8,
                predictions=predictions,
                iters=iters // 2,
                interp_scale=2,
                t=T,
            )

            scale = h / flow.shape[2]
            flow = -scale * interp(flow, (h, w))

        net = (
            net + interp(net_dw8, (2 * net_dw8.shape[2], 2 * net_dw8.shape[3]))
        ) / 2.0
        # Update 1/4
        update_block = (
            self.update_block04 if self.different_update_blocks else self.update_block
        )
        corr_fn = CorrBlock1D(fmap1, fmap2)
        flow, __ = self.forward_update_block(
            update_block=update_block,
            corr_fn=corr_fn,
            flow=flow,
            net=net,
            inp=inp,
            predictions=predictions,
            iters=iters,
            interp_scale=1,
            t=T,
        )

        predictions = torch.stack(predictions)

        predictions = rearrange(predictions, "d (b t) c h w -> d t b c h w", b=b, t=T)
        flow_up = predictions[-1]

        if test_mode:
            return flow_up

        return predictions
