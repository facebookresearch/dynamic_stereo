# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from types import SimpleNamespace
from typing import ClassVar

import torch
from pytorch3d.implicitron.tools.config import Configurable

import importlib
import sys

sys.path.append("third_party/RAFT-Stereo")
raft_stereo = importlib.import_module(
    "dynamic_stereo.third_party.RAFT-Stereo.core.raft_stereo"
)
raft_stereo_utils = importlib.import_module(
    "dynamic_stereo.third_party.RAFT-Stereo.core.utils.utils"
)
autocast = torch.cuda.amp.autocast


class RAFTStereoModel(Configurable, torch.nn.Module):
    MODEL_CONFIG_NAME: ClassVar[str] = "RAFTStereoModel"
    model_weights: str = "./third_party/RAFT-Stereo/models/raftstereo-middlebury.pth"

    def __post_init__(self):
        super().__init__()

        model_args = SimpleNamespace(
            hidden_dims=[128] * 3,
            corr_implementation="reg",
            shared_backbone=False,
            corr_levels=4,
            corr_radius=4,
            n_downsample=2,
            slow_fast_gru=False,
            n_gru_layers=3,
            mixed_precision=False,
            context_norm="batch",
        )
        self.args = model_args
        model = torch.nn.DataParallel(
            raft_stereo.RAFTStereo(model_args), device_ids=[0]
        )

        state_dict = torch.load(self.model_weights, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        self.model = model.module
        self.model.to("cuda")
        self.model.eval()

    def forward(self, batch_dict, iters=32):
        predictions = defaultdict(list)
        for stereo_pair in batch_dict["stereo_video"]:
            left_image_rgb = stereo_pair[None, 0].cuda()
            right_image_rgb = stereo_pair[None, 1].cuda()

            padder = raft_stereo_utils.InputPadder(left_image_rgb.shape, divis_by=32)
            left_image_rgb, right_image_rgb = padder.pad(
                left_image_rgb, right_image_rgb
            )

            with autocast(enabled=self.args.mixed_precision):
                _, flow_up = self.model.forward(
                    left_image_rgb,
                    right_image_rgb,
                    iters=iters,
                    test_mode=True,
                )
            flow_up = padder.unpad(flow_up)
            predictions["disparity"].append(flow_up)
        predictions["disparity"] = (
            torch.stack(predictions["disparity"]).squeeze(1).abs()
        )
        return predictions
