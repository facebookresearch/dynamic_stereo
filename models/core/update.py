from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_stereo.models.core.attention import LoFTREncoderLayer


#Ref: https://github.com/princeton-vl/RAFT/blob/master/core/update.py
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h
        
class SepConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU3D, self).__init__()
        self.convz1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )
        self.convr1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )
        self.convq1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )

        self.convz2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convr2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convq2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        
        self.convz3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convr3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convq3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )

    def forward(self, h, x):
        # horizontal
        # print('GRU')
        # print('horizontal')
        hx = torch.cat([h, x], dim=1)
        # print('hx',hx.shape)
        z = torch.sigmoid(self.convz1(hx))
        # print('z',z.shape)
        r = torch.sigmoid(self.convr1(hx))
        # print('r',r.shape)
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        # print('q',q.shape)
        h = (1 - z) * h + z * q
        
        # vertical
        hx = torch.cat([h, x], dim=1)
        # print('hx',hx.shape)
        z = torch.sigmoid(self.convz2(hx))
        # print('z',z.shape)
        r = torch.sigmoid(self.convr2(hx))
        # print('r',r.shape)
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        # print('q',q.shape)
        h = (1 - z) * h + z * q
        
        # time
        hx = torch.cat([h, x], dim=1)
        # print('hx',hx.shape)
        z = torch.sigmoid(self.convz3(hx))
        # print('z',z.shape)
        r = torch.sigmoid(self.convr3(hx))
        # print('r',r.shape)
        q = torch.tanh(self.convq3(torch.cat([r * h, x], dim=1)))
        # print('q',q.shape)
        h = (1 - z) * h + z * q
        # print('h',h.shape)

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super(BasicMotionEncoder, self).__init__()
        
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        

    def forward(self, x):
        B, N, C = x.shape
        qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TimeAttnBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super(TimeAttnBlock, self).__init__()
        self.temporal_attn = Attention(
            dim, num_heads=8, qkv_bias=False, qk_scale=None)
        self.temporal_fc = nn.Linear(dim, dim)
        self.temporal_norm1 = nn.LayerNorm(dim)
        
        nn.init.constant_(self.temporal_fc.weight, 0)
        nn.init.constant_(self.temporal_fc.bias, 0)
        
    def forward(self, x, T=1):
        # if self.time_attention:
        # print('fmap1.shape',fmap1.shape)
        _, _, h, w = x.shape
        
        x = rearrange(x, '(b t) m h w -> (b h w) t m',  h = h, w=w, t=T)
        # fmap2 = rearrange(fmap2, '(b t) m h w -> (b h w) t m', b=b, h = h, w=w, t=T)
        
        # if temporal_attn:
        res_temporal1 = self.temporal_attn(self.temporal_norm1(x))
        # res_temporal2 = self.temporal_attn(self.temporal_norm1(fmap2))
        
        res_temporal1 = rearrange(res_temporal1, '(b h w) t m -> b (h w t) m', h = h, w=w, t=T)
        # res_temporal2 = rearrange(res_temporal2, '(b h w) t m -> b (h w t) m', b=b, h = h, w=w, t=T)
        
        res_temporal1 = self.temporal_fc(res_temporal1)
        # res_temporal2 = self.temporal_fc(res_temporal2)
        
        res_temporal1 = rearrange(res_temporal1, ' b (h w t) m -> b t m h w', h = h, w=w, t=T)
        # res_temporal2 = rearrange(res_temporal2, ' b (h w t) m -> b t m h w', b=b, h = h, w=w, t=T)
        
        x = rearrange(x, '(b h w) t m -> b t m h w', h = h, w=w, t=T)
        # fmap2 = rearrange(fmap2, '(b h w) t m -> b t m h w', b=b, h = h, w=w, t=T)

        x = x + res_temporal1
        # fmap2 = fmap2 + res_temporal2
        # else:
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, 'b t m h w -> (b t) m h w', h = h, w=w, t=T)
        # fmap2 = rearrange(fmap2, 'b t m h w -> (b t) m h w', b=b, h = h, w=w, t=T)
        
        return x
        # , fmap2

class SpaceAttnBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super(SpaceAttnBlock, self).__init__()
        # self.space_attn = Attention(
        #     dim, num_heads=8, qkv_bias=False, qk_scale=None)
        # self.norm2 = nn.LayerNorm(dim)
        # mlp_ratio=4.
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.encoder_layer = LoFTREncoderLayer(dim, nhead=num_heads, attention='linear')
        
    def forward(self, x, T=1):
        # if self.time_attention:
        # print('fmap1.shape',fmap1.shape)
        _, _, h, w = x.shape
        x = rearrange(x, '(b t) m h w -> (b t) (h w) m',  h = h, w=w, t=T)
        x = self.encoder_layer(x, x)
        # res_space = self.space_attn(self.norm2(x))

        # x = x + res_space
        # x = x + self.mlp(self.norm2(x))
        x = rearrange(x, '(b t) (h w) m -> (b t) m h w',  h = h, w=w, t=T)
        
        return x


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8, attention_type=None):
        super(BasicUpdateBlock, self).__init__()
        self.attention_type=attention_type
        if attention_type is not None:
            # self.attn_blocks = []
            if 'update_time' in attention_type:
                self.time_attn = TimeAttnBlock(dim=256, num_heads=8)
                # self.attn_blocks.append(self.time_attn)
            if 'update_space' in attention_type:
                self.space_attn = SpaceAttnBlock(dim=256, num_heads=8)
                # self.attn_blocks.append(self.space_attn)
            # self.attn_blocks = nn.ModuleList(self.attn_blocks)

        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_size**2 *9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True, t=1):
        # print(inp.shape, corr.shape, flow.shape)
        motion_features = self.encoder(flow, corr)
        # print(motion_features.shape, inp.shape)
        inp = torch.cat((inp, motion_features), dim=1)
        if self.attention_type is not None:
            if 'update_time' in self.attention_type:
                inp = self.time_attn(inp, T=t)
            if 'update_space' in self.attention_type:
                inp = self.space_attn(inp, T=t)
            # print('attention update block!!!')
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

class FlowUpdateBlock(nn.Module):
    def __init__(self, hidden_dim, mask_size=8):
        super(FlowUpdateBlock, self).__init__()

        # self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=2+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_size**2 *9, 1, padding=0))

    def forward(self, net, inp, flow, upsample=True):
        # print(inp.shape, corr.shape, flow.shape)
        # motion_features = self.encoder(flow, corr)
        # print(motion_features.shape, inp.shape)
        inp = torch.cat((inp, flow), dim=1)

        net = self.gru(net, inp)
        flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, flow
        
class FlowHead3D(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead3D, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class SequenceUpdateBlock3D(nn.Module):
    def __init__(self,  hidden_dim, cor_planes, mask_size=8, attention_type=None):
    # args, hidden_dim=128, input_dim=128, n_downsample=3):
        super(SequenceUpdateBlock3D, self).__init__()
        # self.args = args
        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU3D(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead3D(hidden_dim, hidden_dim=256)
        # factor = 2**n_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim+128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim+128, (mask_size**2) * 9, 1, padding=0),
        )
        self.attention_type=attention_type
        if attention_type is not None:
            # self.attn_blocks = []
            if 'update_time' in attention_type:
                self.time_attn = TimeAttnBlock(dim=256, num_heads=8)
                # self.attn_blocks.append(self.time_attn)
            if 'update_space' in attention_type:
                self.space_attn = SpaceAttnBlock(dim=256, num_heads=8)
                # self.attn_blocks.append(self.space_attn)
            # self.attn_blocks = nn.ModuleList(self.attn_blocks)

    def forward(self, net, inp, corrs, flows, t, upsample=True):
        inp_tensor = []
        
        # for flow_left, corr_left, flow_right, corr_right in zip(flows_left, corrs_left, flows_right, corrs_right):
        # motion_features = self.encoder(flows, corrs)
        # print('flows',flows.shape)
        # print('net',net.shape)
        # print('inp',inp.shape)
        # print('corrs',corrs.shape)

        motion_features = self.encoder(flows, corrs)
        inp_tensor = torch.cat([inp, motion_features], dim=1)

        if self.attention_type is not None:
            if 'update_time' in self.attention_type:
                inp_tensor = self.time_attn(inp_tensor, T=t)
            if 'update_space' in self.attention_type:
                inp_tensor = self.space_attn(inp_tensor, T=t)
            # print('attention update block!!!')
        # flows = rearrange(flows, '(b t) c h w -> b c t h w', t=t)
        # corrs = rearrange(corrs, '(b t) c h w -> b c t h w', t=t)
        net = rearrange(net, '(b t) c h w -> b c t h w', t=t)
        inp_tensor = rearrange(inp_tensor, '(b t) c h w -> b c t h w', t=t)

        # motion_features_right = self.encoder(flows_right, corrs_right)
        # inp = torch.cat([motion_features_left, motion_features_right], dim=1)
        # inp_tensor = torch.stack(inp_tensor, dim=2)
        # print('inp',inp.shape)
        net = self.gru(net, inp_tensor)
        # print('net',net.shape)
        # net_split = net.split(net.shape[1]//2, dim=1)
        
        delta_flow = self.flow_head(net)
        # delta_flow_right = self.flow_head(net_split[1])

        # scale mask to balence gradients
        net = rearrange(net, ' b c t h w -> (b t) c h w')
        mask = 0.25 * self.mask(net)
        # mask_right = 0.25 * self.mask(net_split[1])
        # print('net out',net.shape)
        # print('mask',mask.shape)
        # print('delta_flow',delta_flow.shape)
        
        # mask = rearrange(mask, ' b c t h w -> (b t) c h w')
        delta_flow = rearrange(delta_flow, ' b c t h w -> (b t) c h w')
        return net, mask, delta_flow
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimo_networks.RAFT_MIMO.core.nets.attention.transformer import LoFTREncoderLayer

# from core.nets.attention.transformer import LoFTREncoderLayer

#Ref: https://github.com/princeton-vl/RAFT/blob/master/core/update.py
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size, padding=kernel_size//2)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h
        
class SepConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU3D, self).__init__()
        self.convz1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )
        self.convr1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )
        self.convq1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )

        self.convz2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convr2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convq2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        
        self.convz3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convr3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convq3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )

    def forward(self, h, x):
        # horizontal
        # print('GRU')
        # print('horizontal')
        hx = torch.cat([h, x], dim=1)
        # print('hx',hx.shape)
        z = torch.sigmoid(self.convz1(hx))
        # print('z',z.shape)
        r = torch.sigmoid(self.convr1(hx))
        # print('r',r.shape)
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        # print('q',q.shape)
        h = (1 - z) * h + z * q
        
        # vertical
        hx = torch.cat([h, x], dim=1)
        # print('hx',hx.shape)
        z = torch.sigmoid(self.convz2(hx))
        # print('z',z.shape)
        r = torch.sigmoid(self.convr2(hx))
        # print('r',r.shape)
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        # print('q',q.shape)
        h = (1 - z) * h + z * q
        
        # time
        hx = torch.cat([h, x], dim=1)
        # print('hx',hx.shape)
        z = torch.sigmoid(self.convz3(hx))
        # print('z',z.shape)
        r = torch.sigmoid(self.convr3(hx))
        # print('r',r.shape)
        q = torch.tanh(self.convq3(torch.cat([r * h, x], dim=1)))
        # print('q',q.shape)
        h = (1 - z) * h + z * q
        # print('h',h.shape)

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super(BasicMotionEncoder, self).__init__()
        
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        

    def forward(self, x):
        B, N, C = x.shape
        qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TimeAttnBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super(TimeAttnBlock, self).__init__()
        self.temporal_attn = Attention(
            dim, num_heads=8, qkv_bias=False, qk_scale=None)
        self.temporal_fc = nn.Linear(dim, dim)
        self.temporal_norm1 = nn.LayerNorm(dim)
        
        nn.init.constant_(self.temporal_fc.weight, 0)
        nn.init.constant_(self.temporal_fc.bias, 0)
        
    def forward(self, x, T=1):
        # if self.time_attention:
        # print('fmap1.shape',fmap1.shape)
        _, _, h, w = x.shape
        
        x = rearrange(x, '(b t) m h w -> (b h w) t m',  h = h, w=w, t=T)
        # fmap2 = rearrange(fmap2, '(b t) m h w -> (b h w) t m', b=b, h = h, w=w, t=T)
        
        # if temporal_attn:
        res_temporal1 = self.temporal_attn(self.temporal_norm1(x))
        # res_temporal2 = self.temporal_attn(self.temporal_norm1(fmap2))
        
        res_temporal1 = rearrange(res_temporal1, '(b h w) t m -> b (h w t) m', h = h, w=w, t=T)
        # res_temporal2 = rearrange(res_temporal2, '(b h w) t m -> b (h w t) m', b=b, h = h, w=w, t=T)
        
        res_temporal1 = self.temporal_fc(res_temporal1)
        # res_temporal2 = self.temporal_fc(res_temporal2)
        
        res_temporal1 = rearrange(res_temporal1, ' b (h w t) m -> b t m h w', h = h, w=w, t=T)
        # res_temporal2 = rearrange(res_temporal2, ' b (h w t) m -> b t m h w', b=b, h = h, w=w, t=T)
        
        x = rearrange(x, '(b h w) t m -> b t m h w', h = h, w=w, t=T)
        # fmap2 = rearrange(fmap2, '(b h w) t m -> b t m h w', b=b, h = h, w=w, t=T)

        x = x + res_temporal1
        # fmap2 = fmap2 + res_temporal2
        # else:
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, 'b t m h w -> (b t) m h w', h = h, w=w, t=T)
        # fmap2 = rearrange(fmap2, 'b t m h w -> (b t) m h w', b=b, h = h, w=w, t=T)
        
        return x
        # , fmap2

class SpaceAttnBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8):
        super(SpaceAttnBlock, self).__init__()
        # self.space_attn = Attention(
        #     dim, num_heads=8, qkv_bias=False, qk_scale=None)
        # self.norm2 = nn.LayerNorm(dim)
        # mlp_ratio=4.
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.encoder_layer = LoFTREncoderLayer(dim, nhead=num_heads, attention='linear')
        
    def forward(self, x, T=1):
        # if self.time_attention:
        # print('fmap1.shape',fmap1.shape)
        _, _, h, w = x.shape
        x = rearrange(x, '(b t) m h w -> (b t) (h w) m',  h = h, w=w, t=T)
        x = self.encoder_layer(x, x)
        # res_space = self.space_attn(self.norm2(x))

        # x = x + res_space
        # x = x + self.mlp(self.norm2(x))
        x = rearrange(x, '(b t) (h w) m -> (b t) m h w',  h = h, w=w, t=T)
        
        return x


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8, attention_type=None):
        super(BasicUpdateBlock, self).__init__()
        self.attention_type=attention_type
        if attention_type is not None:
            # self.attn_blocks = []
            if 'update_time' in attention_type:
                self.time_attn = TimeAttnBlock(dim=256, num_heads=8)
                # self.attn_blocks.append(self.time_attn)
            if 'update_space' in attention_type:
                self.space_attn = SpaceAttnBlock(dim=256, num_heads=8)
                # self.attn_blocks.append(self.space_attn)
            # self.attn_blocks = nn.ModuleList(self.attn_blocks)

        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_size**2 *9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True, t=1):
        # print(inp.shape, corr.shape, flow.shape)
        motion_features = self.encoder(flow, corr)
        # print(motion_features.shape, inp.shape)
        inp = torch.cat((inp, motion_features), dim=1)
        if self.attention_type is not None:
            if 'update_time' in self.attention_type:
                inp = self.time_attn(inp, T=t)
            if 'update_space' in self.attention_type:
                inp = self.space_attn(inp, T=t)
            # print('attention update block!!!')
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow

class FlowUpdateBlock(nn.Module):
    def __init__(self, hidden_dim, mask_size=8):
        super(FlowUpdateBlock, self).__init__()

        # self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=2+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_size**2 *9, 1, padding=0))

    def forward(self, net, inp, flow, upsample=True):
        # print(inp.shape, corr.shape, flow.shape)
        # motion_features = self.encoder(flow, corr)
        # print(motion_features.shape, inp.shape)
        inp = torch.cat((inp, flow), dim=1)

        net = self.gru(net, inp)
        flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, flow
        
class FlowHead3D(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead3D, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class SequenceUpdateBlock3D(nn.Module):
    def __init__(self,  hidden_dim, cor_planes, mask_size=8, attention_type=None):
    # args, hidden_dim=128, input_dim=128, n_downsample=3):
        super(SequenceUpdateBlock3D, self).__init__()
        # self.args = args
        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU3D(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead3D(hidden_dim, hidden_dim=256)
        # factor = 2**n_downsample
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim+128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim+128, (mask_size**2) * 9, 1, padding=0),
        )
        self.attention_type=attention_type
        if attention_type is not None:
            # self.attn_blocks = []
            if 'update_time' in attention_type:
                self.time_attn = TimeAttnBlock(dim=256, num_heads=8)
                # self.attn_blocks.append(self.time_attn)
            if 'update_space' in attention_type:
                self.space_attn = SpaceAttnBlock(dim=256, num_heads=8)
                # self.attn_blocks.append(self.space_attn)
            # self.attn_blocks = nn.ModuleList(self.attn_blocks)

    def forward(self, net, inp, corrs, flows, t, upsample=True):
        inp_tensor = []
        
        # for flow_left, corr_left, flow_right, corr_right in zip(flows_left, corrs_left, flows_right, corrs_right):
        # motion_features = self.encoder(flows, corrs)
        # print('flows',flows.shape)
        # print('net',net.shape)
        # print('inp',inp.shape)
        # print('corrs',corrs.shape)

        motion_features = self.encoder(flows, corrs)
        inp_tensor = torch.cat([inp, motion_features], dim=1)

        if self.attention_type is not None:
            if 'update_time' in self.attention_type:
                inp_tensor = self.time_attn(inp_tensor, T=t)
            if 'update_space' in self.attention_type:
                inp_tensor = self.space_attn(inp_tensor, T=t)
            # print('attention update block!!!')
        # flows = rearrange(flows, '(b t) c h w -> b c t h w', t=t)
        # corrs = rearrange(corrs, '(b t) c h w -> b c t h w', t=t)
        net = rearrange(net, '(b t) c h w -> b c t h w', t=t)
        inp_tensor = rearrange(inp_tensor, '(b t) c h w -> b c t h w', t=t)

        # motion_features_right = self.encoder(flows_right, corrs_right)
        # inp = torch.cat([motion_features_left, motion_features_right], dim=1)
        # inp_tensor = torch.stack(inp_tensor, dim=2)
        # print('inp',inp.shape)
        net = self.gru(net, inp_tensor)
        # print('net',net.shape)
        # net_split = net.split(net.shape[1]//2, dim=1)
        
        delta_flow = self.flow_head(net)
        # delta_flow_right = self.flow_head(net_split[1])

        # scale mask to balence gradients
        net = rearrange(net, ' b c t h w -> (b t) c h w')
        mask = 0.25 * self.mask(net)
        # mask_right = 0.25 * self.mask(net_split[1])
        # print('net out',net.shape)
        # print('mask',mask.shape)
        # print('delta_flow',delta_flow.shape)
        
        # mask = rearrange(mask, ' b c t h w -> (b t) c h w')
        delta_flow = rearrange(delta_flow, ' b c t h w -> (b t) c h w')
        return net, mask, delta_flow
