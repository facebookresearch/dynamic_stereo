from typing import ClassVar

import torch
from pytorch3d.implicitron.tools.config import Configurable
from mimo_networks.RAFT_MIMO.core.nets.dynamic_stereo import DynamicStereo

autocast = torch.cuda.amp.autocast


class DynamicStereoModel(Configurable, torch.nn.Module):

    MODEL_CONFIG_NAME: ClassVar[str] = "DynamicStereoModel"

    model_weights: str = '/checkpoint/nikitakaraev/2022_mimo/experiments/2022_11_02_DynamicStereo_120k_sample_len_5_update_3d_different_blocks_self_cross_temporal_update_time_update_space/model_raft-stereo_073729.pth'
    kernel_size: int = 20

    def __post_init__(self):
        super().__init__()

        self.mixed_precision = False
        model = DynamicStereo(
            mixed_precision = self.mixed_precision,
            num_frames=5, 
            attention_type='self_cross_temporal_update_time_update_space',
            use_3d_update_block=True,
            different_3d_update_blocks=True)

        state_dict = torch.load(self.model_weights, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            state_dict = {'module.'+k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        self.model = model
        self.model.to("cuda")
        self.model.eval()


    def forward(self, batch_dict, iters=20):
        return self.model.forward_batch_test_cut_preds(
            batch_dict,
            mixed_prec=self.mixed_precision,
            flow_frame_list=None,
            kernel_size=self.kernel_size,
            iters=iters)
    