# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict

import torch
from tqdm import tqdm
from dynamic_stereo.evaluation.utils.eval_utils import depth2disparity_scale, eval_batch_mimo

from dynamic_stereo.evaluation.core.base_evaluator import BaseEvaluator
from dynamic_stereo.evaluation.utils.utils import PerceptionPrediction, pretty_print_perception_metrics, visualize_batch_mimo

import sys

sys.path.append('/private/home/nikitakaraev/dev/pixar_replay/')


# @Configurable
class MIMOEvaluatorSimplified(BaseEvaluator):
    """
    A class defining the NVS evaluator.

    Args:
        bg_color: The background color of the generated new views and the
            ground truth.
        mask_thr: The threshold to use to make ground truth binary masks.
    """

    eps = 1e-5

    @torch.no_grad()
    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        only_foreground: bool = False,
        is_real_data: bool = False,
        step=None,
        writer=None,
        train_mode=False,
        visualize_errors=False,
        save_epe_thresholds=False
    ):
        model.eval()
        per_batch_eval_results = []

        if self.visualize_interval > 0:
            os.makedirs(self.visualize_dir, exist_ok=True)
            
        if save_epe_thresholds:
            epe_thresholds = [0]*100
        for batch_idx, sequence in enumerate(tqdm(test_dataloader)):
            batch_dict = defaultdict(list)
            batch_dict["stereo_video"] = sequence['img']
            if not is_real_data:
                batch_dict["disparity"] = sequence['disp'][:,0].abs()
                batch_dict["disparity_mask"] = sequence['valid_disp'][:,:1]
                
                if 'mask' in sequence:
                    batch_dict['fg_mask'] = sequence['mask'][:,:1]
                else:
                    batch_dict['fg_mask'] = torch.ones_like(batch_dict["disparity_mask"])
                    
            if train_mode:
                predictions = model.forward_batch_test(batch_dict)
            else:
                predictions = model(batch_dict)
                
            if "disparity" in predictions:
                predictions["disparity"] = predictions["disparity"][:, :1].clone().cpu()
                
                if not is_real_data:
                    predictions["disparity"] = predictions["disparity"]*(batch_dict["disparity_mask"].round())

            if "flow" in predictions:
                raise ValueError('flow isn\'t supported')

            ref_frame = 0
            if not is_real_data:
                batch_eval_result, seq_length = eval_batch_mimo(
                    batch_dict,
                    predictions,
                    ref_frame=ref_frame,
                    only_foreground=only_foreground,
                    return_epe=visualize_errors or save_epe_thresholds
                    
                )
                if visualize_errors or save_epe_thresholds:
                    # assert 'disp_endpoint_error_per_pixel' in batch_eval_result
                    disp_endpoint_error_per_pixel = batch_eval_result['disp_endpoint_error_per_pixel']
                    del batch_eval_result['disp_endpoint_error_per_pixel']
                    nonzero =  torch.count_nonzero(disp_endpoint_error_per_pixel)
                    if save_epe_thresholds:
                        for bad_px in range(1, 100):
                            epe_thresholds[bad_px] += ((disp_endpoint_error_per_pixel > (bad_px/10.)).sum() / torch.clamp( nonzero, 1e-5)).item()
                per_batch_eval_results.append((batch_eval_result, seq_length))
                pretty_print_perception_metrics(batch_eval_result)

            if (self.visualize_interval > 0) and (
                batch_idx % self.visualize_interval == 0
            ):
                perception_prediction = PerceptionPrediction()
                if "disparity" in predictions:
                    pred_disp = predictions["disparity"]
                    print('pred_disp',pred_disp.shape)
                    pred_disp[pred_disp < self.eps] = self.eps

                    scale = depth2disparity_scale(
                        sequence['viewpoint'][0][0],
                        sequence['viewpoint'][0][1],
                        torch.tensor([pred_disp.shape[2],pred_disp.shape[3]])[None]
                    )

                    perception_prediction.depth_map = (scale / pred_disp).cuda()
                    perspective_cameras = []
                    for cam in sequence['viewpoint']:
                        perspective_cameras.append(cam[0])
                        
                    perception_prediction.perspective_cameras = perspective_cameras


                if "stereo_original_video" in batch_dict:
                    batch_dict["stereo_video"] = (
                        batch_dict["stereo_original_video"]
                        .clone()
                    )
                    print('vis vid',batch_dict["stereo_video"].shape)

                for k,v in batch_dict.items():
                    if isinstance(v, torch.Tensor):
                        batch_dict[k] = v.cuda()

                outputs = visualize_batch_mimo(
                    batch_dict,
                    ref_frame,
                    perception_prediction,
                    self.visualize_dir,
                    only_foreground=only_foreground,
                    sequence_name=sequence['metadata'][0][0][0],
                    step=step,
                    writer=writer,
                    disp_endpoint_error_per_pixel=disp_endpoint_error_per_pixel if visualize_errors else None
                )
        if save_epe_thresholds:
            for bad_px in range(1, 100):
                epe_thresholds[bad_px] = epe_thresholds[bad_px] / float(len(test_dataloader))
            torch.save(torch.tensor(epe_thresholds), os.path.join(self.exp_dir,'epe_thresholds.pth'))
        return per_batch_eval_results
