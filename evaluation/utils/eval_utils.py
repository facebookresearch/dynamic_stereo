# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from pytorch3d.utils import opencv_from_cameras_projection


@dataclass(eq=True, frozen=True)
class PerceptionMetric:
    metric: str
    depth_scaling_norm: Optional[str] = None
    suffix: str = ""
    index: str = ""

    def __str__(self):
        return (
            self.metric
            + self.index
            + (
                ("_norm_" + self.depth_scaling_norm)
                if self.depth_scaling_norm is not None
                else ""
            )
            + self.suffix
        )


def eval_endpoint_error_sequence(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    crop: int = 0,
    mask_thr: float = 0.5,
    clamp_thr: float = 1e-5,
) -> Dict[str, torch.Tensor]:

    assert len(x.shape) == len(y.shape) == len(mask.shape) == 4, (
        x.shape,
        y.shape,
        mask.shape,
    )
    assert x.shape[0] == y.shape[0] == mask.shape[0], (x.shape, y.shape, mask.shape)

    # chuck out the border
    if crop > 0:
        if crop > min(y.shape[2:]) - crop:
            raise ValueError("Incorrect crop size.")
        y = y[:, :, crop:-crop, crop:-crop]
        x = x[:, :, crop:-crop, crop:-crop]
        mask = mask[:, :, crop:-crop, crop:-crop]

    y = y * (mask > mask_thr).float()
    x = x * (mask > mask_thr).float()
    y[torch.isnan(y)] = 0

    results = {}
    for epe_name in ("epe", "temp_epe"):
        if epe_name == "epe":
            endpoint_error = (mask * (x - y) ** 2).sum(dim=1).sqrt()
        elif epe_name == "temp_epe":
            delta_mask = mask[:-1] * mask[1:]
            endpoint_error = (
                (delta_mask * ((x[:-1] - x[1:]) - (y[:-1] - y[1:])) ** 2)
                .sum(dim=1)
                .sqrt()
            )

        # epe_nonzero = endpoint_error != 0
        nonzero = torch.count_nonzero(endpoint_error)

        epe_mean = endpoint_error.sum() / torch.clamp(
            nonzero, clamp_thr
        )  # average error for all the sequence pixels
        epe_inv_accuracy_05px = (endpoint_error > 0.5).sum() / torch.clamp(
            nonzero, clamp_thr
        )
        epe_inv_accuracy_1px = (endpoint_error > 1).sum() / torch.clamp(
            nonzero, clamp_thr
        )
        epe_inv_accuracy_2px = (endpoint_error > 2).sum() / torch.clamp(
            nonzero, clamp_thr
        )
        epe_inv_accuracy_3px = (endpoint_error > 3).sum() / torch.clamp(
            nonzero, clamp_thr
        )

        results[f"{epe_name}_mean"] = epe_mean[None]
        results[f"{epe_name}_bad_0.5px"] = epe_inv_accuracy_05px[None] * 100
        results[f"{epe_name}_bad_1px"] = epe_inv_accuracy_1px[None] * 100
        results[f"{epe_name}_bad_2px"] = epe_inv_accuracy_2px[None] * 100
        results[f"{epe_name}_bad_3px"] = epe_inv_accuracy_3px[None] * 100
    return results


def depth2disparity_scale(left_camera, right_camera, image_size_tensor):
    # # opencv camera matrices
    (_, T1, K1), (_, T2, _) = [
        opencv_from_cameras_projection(
            f,
            image_size_tensor,
        )
        for f in (left_camera, right_camera)
    ]
    fix_baseline = T1[0][0] - T2[0][0]
    focal_length_px = K1[0][0][0]
    # following this https://github.com/princeton-vl/RAFT-Stereo#converting-disparity-to-depth
    return focal_length_px * fix_baseline


def depth_to_pcd(
    depth_map,
    img,
    focal_length,
    cx,
    cy,
    step: int = None,
    inv_extrinsic=None,
    mask=None,
    filter=False,
):
    __, w, __ = img.shape
    if step is None:
        step = int(w / 100)
    Z = depth_map[::step, ::step]
    colors = img[::step, ::step, :]

    Pixels_Y = torch.arange(Z.shape[0]).to(Z.device) * step
    Pixels_X = torch.arange(Z.shape[1]).to(Z.device) * step

    X = (Pixels_X[None, :] - cx) * Z / focal_length
    Y = (Pixels_Y[:, None] - cy) * Z / focal_length

    inds = Z > 0

    if mask is not None:
        inds = inds * (mask[::step, ::step] > 0)

    X = X[inds].reshape(-1)
    Y = Y[inds].reshape(-1)
    Z = Z[inds].reshape(-1)
    colors = colors[inds]
    pcd = torch.stack([X, Y, Z]).T

    if inv_extrinsic is not None:
        pcd_ext = torch.vstack([pcd.T, torch.ones((1, pcd.shape[0])).to(Z.device)])
        pcd = (inv_extrinsic @ pcd_ext)[:3, :].T

    if filter:
        pcd, filt_inds = filter_outliers(pcd)
        colors = colors[filt_inds]
    return pcd, colors


def filter_outliers(pcd, sigma=3):
    mean = pcd.mean(0)
    std = pcd.std(0)
    inds = ((pcd - mean).abs() < sigma * std)[:, 2]
    pcd = pcd[inds]
    return pcd, inds


def eval_batch(batch_dict, predictions) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Produce performance metrics for a single batch of perception
    predictions.
    Args:
        frame_data: A PixarFrameData object containing the input to the new view
            synthesis method.
        preds: A PerceptionPrediction object with the predicted data.
    Returns:
        results: A dictionary holding evaluation metrics.
    """
    results = {}

    assert "disparity" in predictions
    mask_now = torch.ones_like(batch_dict["fg_mask"])

    mask_now = mask_now * batch_dict["disparity_mask"]

    eval_flow_traj_output = eval_endpoint_error_sequence(
        predictions["disparity"], batch_dict["disparity"], mask_now
    )
    for epe_name in ("epe", "temp_epe"):
        results[PerceptionMetric(f"disp_{epe_name}_mean")] = eval_flow_traj_output[
            f"{epe_name}_mean"
        ]

        results[PerceptionMetric(f"disp_{epe_name}_bad_3px")] = eval_flow_traj_output[
            f"{epe_name}_bad_3px"
        ]

        results[PerceptionMetric(f"disp_{epe_name}_bad_2px")] = eval_flow_traj_output[
            f"{epe_name}_bad_2px"
        ]

        results[PerceptionMetric(f"disp_{epe_name}_bad_1px")] = eval_flow_traj_output[
            f"{epe_name}_bad_1px"
        ]

        results[PerceptionMetric(f"disp_{epe_name}_bad_0.5px")] = eval_flow_traj_output[
            f"{epe_name}_bad_0.5px"
        ]
    if "endpoint_error_per_pixel" in eval_flow_traj_output:
        results["disp_endpoint_error_per_pixel"] = eval_flow_traj_output[
            "endpoint_error_per_pixel"
        ]
    return (results, len(predictions["disparity"]))
