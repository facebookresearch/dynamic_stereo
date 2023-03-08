from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from pytorch3d.utils import opencv_from_cameras_projection



@dataclass(eq=True, frozen=True)
class PerceptionMetric:
    metric: str
    include_foreground: bool = False
    depth_scaling_norm: Optional[str] = None
    suffix: str = ""
    index: str = ""

    def __str__(self):
        return (
            self.metric
            + ("_fg" if self.include_foreground else "")
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
    return_epe: bool = False
) -> Dict[str, torch.Tensor]:

    assert len(x.shape) == len(y.shape) == len(mask.shape) == 4, (x.shape, y.shape, mask.shape)
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
    y[torch.isnan(y)]=0
    # print('y',y,'x',x)
    

    results = {}
    print('mask',mask.shape,'x',x.shape,'y',y.shape)
    for epe_name in ("epe", "temp_epe", "temp_epe_r"):
        if epe_name == "epe":
            endpoint_error = (mask * (x - y) ** 2).sum(dim=1).sqrt()
        elif epe_name == "temp_epe" or epe_name == "temp_epe_r":
            delta_mask = mask[:-1] * mask[1:]
            endpoint_error = (
                (delta_mask * ((x[:-1] - x[1:]) - (y[:-1] - y[1:])) ** 2)
                .sum(dim=1)
                .sqrt()
            )
            if epe_name == "temp_epe_r":
                endpoint_error = endpoint_error / (
                    (((y[:-1] - y[1:]) ** 2) * delta_mask).sum(dim=1).sqrt() + 1e-3
                )

        # epe_nonzero = endpoint_error != 0
        nonzero =  torch.count_nonzero(endpoint_error)

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
        if return_epe and epe_name=='epe':
            results['endpoint_error_per_pixel'] = endpoint_error
    return results



def eval_flow_trajectories(x, x_mask, y, y_mask, ref_frame, clamp_thr=1e-4):
    print('len', len(x), len(y), len(x_mask), len(y_mask))
    assert len(x) == len(y) == len(x_mask) == len(y_mask)
    results = {}
    x = x - x[ref_frame]
    y = y - y[ref_frame]

    # print('x',x.shape)

    endpoint_error = (x_mask * y_mask * (x - y) ** 2).sum(dim=1).sqrt()
    # print('endpoint_error',endpoint_error.shape)
    epe_nonzero = endpoint_error != 0

    epe_mean = endpoint_error.sum() / torch.clamp(epe_nonzero.sum(), clamp_thr)

    results = {
        "epe_traj_mean": epe_mean[None],
        "epe_frame_traj": [],
        "accuracy_10px": [],
        "accuracy_5px": [],
        "accuracy_3px": [],
        "accuracy_1px": [],
        "mean_accuracy_10px": (
            (endpoint_error[epe_nonzero] < 10).sum()
            / torch.clamp(epe_nonzero.sum(), clamp_thr)
        )[None],
        "mean_accuracy_5px": (
            (endpoint_error[epe_nonzero] < 5).sum()
            / torch.clamp(epe_nonzero.sum(), clamp_thr)
        )[None],
        "mean_accuracy_3px": (
            (endpoint_error[epe_nonzero] < 3).sum()
            / torch.clamp(epe_nonzero.sum(), clamp_thr)
        )[None],
        "mean_accuracy_1px": (
            (endpoint_error[epe_nonzero] < 1).sum()
            / torch.clamp(epe_nonzero.sum(), clamp_thr)
        )[None],
    }

    epe_mean_frames = endpoint_error.sum(1) / torch.clamp(epe_nonzero.sum(1), clamp_thr)
    for i, epe_frame_traj in enumerate(epe_mean_frames):
        results["epe_frame_traj"].append(epe_frame_traj[None])
        epe_i = endpoint_error[i]
        nonzero_epe = epe_i[epe_i != 0]

        results["accuracy_10px"].append(
            ((nonzero_epe < 10).sum() / torch.clamp((epe_i != 0).sum(), clamp_thr))[
                None
            ]
        )
        results["accuracy_5px"].append(
            ((nonzero_epe < 5).sum() / torch.clamp((epe_i != 0).sum(), clamp_thr))[None]
        )
        results["accuracy_3px"].append(
            ((nonzero_epe < 3).sum() / torch.clamp((epe_i != 0).sum(), clamp_thr))[None]
        )
        results["accuracy_1px"].append(
            ((nonzero_epe < 1).sum() / torch.clamp((epe_i != 0).sum(), clamp_thr))[None]
        )

    return results



def depth2disparity_scale(left_camera, right_camera, image_size_tensor):
    # # opencv camera matrices
    # print('image_size_tensor',image_size_tensor)
    # print('left_camera',)
    print('image_size_tensor',image_size_tensor)
    (_, T1, K1), (_, T2, _) = [
        opencv_from_cameras_projection(
            f,
            image_size_tensor,
        )
        for f in (left_camera, right_camera)
    ]
    fix_baseline = T1[0][0] - T2[0][0]
    # print('fix_baseline',fix_baseline)
    focal_length_px = K1[0][0][0]
    # print('focal_length_px',focal_length_px)
    # following this https://github.com/princeton-vl/RAFT-Stereo#converting-disparity-to-depth
    return focal_length_px * fix_baseline


# def _update_traj_interp_new(idx_y, idx_x, traj_mask, flow, flow_mask=None):
#     inds = traj_mask > 0
#     if flow_mask is not None:
#         traj_mask[inds] = (
#             traj_mask[inds] * flow_mask[idx_y[inds].long(), idx_x[inds].long()]
#         )
#     # print('flow.shape',flow.shape)
#     n, h, w = flow.shape
#     flow_incr_interp = flow

#     idx_y = idx_y + flow_incr_interp[1].reshape(h * w)
#     idx_x = idx_x + flow_incr_interp[0].reshape(h * w)
#     return idx_y, idx_x, traj_mask
def crop_batch_ims(batch_dict, crop=1):
    for k, v in batch_dict.items():
        if len(v.shape) == 4 and v.shape[2:] == batch_dict["stereo_video"].shape[3:]:
            batch_dict[k] = v[:, :, crop:-crop, crop:-crop].clone()
            # else:
            # print(k,v.shape)
    batch_dict["stereo_video"] = batch_dict["stereo_video"][
        :, :, :, crop:-crop, crop:-crop
    ].clone()
    return batch_dict


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
    #     print(img.shape)
    #     print(depth_map.shape)
    h, w, __ = img.shape
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


def _update_traj_round(idx_y, idx_x, traj_mask, flow, flow_mask=None):
    # print('flow', flow.shape)
    traj_mask = (
        traj_mask
        * (idx_x.round() > 0)
        * (idx_y.round() > 0)
        * (idx_x.round() < flow.shape[2])
        * (idx_y.round() < flow.shape[1])
    )
    
    # print('traj_mask',traj_mask.shape)
    # print('idx_y',idx_y.shape)
    # print('idx_x',idx_x.shape)
    inds = traj_mask > 0
    if flow_mask is not None:
        # print('flow_mask',flow_mask.shape)
        traj_mask[inds] = (
            traj_mask[inds]
            * flow_mask[idx_y[inds].round().long(), idx_x[inds].round().long()]
        )

    flow_incr = flow[
        :, idx_y[traj_mask > 0].round().long(), idx_x[traj_mask > 0].round().long()
    ]
    flow_incr = flow[
        :, idx_y[traj_mask > 0].round().long(), idx_x[traj_mask > 0].round().long()
    ]  # .round().long()

    idx_y[traj_mask > 0] = idx_y[traj_mask > 0] + flow_incr[1]
    idx_x[traj_mask > 0] = idx_x[traj_mask > 0] + flow_incr[0]

    return idx_y, idx_x, traj_mask


def _get_flow_trajectories(
    init_y, init_x, ref_frame, ids, gt_flow, gt_flow_mask=None, init_mode=True
):
    N = len(gt_flow)
    device = gt_flow.device
    idx_y = init_y.clone()
    idx_x = init_x.clone()
    traj_mask = torch.ones_like(idx_x)

    gt_traj, gt_traj_masks = torch.zeros((N, 2, len(ids)), device=device), torch.ones((N, 1, len(ids)), device=device)
    gt_traj[ref_frame] = torch.stack([init_y[ids], init_x[ids]], dim=0)

    # forward traj
    for i in range(ref_frame, N - 1):
        flow, flow_mask = gt_flow[i + 1], None
        if gt_flow_mask is not None:
            flow_mask = gt_flow_mask[i + 1, 0]
        if init_mode:
            idx_y, idx_x = init_y.clone(), init_x.clone()
        idx_y, idx_x, traj_mask = _update_traj_round(
            idx_y, idx_x, traj_mask, flow, flow_mask=flow_mask
        )

        gt_traj[i + 1] = torch.stack([idx_y[ids], idx_x[ids]], dim=0)
        gt_traj_masks[i + 1] = traj_mask[None, ids]

    idx_y = init_y.clone()
    idx_x = init_x.clone()
    traj_mask = torch.ones_like(idx_x)

    # backward traj
    for i in range(ref_frame - 1, -1, -1):
        flow, flow_mask = gt_flow[i], None
        if gt_flow_mask is not None:
            flow_mask = gt_flow_mask[i, 0]
        if init_mode:
            idx_y, idx_x = init_y.clone(), init_x.clone()
        idx_y, idx_x, traj_mask = _update_traj_round(
            idx_y, idx_x, traj_mask, flow, flow_mask=flow_mask
        )

        gt_traj[i] = torch.stack([idx_y[ids], idx_x[ids]], dim=0)
        gt_traj_masks[i] = traj_mask[None, ids]
    return gt_traj, gt_traj_masks


def eval_batch_mimo(
    batch_dict, predictions, ref_frame, only_foreground=False, return_epe=False
) -> Dict[str, Union[float, torch.Tensor]]:
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
    fg = only_foreground
    # for fg in [False, True]:

    if "flow_traj" in predictions:
        eval_flow_traj_output = eval_flow_trajectories(
            batch_dict["flow_traj"],
            batch_dict["flow_traj_mask"],
            predictions["flow_traj"],
            predictions["flow_traj_mask"],
            ref_frame,
        )

        results[
            PerceptionMetric("flow_epe_traj_mean", include_foreground=fg)
        ] = eval_flow_traj_output["epe_traj_mean"]
        results[
            PerceptionMetric("flow_mean_accuracy_5px", include_foreground=fg)
        ] = eval_flow_traj_output["mean_accuracy_5px"]
        results[
            PerceptionMetric("flow_mean_accuracy_3px", include_foreground=fg)
        ] = eval_flow_traj_output["mean_accuracy_3px"]
        results[
            PerceptionMetric("flow_mean_accuracy_1px", include_foreground=fg)
        ] = eval_flow_traj_output["mean_accuracy_1px"]

        for i, epe_frame_traj in enumerate(eval_flow_traj_output["epe_frame_traj"]):
            str_i = "_0" + str(i) if i < 10 else "_" + str(i)
            results[
                PerceptionMetric(
                    "flow_epe_frame_traj",
                    index=str_i,
                    include_foreground=fg,
                )
            ] = epe_frame_traj
            results[
                PerceptionMetric(
                    "flow_accuracy_1px",
                    index=str_i,
                    include_foreground=fg,
                )
            ] = eval_flow_traj_output["accuracy_1px"][i]
            results[
                PerceptionMetric(
                    "flow_accuracy_3px",
                    index=str_i,
                    include_foreground=fg,
                )
            ] = eval_flow_traj_output["accuracy_3px"][i]
            results[
                PerceptionMetric(
                    "flow_accuracy_5px",
                    index=str_i,
                    include_foreground=fg,
                )
            ] = eval_flow_traj_output["accuracy_5px"][i]
    if fg:
        mask_now = batch_dict["fg_mask"]
    else:
        mask_now = torch.ones_like(batch_dict["fg_mask"])

    if "disparity" in predictions:
        mask_now = mask_now * batch_dict["disparity_mask"]
        # print('mask_now',mask_now.min(),mask_now.max(),mask_now.shape)
        
        eval_flow_traj_output = eval_endpoint_error_sequence(
            predictions["disparity"], batch_dict["disparity"], mask_now, return_epe=return_epe
        )
        # print('eval_flow_traj_output',eval_flow_traj_output.keys())
        for epe_name in ("epe", "temp_epe", "temp_epe_r"):
            results[
                PerceptionMetric(f"disp_{epe_name}_mean", include_foreground=fg)
            ] = eval_flow_traj_output[f"{epe_name}_mean"]

            results[
                PerceptionMetric(f"disp_{epe_name}_bad_3px", include_foreground=fg)
            ] = eval_flow_traj_output[f"{epe_name}_bad_3px"]

            results[
                PerceptionMetric(f"disp_{epe_name}_bad_2px", include_foreground=fg)
            ] = eval_flow_traj_output[f"{epe_name}_bad_2px"]

            results[
                PerceptionMetric(f"disp_{epe_name}_bad_1px", include_foreground=fg)
            ] = eval_flow_traj_output[f"{epe_name}_bad_1px"]

            results[
                PerceptionMetric(f"disp_{epe_name}_bad_0.5px", include_foreground=fg)
            ] = eval_flow_traj_output[f"{epe_name}_bad_0.5px"]
        if 'endpoint_error_per_pixel' in eval_flow_traj_output:
            results["disp_endpoint_error_per_pixel"] = eval_flow_traj_output['endpoint_error_per_pixel']
    return (results, len(predictions["disparity"]))



