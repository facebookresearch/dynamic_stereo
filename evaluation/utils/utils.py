
from collections import defaultdict
import configparser
import os
import math
from typing import Any, Dict, Optional, Tuple, Union, List
import torch
import cv2
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tabulate import tabulate
# from pytorch3d.implicitron.tools.point_cloud_utils import render_point_cloud_pytorch3d
# from pytorch3d.implicitron.tools.video_writer import VideoWriter
# from pytorch3d.implicitron.tools.vis_utils import (
#     get_visdom_connection,
#     make_depth_image,
# )
from PIL import Image, ImageDraw
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVOrthographicCameras,
    NormWeightedCompositor,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PulsarPointsRenderer,
    look_at_view_transform,
)
# from pytorch3d.renderer.cameras import CamerasBase, look_at_view_transform

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.utils import (
    # cameras_from_opencv_projection,
    opencv_from_cameras_projection,
)
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# from pixar_replay.core.evaluation.eval_perception_utils import PerceptionPrediction
# from pixar_replay.experimental.scripts.mimo_utils import (
   
#     # depth2disparity_scale,
#     depth_to_pcd
# )

@dataclass
class PerceptionPrediction:
    """
    Holds the tensors that describe a result of any perception module.
    """

    depth_map: Optional[torch.Tensor] = None
    image_rgb: Optional[torch.Tensor] = None
    fg_probability: Optional[torch.Tensor] = None
    pred_flow_traj: Optional[torch.Tensor] = None
    pred_flow_traj_masks: Optional[torch.Tensor] = None
    


# def aggregate_perception_eval_results(
#     per_batch_eval_results: List[Dict[str, Any]],
# ) -> Dict[str, Any]:
#     """
#     Compile the per-batch evaluation results `per_batch_eval_results` into
#     a set of aggregate metrics. The produced metrics depend on the task.
#     Args:
#         per_batch_eval_results: Metrics of each per-batch evaluation.
#     Returns:
#         aggregate_results: A flattened dict of all aggregate metrics.
#     """

#     aggregate_results = defaultdict(list)
#     for result in per_batch_eval_results:
#         for metric, val in result.items():
#             aggregate_results[metric].append(val)

#     return {k: torch.cat(v).mean().item() for k, v in aggregate_results.items()}

def aggregate_eval_results(
    per_batch_eval_results, reduction = 'mean'
) :
    
    total_length=0
    aggregate_results = defaultdict(list)
    for result in per_batch_eval_results:
        if isinstance(result, tuple):
            reduction = 'sum'
            length = result[1]
            total_length+=length
            result = result[0]
        for metric, val in result.items():
            if reduction == 'sum':
                aggregate_results[metric].append(val*length)
            
    if reduction=='mean':
        return {k: torch.cat(v).mean().item() for k, v in aggregate_results.items()}
    elif reduction=='sum':
        return {k: torch.cat(v).sum().item() / float(total_length) for k, v in aggregate_results.items()}


def aggregate_and_print_results(
    per_batch_eval_results: List[dict],
):
    print("")
    result = aggregate_eval_results(
            per_batch_eval_results,
        )
    pretty_print_perception_metrics(result)
    result = {str(k): v for k, v in result.items()}
    
    print("")
    return result


def pretty_print_perception_metrics(results):

    metrics = sorted(
        list(results.keys()), key=lambda x: (x.metric, x.include_foreground)
    )

    print("===== Perception results =====")
    print(
        tabulate(
            [[metric, results[metric]] for metric in metrics],
        )
    )
@torch.no_grad()
def _draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 2,
    width: int = 3,
) -> torch.Tensor:

    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.
    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.
    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        raise ValueError("Pass an RGB image. Other Image formats are not supported")

    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()
    
    for kpt_id, kpt_inst in enumerate(img_kpts):
        # print('img_kpts',len(kpt_inst))
        for inst_id, kpt in enumerate(kpt_inst):
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            if isinstance(colors, torch.Tensor):
                color = tuple(colors[inst_id].numpy())
            else:
                color = colors
            # print('color',color)
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )

    return (
        torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)
    )
# @torch.no_grad()
# def _draw_keypoints(
#     image: torch.Tensor,
#     keypoints: torch.Tensor,
#     connectivity: Optional[List[Tuple[int, int]]] = None,
#     colors: Optional[Union[str, Tuple[int, int, int]]] = None,
#     radius: int = 2,
#     width: int = 3,
# ) -> torch.Tensor:

#     """
#     Draws Keypoints on given RGB image.
#     The values of the input image should be uint8 between 0 and 255.
#     Args:
#         image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
#         keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
#             in the format [x, y].
#         connectivity (List[Tuple[int, int]]]): A List of tuple where,
#             each tuple contains pair of keypoints to be connected.
#         colors (str, Tuple): The color can be represented as
#             PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
#         radius (int): Integer denoting radius of keypoint.
#         width (int): Integer denoting width of line connecting keypoints.
#     Returns:
#         img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
#     """

#     if not isinstance(image, torch.Tensor):
#         raise TypeError(f"The image must be a tensor, got {type(image)}")
#     elif image.dtype != torch.uint8:
#         raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
#     elif image.dim() != 3:
#         raise ValueError("Pass individual images, not batches")
#     elif image.size()[0] != 3:
#         raise ValueError("Pass an RGB image. Other Image formats are not supported")

#     if keypoints.ndim != 3:
#         raise ValueError("keypoints must be of shape (num_instances, K, 2)")

#     ndarr = image.permute(1, 2, 0).cpu().numpy()
#     img_to_draw = Image.fromarray(ndarr)
#     draw = ImageDraw.Draw(img_to_draw)
#     img_kpts = keypoints.to(torch.int64).tolist()

#     for kpt_id, kpt_inst in enumerate(img_kpts):
#         for inst_id, kpt in enumerate(kpt_inst):
#             x1 = kpt[0] - radius
#             x2 = kpt[0] + radius
#             y1 = kpt[1] - radius
#             y2 = kpt[1] + radius
#             if isinstance(colors, list) or isinstance(colors, np.array):
#                 color = colors[kpt_id]
#             else:
#                 color = colors
#             draw.ellipse([x1, y1, x2, y2], fill=color, outline=None, width=0)

#         if connectivity:
#             for connection in connectivity:
#                 start_pt_x = kpt_inst[connection[0]][0]
#                 start_pt_y = kpt_inst[connection[0]][1]

#                 end_pt_x = kpt_inst[connection[1]][0]
#                 end_pt_y = kpt_inst[connection[1]][1]

#                 draw.line(
#                     ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
#                     width=width,
#                 )

#     return (
#         torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)
#     )


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

def read_calibration(calibration_file, resolution_string):
        # ported from https://github.com/stereolabs/zed-open-capture/
        # blob/dfa0aee51ccd2297782230a05ca59e697df496b2/examples/include/calibration.hpp#L4172

        zed_resolutions = {
            "2K": (1242, 2208),
            "FHD": (1080, 1920),
            "HD": (720, 1280),
            # "qHD": (540, 960),
            "VGA": (376, 672),
        }
        assert resolution_string in zed_resolutions.keys()
        image_height, image_width = zed_resolutions[resolution_string]

        # Open camera configuration file
        assert os.path.isfile(calibration_file)
        calib = configparser.ConfigParser()
        calib.read(calibration_file)

        # Get translations
        T = np.zeros((3, 1))
        T[0] = float(calib["STEREO"]["baseline"])
        T[1] = float(calib["STEREO"]["ty"])
        T[2] = float(calib["STEREO"]["tz"])

        baseline = T[0]

        # Get left parameters
        left_cam_cx = float(calib[f"LEFT_CAM_{resolution_string}"]["cx"])
        left_cam_cy = float(calib[f"LEFT_CAM_{resolution_string}"]["cy"])
        left_cam_fx = float(calib[f"LEFT_CAM_{resolution_string}"]["fx"])
        left_cam_fy = float(calib[f"LEFT_CAM_{resolution_string}"]["fy"])
        left_cam_k1 = float(calib[f"LEFT_CAM_{resolution_string}"]["k1"])
        left_cam_k2 = float(calib[f"LEFT_CAM_{resolution_string}"]["k2"])
        left_cam_p1 = float(calib[f"LEFT_CAM_{resolution_string}"]["p1"])
        left_cam_p2 = float(calib[f"LEFT_CAM_{resolution_string}"]["p2"])
        left_cam_k3 = float(calib[f"LEFT_CAM_{resolution_string}"]["k3"])

        # Get right parameters
        right_cam_cx = float(calib[f"RIGHT_CAM_{resolution_string}"]["cx"])
        right_cam_cy = float(calib[f"RIGHT_CAM_{resolution_string}"]["cy"])
        right_cam_fx = float(calib[f"RIGHT_CAM_{resolution_string}"]["fx"])
        right_cam_fy = float(calib[f"RIGHT_CAM_{resolution_string}"]["fy"])
        right_cam_k1 = float(calib[f"RIGHT_CAM_{resolution_string}"]["k1"])
        right_cam_k2 = float(calib[f"RIGHT_CAM_{resolution_string}"]["k2"])
        right_cam_p1 = float(calib[f"RIGHT_CAM_{resolution_string}"]["p1"])
        right_cam_p2 = float(calib[f"RIGHT_CAM_{resolution_string}"]["p2"])
        right_cam_k3 = float(calib[f"RIGHT_CAM_{resolution_string}"]["k3"])

        # Get rotations
        R_zed = np.zeros(3)
        R_zed[0] = float(calib["STEREO"][f"rx_{resolution_string.lower()}"])
        R_zed[1] = float(calib["STEREO"][f"cv_{resolution_string.lower()}"])
        R_zed[2] = float(calib["STEREO"][f"rz_{resolution_string.lower()}"])

        R = cv2.Rodrigues(R_zed)[0]

        # Left
        cameraMatrix_left = np.array(
            [[left_cam_fx, 0, left_cam_cx], [0, left_cam_fy, left_cam_cy], [0, 0, 1]]
        )
        distCoeffs_left = np.array(
            [left_cam_k1, left_cam_k2, left_cam_p1, left_cam_p2, left_cam_k3]
        )

        # Right
        cameraMatrix_right = np.array(
            [
                [right_cam_fx, 0, right_cam_cx],
                [0, right_cam_fy, right_cam_cy],
                [0, 0, 1],
            ]
        )
        distCoeffs_right = np.array(
            [right_cam_k1, right_cam_k2, right_cam_p1, right_cam_p2, right_cam_k3]
        )

        # Stereo
        R1, R2, P1, P2, Q = cv2.stereoRectify(
            cameraMatrix1=cameraMatrix_left,
            distCoeffs1=distCoeffs_left,
            cameraMatrix2=cameraMatrix_right,
            distCoeffs2=distCoeffs_right,
            imageSize=(image_width, image_height),
            R=R,
            T=T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=(image_width, image_height),
            alpha=0,
        )[:5]

        # Precompute maps for cv::remap()
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            cameraMatrix_left,
            distCoeffs_left,
            R1,
            P1,
            (image_width, image_height),
            cv2.CV_32FC1,
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            cameraMatrix_right,
            distCoeffs_right,
            R2,
            P2,
            (image_width, image_height),
            cv2.CV_32FC1,
        )

        zed_calib = {
            "map_left_x": map_left_x,
            "map_left_y": map_left_y,
            "map_right_x": map_right_x,
            "map_right_y": map_right_y,
            "pose_left": P1,
            "pose_right": P2,
            "baseline": baseline,
            "image_width": image_width,
            "image_height": image_height,
        }

        return zed_calib
        
def visualize_batch_mimo(
    batch_dict: dict,
    ref_frame,
    preds: PerceptionPrediction,
    output_dir: str,
    # viz: Visdom,
    # viz_env="debug",
    only_foreground=False,
    # visdom_show_preds=False,
    step=0,
    sequence_name=None,
    frame_number=None,
    gradient=False,
    writer=None,
    disp_endpoint_error_per_pixel=None
):
    os.makedirs(output_dir, exist_ok=True)
    if disp_endpoint_error_per_pixel is not None:
        cmap_error = plt.cm.get_cmap('plasma')
    outputs = {}
    
    # fg = 'fg' if only_foreground else 'bg'
    
    if preds.depth_map is not None:
        device = preds.depth_map.device

        pcd_global_seq = []
        H, W = batch_dict["stereo_video"].shape[3:]

        for i in range(len(batch_dict["stereo_video"])):
            R, T, K = opencv_from_cameras_projection(
                preds.perspective_cameras[i],
                torch.tensor([H, W])[None].to(device),
            )
            # colors = batch_dict["stereo_video"][i][0].permute(1,2,0)
            extrinsic_3x4_0 = torch.cat([R[0], T[0, :, None]], dim=1)
            # .to(device)

            extr_matrix = torch.cat(
                [extrinsic_3x4_0, torch.Tensor([[0, 0, 0, 1]]).to(extrinsic_3x4_0.device)], dim=0
            )
            # #     #seq_transformed = (s[0], s[1], extr_matrix)
            #     inv_extr_matrix = extr_matrix.inverse()
            inv_extr_matrix = extr_matrix.inverse().to(device)
            pcd, colors = depth_to_pcd(
                # preds.perspective_cameras[i,0],
                # batch_dict["depth"][i,0],
                preds.depth_map[i, 0],
                batch_dict["stereo_video"][i][0].permute(1, 2, 0),
                K[0][0][0],
                K[0][0][2],
                K[0][1][2],
                step=1,
                inv_extrinsic=inv_extr_matrix,
                mask=batch_dict["fg_mask"][i, 0] if only_foreground else None,
                filter=False,
            )

            R, T = inv_extr_matrix[None, :3, :3], inv_extr_matrix[None, :3, 3]
            pcd_global_seq.append((pcd, colors, (R, T, preds.perspective_cameras[i])))

        raster_settings = PointsRasterizationSettings(
            image_size=[H, W], radius=0.003, points_per_pixel=10
        )
        R, T, cam_ = pcd_global_seq[ref_frame][2]
        # R_[0][0]
        # median_depth = pcd_global_seq[ref_frame][0][:, 2].median()
        median_depth = preds.depth_map.median()
        cam_.cuda()

        if disp_endpoint_error_per_pixel is not None and not only_foreground:
            seq_error = disp_endpoint_error_per_pixel.flatten(1).cuda()
            # print('seq_error',seq_error.shape)
            epe_less_5 = (seq_error < 2.)[:,:,None] * torch.stack([torch.ones_like(seq_error), 1.-(seq_error / 2.), torch.zeros_like(seq_error)], dim=2)
            epe_more_5 = (seq_error >= 2.)[:,:,None] * (torch.ones_like(seq_error)[:,:,None]).cuda() * (torch.tensor([1.,0.,0.])[None,None]).cuda()
            epe = (epe_less_5 + epe_more_5).cuda()


        for mode in ["angle_15", "angle_-15", "changing_angle"]:
            res = []

            
            for t, (pcd, color, __) in enumerate(pcd_global_seq):

                if mode == "changing_angle":
                    angle = math.cos((math.pi) * (t / 15)) * 15
                elif mode == "angle_15":
                    angle = 15
                elif mode == "angle_-15":
                    angle = -15
                # print('angle',angle)
                delta_x = median_depth*math.sin(math.radians(angle))
                # # delta_y = median_depth * math.sin(math.radians(angle))
                delta_z = median_depth*(1 - math.cos(math.radians(angle)))

                # R_, T_ = R.clone(), T.clone()
                # R_ = torch.bmm(
                #     R_,
                #     RotateAxisAngle(angle=-angle, axis="Y", device=device).get_matrix()[
                #         :, :3, :3
                #     ],
                # )
                # T_[0, 0] = T_[0, 0] + delta_x
                # T_[0, 2] = T_[0, 2] - delta_z

                cam = cam_.clone()
                cam.R = torch.bmm(
                    cam.R,
                    RotateAxisAngle(angle=angle, axis="Y", device=device).get_matrix()[
                        :, :3, :3
                    ],
                )
                cam.T[0,0] = cam.T[0,0]-delta_x
                cam.T[0,2] = cam.T[0,2]-delta_z+median_depth/2.
                # PerspectiveCameras(
                #     focal_length=((-K[0][0][0], -K[0][0][0]),),
                #     principal_point=(
                #         (
                #             K[0][0][2],
                #             K[0][1][2],
                #         ),
                #     ),
                #     R=R_,
                #     T=T_,
                #     in_ndc=False,
                #     image_size=((H, W),),
                #     device=device,
                # )

                rasterizer = PointsRasterizer(
                    cameras=cam, raster_settings=raster_settings
                )
                renderer = PointsRenderer(
                    rasterizer=rasterizer, compositor=AlphaCompositor(background_color=(1,1,1))
                )
                pcd_copy = pcd.clone()
                # print('pcd[:,2]',pcd[:,2].min(),pcd[:,2].max(),pcd[:,2].mean())
                # pcd_copy[:,2] = 0.8
                if disp_endpoint_error_per_pixel is not None and not only_foreground:
                    # frame_error = disp_endpoint_error_per_pixel[t].flatten()
                    # frame_error = frame_error.cpu()
                    # # print('(cmap_error(frame_error / 5.))',(cmap_error(frame_error / 5.)).shape, (cmap_error(frame_error / 5.)))
                    # # print('(frame_error < 5.)[:,None]',(frame_error < 5.)[:,None].shape,(frame_error < 5.)[:,None])
                    # # print('cmap_error(torch.ones_like(frame_error))[:,None]',cmap_error(torch.ones_like(frame_error)).shape,cmap_error(torch.ones_like(frame_error))[:,None])
                    # print('torch.tensor([1.,0.,0.])[:,None]',torch.tensor([1.,0.,0.])[:,None].shape,torch.tensor([1.,0.,0.])[:,None])
                    # print('torch.ones_like(frame_error)',torch.ones_like(frame_error).shape)

                    
                    color = 0.4 * color + 0.6 * epe[t,:,:3] * 255.
                    # print('disp_endpoint_error_per_pixel', disp_endpoint_error_per_pixel.shape,disp_endpoint_error_per_pixel.min(),disp_endpoint_error_per_pixel.max())
                    # print('color', color.shape,color.min(),color.max())
                    
                point_cloud = Pointclouds(points=[pcd_copy], features=[color / 255.0])
                images = renderer(point_cloud)
                res.append(images[0, ..., :3].cpu())
            res = torch.stack(res)
            print('res',res.shape,type(res),res.max(),res.min(),(res * 255).to(torch.uint8).max(),(res * 255).to(torch.uint8).min())
        
            
            video = (res * 255).numpy().astype(np.uint8)
            save_name = f"{sequence_name}_reconstruction_{step}_mode_{mode}_"
            if writer is None:
                outputs[mode] = video
            if only_foreground:
                save_name += "fg_only"
            else:
                save_name += "full_scene"
            video_out = cv2.VideoWriter(
                      os.path.join(
                            output_dir,
                            f"{save_name}.mp4",
                        ),
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps = 10,
                      frameSize=(res.shape[2],res.shape[1]),
                      isColor=True)
            # print('(res.shape[1],res.shape[2])',(res.shape[2],res.shape[1]))
            for i in range(len(video)):
                # data = np.random.randint(0, 256, size, dtype='uint8')
                video_out.write(cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB))
            video_out.release()
            
            if writer is not None:
                writer.add_video(
                    f"{sequence_name}_reconstruction_mode_{mode}",
                    (res * 255).permute(0, 3, 1, 2).to(torch.uint8)[None],
                    global_step=step,
                    fps=8,
                )
            
    if preds.pred_flow_traj is not None:
        import flow_vis
        
        rgb_data = batch_dict["stereo_video"][:, 0, ...]

        new_ims = []
        pred_kp = preds.pred_flow_traj * preds.pred_flow_traj_masks
        gt_kp = None
        if "flow_traj" in batch_dict and "flow_traj_mask" in batch_dict:
            gt_kp = batch_dict["flow_traj"] * batch_dict["flow_traj_mask"]
            gt_kp = gt_kp.permute(0, 2, 1).flip(2)
            pred_kp = pred_kp * batch_dict["flow_traj_mask"]

        pred_kp = pred_kp.permute(0, 2, 1).flip(2)
        rgb_data = (rgb_data).to(torch.uint8)

        # print('gt_kp',gt_kp.shape)
        # print('rgb_data',rgb_data.shape)

        gt_cmap = plt.get_cmap('winter', len(rgb_data))
        pred_cmap = plt.get_cmap('spring', len(rgb_data))


        for i, im in enumerate(rgb_data):
            if gradient:
                for j in range(i):
                    if gt_kp is not None:
                            im = _draw_keypoints(
                                im, gt_kp[j][None], colors=(int(gt_cmap(j)[0]*255),int(gt_cmap(j)[1]*255),int(gt_cmap(j)[2]*255)), radius=1, width=3
                            )
                    im = _draw_keypoints(im, pred_kp[j][None], colors=(int(pred_cmap(j)[0]*255),int(pred_cmap(j)[1]*255),int(pred_cmap(j)[2]*255)), radius=1, width=3)
            else:
                if gt_kp is not None:
                    im = _draw_keypoints(
                            im, gt_kp[i][None], colors='red', radius=1.5, width=3
                        )
                im = _draw_keypoints(im, pred_kp[i][None], colors='blue', radius=1.5, width=3)
            
            new_ims.append(im.cpu())

        rgb_data = torch.stack(new_ims)
        # print("rgb_data", rgb_data.shape, type(rgb_data))

        save_name = f"{sequence_name}_trajectories_"
        if only_foreground:
            save_name += "fg_only"
        else:
            save_name += "full_scene"
            
        video = rgb_data.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        
        video_out = cv2.VideoWriter(
                      os.path.join(
                            output_dir,
                            f"{save_name}.mp4",
                        ),
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps = 10,
                      frameSize=(video.shape[2],video.shape[1]),
                      isColor=True)
        for i in range(len(video)):
            # data = np.random.randint(0, 256, size, dtype='uint8')
            video_out.write(cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB))
        video_out.release()

        if writer is not None:
            writer.add_video(
                save_name, rgb_data.to(torch.uint8)[None], global_step=step, fps=8
            )
                        
                   
        
    return outputs