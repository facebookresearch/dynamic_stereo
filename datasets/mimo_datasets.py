# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

from dataclasses import dataclass
from functools import partial
import numpy as np
from pytorch3d.renderer.cameras import PerspectiveCameras
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
import sys
import cv2

sys.path.append('/private/home/nikitakaraev/dev/')
from mimo_networks.utils.utils import depth2disparity_scale
from mimo_networks.RAFT_MIMO.core.utils import frame_utils
from mimo_networks.RAFT_MIMO.core.utils.augmentor import SequenceDispFlowAugmentor
from collections import defaultdict
from PIL import Image
from pytorch3d import implicitron
import gzip
from typing import List, Optional
from pytorch3d.implicitron.dataset.types import (
    FrameAnnotation as ImplicitronFrameAnnotation,
    load_dataclass
)

@dataclass
class DynamicStereoFrameAnnotation(ImplicitronFrameAnnotation):
    """A dataclass used to load annotations from json."""

    # instance_id_map_path: Optional[str] = None
    matching_annot_path: Optional[str] = None
    # instance_ids: Optional[List[str]] = None
    camera_name: Optional[str] = None

class StereoSequenceDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                raise ValueError('Sparse augmentor is not implemented')
                # self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = SequenceDispFlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        
        self.depth_reader =  self._load_16big_png_depth      
        self.flow_reader = frame_utils.read_gen
        self.is_test = False
        self.sample_list = []
        self.extra_info = []
        self.depth_eps = 1e-5
        # self.depth2disp_scale = 44.1

    def _load_16big_png_depth(self, depth_png):
        with Image.open(depth_png) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = np.frombuffer(
                np.array(depth_pil, dtype=np.uint16), dtype=np.float16
            ).astype(np.float32).reshape((depth_pil.size[1], depth_pil.size[0]))
        return depth
    
    def _get_pytorch3d_camera(
        self,
        entry_viewpoint,
        image_size,
        scale: float,
        # clamp_bbox_xyxy: Optional[torch.Tensor],
    ) -> PerspectiveCameras:
        # entry_viewpoint = entry.viewpoint
        assert entry_viewpoint is not None
        # principal point and focal length
        principal_point = torch.tensor(
            entry_viewpoint.principal_point, dtype=torch.float
        )
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)

        half_image_size_wh_orig = (
            torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
        )

        # first, we convert from the dataset's NDC convention to pixels
        format = entry_viewpoint.intrinsics_format
        if format.lower() == "ndc_norm_image_bounds":
            # this is e.g. currently used in CO3D for storing intrinsics
            rescale = half_image_size_wh_orig
        elif format.lower() == "ndc_isotropic":
            rescale = half_image_size_wh_orig.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {format}")

        # principal point and focal length in pixels
        principal_point_px = half_image_size_wh_orig - principal_point * rescale
        focal_length_px = focal_length * rescale
        # if self.box_crop:
        #     assert clamp_bbox_xyxy is not None
        #     principal_point_px -= clamp_bbox_xyxy[:2]

        # now, convert from pixels to PyTorch3D v0.5+ NDC convention
        # if self.image_height is None or self.image_width is None:
        out_size = list(reversed(image_size))
        # else:
        #     out_size = [self.image_width, self.image_height]

        half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
        half_min_image_size_output = half_image_size_output.min()

        # rescaled principal point and focal length in ndc
        principal_point = (
            half_image_size_output - principal_point_px * scale
        ) / half_min_image_size_output
        focal_length = focal_length_px * scale / half_min_image_size_output

        return PerspectiveCameras(
            focal_length=focal_length[None],
            principal_point=principal_point[None],
            R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
            T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
        )

    def get_output_tensor(self, sample):
        output_tensor = defaultdict(list)
        sample_size = len(sample['image']['left'])
        output_tensor_keys = ['img', 'disp','valid_disp','flow','valid_flow','inv_flow','valid_inv_flow','mask']
        add_keys = ['viewpoint','metadata']
        for add_key in add_keys:
            if add_key in sample:
                # print('ADDKEY', add_key)
                output_tensor_keys.append(add_key)
                # output_tensor[add_key] = copy.deepcopy(sample[add_key])
                
        # print('output_tensor',output_tensor['mask'],output_tensor['viewpoint'],output_tensor['metadata'])
        for key in output_tensor_keys:
            size = sample_size-1 if 'flow'in key else sample_size
            output_tensor[key] = [[] for _ in range(size)]

        if 'viewpoint' in sample:
            # print('meta',torch.Tensor(sample['metadata'][cam][i][1])[None])
            viewpoint_left = self._get_pytorch3d_camera(sample['viewpoint']['left'][0], sample['metadata']['left'][0][1], scale=1.0)
            viewpoint_right = self._get_pytorch3d_camera(sample['viewpoint']['right'][0], sample['metadata']['right'][0][1], scale=1.0)
            depth2disp_scale = depth2disparity_scale(
                viewpoint_left,
                viewpoint_right,
                torch.Tensor(sample['metadata']['left'][0][1])[None]
                # frame_data.image_size_hw[left_idx][None],
            )
            # print(sample['metadata'])
            # print(sample['viewpoint']['left'])
            # print('depth2disp_scale',depth2disp_scale)
        for i in range(sample_size):
            for cam in ['left','right']:
                # for img_path, disp_path, flow_path in zip(sample['image'][cam],  sample['disparity'][cam], sample['flow'][cam]):
                if 'mask' in sample and cam in sample['mask']:
                    mask = frame_utils.read_gen(sample['mask'][cam][i])
                    # print('mask beofre',mask)
                    mask = np.array(mask) / 255.
                    # print('mask',mask.shape,mask.min(),mask.max())
                    # print('mask',mask)
                    output_tensor['mask'][i].append(mask)
                
                if 'viewpoint' in sample and cam in sample['viewpoint']:
                    viewpoint = self._get_pytorch3d_camera(sample['viewpoint'][cam][i], sample['metadata'][cam][i][1], scale=1.0)
                    output_tensor['viewpoint'][i].append(viewpoint)
                    
                    # print('viewpoint',viewpoint)
                    # print('keys',output_tensor.keys())
                    
                    # print(output_tensor['viewpoint'])
                
                if 'metadata' in sample and cam in sample['metadata']:
                    metadata = sample['metadata'][cam][i]
                    output_tensor['metadata'][i].append(metadata)

                if cam in sample['image']:
                    
                    img = frame_utils.read_gen(sample['image'][cam][i])
                    img = np.array(img).astype(np.uint8)

                    # grayscale images
                    if len(img.shape) == 2:
                        img = np.tile(img[...,None], (1, 1, 3))
                    else:
                        img = img[..., :3]
                    output_tensor['img'][i].append(img)
                
                if cam in sample['disparity']:
                    disp = self.disparity_reader(sample['disparity'][cam][i])
                    if isinstance(disp, tuple):
                        disp, valid_disp = disp
                    else:
                        valid_disp = disp < 512
                    disp = np.array(disp).astype(np.float32)
                    
                    disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)
                
                    output_tensor['disp'][i].append(disp)
                    output_tensor['valid_disp'][i].append(valid_disp)
                
                elif 'depth' in  sample and cam in sample['depth']:
                    if 'viewpoint' in sample:
                        depth = self.depth_reader(sample['depth'][cam][i])
                    else:
                        # Falling things
                        depth = frame_utils.read_gen(sample['depth'][cam][i])
                        depth = (np.array(depth).astype(np.float32) / 10000.)
                        depth2disp_scale = 768.1605834960938 * 0.06
                    
                    depth_mask = depth < self.depth_eps
                    depth[depth_mask] = self.depth_eps

                    disp = depth2disp_scale / depth
                    disp[depth_mask] = 0
                    # print('disp',disp[disp>0].shape,disp[disp>0])
                    valid_disp = (disp < 512) * (1-depth_mask) 

                    # print('valid_disp',valid_disp.sum())
                    disp = np.array(disp).astype(np.float32)
                    disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)
                    # print('disp',disp.shape)
                    # print('valid_disp',valid_disp.shape,valid_disp.max(),valid_disp.min())
                    output_tensor['disp'][i].append(disp)
                    output_tensor['valid_disp'][i].append(valid_disp)
                    
                if i<sample_size-1:
                    for flow_dir in ['flow','inv_flow']:
                        if flow_dir in sample:
                            if cam in sample[flow_dir]:
                                flow = self.flow_reader(sample[flow_dir][cam][i])
                                # print('flow',flow.shape,flow)
                                if isinstance(flow, tuple):
                                    flow, valid_flow = flow
                                else:
                                    valid_flow = flow < 1000
                                # print('flow',flow.shape)
                                # print('valid_flow',valid_flow.shape,valid_flow.max(),valid_flow.min())
                                flow = np.array(flow).astype(np.float32)
                                
                                output_tensor[flow_dir][i].append(flow)
                                output_tensor['valid_'+flow_dir][i].append(valid_flow)
        return output_tensor

    def __getitem__(self, index):
        im_tensor = {'img'}
        sample = self.sample_list[index]
        if self.is_test:
            sample_size = len(sample['image']['left'])
            im_tensor['img'] = [[] for _ in range(sample_size)]
            for i in range(sample_size):
                for cam in ['left','right']:
                    img = frame_utils.read_gen(sample['image'][cam][i])
                    img = np.array(img).astype(np.uint8)[..., :3]
                    img = torch.from_numpy(img).permute(2, 0, 1).float()
                    im_tensor['img'][i].append(img)
            im_tensor['img'] = torch.stack(im_tensor['img'])
            return im_tensor, self.extra_info[index]

        # if not self.init_seed:
        #     worker_info = torch.utils.data.get_worker_info()
        #     if worker_info is not None:
        #         torch.manual_seed(worker_info.id)
        #         np.random.seed(worker_info.id)
        #         random.seed(worker_info.id)
        #         self.init_seed = True

        index = index % len(self.sample_list)
        # im_tensor, disp_tensor, flow_tensor = [], [], []
        # output_tensor = defaultdict(lambda: defaultdict(list))
        try:
            output_tensor = self.get_output_tensor(sample)
        except:
            print('except')
            index = np.random.randint(len(self.sample_list))  
            print('new index', index)
            sample = self.sample_list[index]    
            output_tensor = self.get_output_tensor(sample)
        sample_size = len(sample['image']['left'])  
        # print('output_tensor',(k,v.shape for k,v in output_tensor.items()))  
        if self.augmentor is not None:
            # if self.sparse:
            #     img, disp, valid_disp, flow, valid_flow = self.augmentor(output_tensor['img'],
            #                                                              output_tensor['flow'],
            #                                                              output_tensor['valid_flow'],
            #                                                              output_tensor['disp'],
            #                                                              output_tensor['valid_disp']
            #                                                              )
            # else:
            output_tensor['img'], output_tensor['flow'], output_tensor['inv_flow'], output_tensor['disp'] = self.augmentor(output_tensor['img'],
                                             output_tensor['flow'],
                                             output_tensor['inv_flow'],
                                             output_tensor['disp'])
        for i in range(sample_size):
            for cam in (0,1):
                if cam<len(output_tensor['img'][i]):
                    img = torch.from_numpy(output_tensor['img'][i][cam]).permute(2, 0, 1).float()
                    if self.img_pad is not None:
                        padH, padW = self.img_pad
                        img = F.pad(img, [padW]*2 + [padH]*2)
                    output_tensor['img'][i][cam] = img

                if cam<len(output_tensor['disp'][i]):
                    disp = torch.from_numpy(output_tensor['disp'][i][cam]).permute(2, 0, 1).float()
                
                    if self.sparse:
                        valid_disp = torch.from_numpy(output_tensor['valid_disp'][i][cam])
                    else:
                        valid_disp = (disp[0].abs() < 512) & (disp[1].abs() < 512) & (disp[0].abs() != 0)
                    disp = disp[:1]
                
                    output_tensor['disp'][i][cam] = disp
                    output_tensor['valid_disp'][i][cam]=valid_disp.float()

                if 'mask' in output_tensor and cam<len(output_tensor['mask'][i]):
                    mask = torch.from_numpy(output_tensor['mask'][i][cam]).float()
                    output_tensor['mask'][i][cam] = mask

                if 'viewpoint' in output_tensor and cam<len(output_tensor['viewpoint'][i]):
                    viewpoint = output_tensor['viewpoint'][i][cam]
                    output_tensor['viewpoint'][i][cam] = viewpoint

                # if 'metadata' in output_tensor and cam<len(output_tensor['metadata'][i]):
                #     metadata = output_tensor['metadata'][i][cam])
                #     output_tensor['metadata'][i][cam] = metadata

                if i < sample_size-1:
                    for flow_dir in ['flow','inv_flow']:
                        if flow_dir in output_tensor:
                            if cam<len(output_tensor[flow_dir][i]):
                                flow = torch.from_numpy(output_tensor[flow_dir][i][cam]).permute(2, 0, 1).float()
                                if self.sparse:
                                    valid_flow = torch.from_numpy(output_tensor['valid_'+flow_dir][i][cam]).permute(2, 0, 1)
                                else:
                                    valid_flow = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
                                
                                output_tensor[flow_dir][i][cam] = flow
                                output_tensor['valid_'+flow_dir][i][cam]=valid_flow.float()
        # print('output_tensor',output_tensor['metadata'])
        # print('output_tensor viewpoint',output_tensor['viewpoint'])
        res = {}
        if 'viewpoint' in output_tensor and self.split!='train':
            res['viewpoint'] = output_tensor['viewpoint']
        if 'metadata' in output_tensor and self.split!='train':
            res['metadata'] = output_tensor['metadata']
        for k, v in output_tensor.items():
            # print(k,v)
            if k!='viewpoint' and k!='metadata':
                for i in range(len(v)):
                    if len(v[i])>0:
                        v[i] = torch.stack(v[i])
                if len(v)>0 and (len(v[0])>0):
                    res[k] = torch.stack(v)
        # return sample, output_tensor
        return res

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.sample_list = v * copy_of_self.sample_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.sample_list)

def flowreader(flow_path):
    flow_mask_path = flow_path.replace('flow_forward','flow_forward_mask')
    with Image.open(flow_path) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        flow = np.frombuffer(
            np.array(depth_pil, dtype=np.uint16), dtype=np.float16
        ).astype(np.float32).reshape((depth_pil.size[1], depth_pil.size[0]))
    
    flow_res_mask = np.array(Image.open(flow_mask_path))

    flow_res = np.stack([flow[:,:flow.shape[1]//2], flow[:,flow.shape[1]//2:]],axis=-1)
    flow_res[:,:,0][np.logical_not(flow_res_mask)] = 1000
    flow_res[:,:,1][np.logical_not(flow_res_mask)] = 1000
    return flow_res, flow_res_mask

class DynamicStereoDataset(StereoSequenceDataset):
    def __init__(self, aug_params=None, root='/large_experiments/p3/replay/datasets/synthetic/replica_animals/dynamic_replica/dataset/', split='train', sample_len=-1, only_first_n_samples=-1, is_real_data=False):
        super(DynamicStereoDataset, self).__init__(aug_params)
        self.root = root
        self.sample_len = sample_len
        self.split = split
        self.flow_reader = flowreader
        # if split=='train':
        #     assert sample_len>0
        
        frame_annotations_file = f'frame_annotations_{split}.jgz'
        print('osp.join(root, frame_annotations_file)',osp.join(root, frame_annotations_file))
        with gzip.open(osp.join(root, frame_annotations_file), "rt", encoding="utf8") as zipfile:
            frame_annots_list = load_dataclass(
                        zipfile, List[DynamicStereoFrameAnnotation]
                    )
        seq_annot = defaultdict(lambda: defaultdict(list))
        for frame_annot in frame_annots_list:
            seq_annot[frame_annot.sequence_name][frame_annot.camera_name].append(frame_annot)
        print('seq_annot',len(seq_annot))
        total_frames=0
        total_seqs=0
        for seq_name in seq_annot.keys():
            if seq_name=='8369cb-7_obj' or seq_name=='a7b915-7_obj':
                continue
            images, depths, viewpoints, metadata, masks, flows = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),  defaultdict(list)
            # defaultdict(lambda: defaultdict(list))
            for cam in ['left','right']:
                for framedata in seq_annot[seq_name][cam]:
                    # frame_annots_list
                    im_path = osp.join(root, split, framedata.image.path)
                    if is_real_data:
                        im_path=im_path.replace('/checkpoint/nikitakaraev/2022_mimo/datasets/zedmini_sequences/', \
                            '/checkpoint/nikitakaraev/2022_mimo/datasets/zedmini_sequences/')
                    assert os.path.isfile(im_path), im_path
                    images[cam].append(im_path)
                    viewpoints[cam].append(framedata.viewpoint)
                    metadata[cam].append([framedata.sequence_name, framedata.image.size])
                    depth_path = osp.join(root, split, framedata.depth.path)
                    # print('depth_path',depth_path)
                    depths[cam].append(depth_path)
                    if not is_real_data:
                        flows[cam].append(depth_path.replace('depths','flow_forward').replace('.geometric.','.'))
                    # inv_flows[cam].append(depth_path.replace('depths','flow_backward').replace('.geometric.','.'))
                    assert os.path.isfile(depth_path), depth_path
                    # 
                    mask_path = osp.join(root, split, framedata.mask.path)                    
                    assert os.path.isfile(mask_path), mask_path
                    masks[cam].append(mask_path)
                    assert len(images[cam])==len(masks[cam])>0, framedata.sequence_name
                    
                    assert len(images[cam])==len(depths[cam])==len(viewpoints[cam])==len(metadata[cam])>0, framedata.sequence_name

            seq_len = len(images[cam])
            total_frames+=seq_len
            total_seqs+=1
            print('seq_len', seq_name, seq_len)
            if split=='train':
                for ref_idx in range(0, seq_len, 3):
                    step = 1 if self.sample_len==1 else np.random.randint(1,6)
                    if ref_idx + step * self.sample_len < seq_len:
                        sample = defaultdict(lambda: defaultdict(list))
                        for cam in ['left','right']:
                            for idx in range(ref_idx, ref_idx+step*self.sample_len, step):
                                # print('idx',idx, 'images[cam]',len(images[cam]),'sample[image]',len(sample['image']))
                                sample['image'][cam].append(images[cam][idx])
                                sample['depth'][cam].append(depths[cam][idx])
                                sample['viewpoint'][cam].append(viewpoints[cam][idx])
                                sample['metadata'][cam].append(metadata[cam][idx])
                                if not is_real_data:
                                    sample['flow'][cam].append(flows[cam][idx])
                        self.sample_list.append(sample)
            else:
                step = self.sample_len if self.sample_len>0 else seq_len
                counter=0
                # print('seq_len',seq_len)
                for ref_idx in range(0, seq_len, step):
                    sample = defaultdict(lambda: defaultdict(list))
                    for cam in ['left','right']:
                        # print('step',step)
                        # print('images',len(images[cam]))
                        for idx in range(ref_idx, ref_idx+step):
                            # print('idx',idx,len(images[cam]))
                            # print('seq_len',seq_len,'ref_idx',ref_idx,'idx',idx)
                            sample['image'][cam].append(images[cam][idx])
                            sample['viewpoint'][cam].append(viewpoints[cam][idx])
                            sample['metadata'][cam].append(metadata[cam][idx])
                            sample['depth'][cam].append(depths[cam][idx])
                            if not is_real_data:
                                sample['flow'][cam].append(flows[cam][idx])
                            # if not is_real_data:
                            sample['mask'][cam].append(masks[cam][idx])
                            
                        # sample['flow'] = []
                    self.sample_list.append(sample)
                    counter+=1
                    if only_first_n_samples>0 and counter>=only_first_n_samples:
                        break
        print('total_frames',total_frames)
        print('total_seqs',total_seqs)   
        
        print(f"Added {len(self.sample_list)} from DynamicStereo {split}")
        logging.info(f"Added {len(self.sample_list)} from DynamicStereo {split}")


class FallingThingsDataset(StereoSequenceDataset):
    def __init__(self, aug_params=None, root='/checkpoint/nikitakaraev/2022_mimo/datasets/falling_things/fat/', sample_len=-1):
        super(FallingThingsDataset, self).__init__(aug_params)
        assert os.path.exists(root)
        self.root = root
        self.sample_len = sample_len
        original_length = len(self.sample_list)
        # images, depths, flows = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
        image_paths=[]
        # for cam in ['left','right']:
        image_paths = sorted( glob(osp.join(root,'single/*/*')) ) + sorted( glob(osp.join(root,'mixed/*')) )
        # print('image_paths',image_paths)
        
        num_seq = len(image_paths)
        # for each sequence       
        for seq_idx in range(num_seq):
            # seq_path = image_paths[seq_idx]
            # seq_len = 100 if 'single' in seq_path else 3000
            images={}
            depths={}

            for cam in ['left','right']:
                images[cam] = sorted(glob(osp.join(image_paths[seq_idx], f'*{cam}.jpg')) )
                # print(sorted(glob(osp.join(image_paths[seq_idx], f'*{cam}.jpg')) ))
                depths[cam] = [ path.replace('.jpg', '.depth.png') for path in images[cam] ]
                # depths[cam] = sorted(glob(osp.join(image_paths[seq_idx], f'*{cam}.depth.png')) )
                seq_len = len(images[cam])
                # print('seq len', seq_len)
            for ref_idx in range(0, seq_len, 1):
                step = 1 
                # if self.sample_len==1 else np.random.randint(1,6)
                if ref_idx + step * self.sample_len < seq_len:
                    sample = defaultdict(lambda: defaultdict(list))
                    for cam in ['left','right']:
                        for idx in range(ref_idx, ref_idx+step*self.sample_len, step):
                            # print('idx',idx, 'images[cam]',len(images[cam]),'sample[image]',len(sample['image']))
                            sample['image'][cam].append(images[cam][idx])
                            sample['depth'][cam].append(depths[cam][idx])
                            
                    sample['flow'] = []
                    self.sample_list.append(sample)


        # for seq_idx in range(num_seq):
        #     sample = defaultdict(lambda: defaultdict(list))
        #     for cam in ['left','right']:
                
        #         sample['image'][cam] = sorted(glob(osp.join(image_paths[seq_idx], f'{cam}.jpg')) )
                
        #         sample['depth'][cam] = [ path.replace('.jpg', '.depth.png') for path in sample['image'][cam] ]
                
        #         #  sorted(glob(osp.join(image_paths[seq_idx], f'{cam}.depth.png')) )
            
        #     self.sample_list.append(sample)
                    
        logging.info(f"Added {len(self.sample_list) - original_length} from FallingThings")


class SequenceSceneFlowDatasets(StereoSequenceDataset):
    def __init__(self, aug_params=None, root='/checkpoint/nikitakaraev/2022_mimo/datasets/', dstype='frames_cleanpass', sample_len=1, things_test=False, add_things=True, add_monkaa=True, add_driving=True, add_humans=False, step=1):
        super(SequenceSceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype
        self.sample_len = sample_len
        self.step = step
        if things_test:
            self._add_things("TEST")
        else:
            if add_things:
                self._add_things("TRAIN")
            if add_monkaa:
                self._add_monkaa()
            if add_driving:
                self._add_driving()
            if add_humans:
                self._add_humans()
            

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.sample_list)
        root = osp.join(self.root, 'FlyingThings3D')
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        flow_paths = defaultdict(lambda: defaultdict(list))
        # for position in ['left','right']:
        # images['left'] = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/')) )
        # images['right'] = [ im.replace('left', 'right') for im in images['left'] ]
        for cam in ['left','right']:
            image_paths[cam] = sorted( glob(osp.join(root, self.dstype, split, f'*/*/{cam}/')) )
            disparity_paths[cam] = [ path.replace(self.dstype, 'disparity') for path in image_paths[cam] ]
            for direction in ['into_future', 'into_past']:
                flow_paths[cam][direction] = sorted( glob(osp.join(root, f'optical_flow/{split}/*/*/{direction}/{cam}/') ))
                assert len(flow_paths[cam][direction])==len(image_paths[cam])
        
        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(image_paths['left']))[:40])
        np.random.set_state(state)
        np.random.seed(0)
        # for cam in ['left']:
        # for direction in ['into_future', 'into_past']:
                # image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                # image_dirs = sorted([osp.join(f, cam) for f in image_dirs])
        # print('val_idxs',val_idxs) 
        num_seq = len(image_paths['left'])
        # for each sequence       

        # num_seq = 5
        for seq_idx in range(num_seq):
            # print('seq_idx',seq_idx)
            if (split == 'TEST' and seq_idx in val_idxs) or (split == 'TRAIN' and not seq_idx in val_idxs):
                images, disparities, flows = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
                for cam in ['left','right']:
                    images[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], '*.png')) )
                    disparities[cam] = sorted(glob(osp.join(disparity_paths[cam][seq_idx], '*.pfm')) )
                    for direction in ['into_future', 'into_past']:
                        flows[cam][direction] = sorted(glob(osp.join(flow_paths[cam][direction][seq_idx], '*.pfm')) )
                    
                self._append_sample(images, disparities, flows, step=self.step)

        print(f"Added {len(self.sample_list) - original_length} from FlyingThings {self.dstype}")
        logging.info(f"Added {len(self.sample_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.sample_list)
        root = osp.join(self.root, 'Monkaa')
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        flow_paths = defaultdict(lambda: defaultdict(list))
        
        for cam in ['left','right']:
            image_paths[cam] = sorted( glob(osp.join(root, self.dstype, f'*/{cam}/')) )
            disparity_paths[cam] = [ path.replace(self.dstype, 'disparity') for path in image_paths[cam] ]
            for direction in ['into_future', 'into_past']:
                flow_paths[cam][direction] = sorted( glob(osp.join(root, f'optical_flow/*/{direction}/{cam}/') ))
                assert len(flow_paths[cam][direction])==len(image_paths[cam])
        
        num_seq = len(image_paths['left'])
        # for each sequence   
        # num_seq = 0   
        for seq_idx in range(num_seq):
            # print('seq_idx',seq_idx)
            images, disparities, flows = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
            for cam in ['left','right']:
                images[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], '*.png')) )
                disparities[cam] = sorted(glob(osp.join(disparity_paths[cam][seq_idx], '*.pfm')) )
                for direction in ['into_future', 'into_past']:
                    flows[cam][direction] = sorted(glob(osp.join(flow_paths[cam][direction][seq_idx], '*.pfm')) )
                    
            self._append_sample(images, disparities, flows, step=self.step)

        print(f"Added {len(self.sample_list) - original_length} from Monkaa {self.dstype}")
        logging.info(f"Added {len(self.sample_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.sample_list)
        root = osp.join(self.root, 'Driving')
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        flow_paths = defaultdict(lambda: defaultdict(list))
        
        for cam in ['left','right']:
            image_paths[cam] = sorted( glob(osp.join(root, self.dstype, f'*/*/*/{cam}/')) )
            disparity_paths[cam] = [ path.replace(self.dstype, 'disparity') for path in image_paths[cam] ]
            for direction in ['into_future', 'into_past']:
                flow_paths[cam][direction] = sorted( glob(osp.join(root, f'optical_flow/*/*/*/{direction}/{cam}/') ))
                assert len(flow_paths[cam][direction])==len(image_paths[cam])

        num_seq = len(image_paths['left'])
        # for each sequence      
        # num_seq = 0
        for seq_idx in range(num_seq):
            # print('seq_idx',seq_idx)
            images, disparities, flows = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
            for cam in ['left','right']:
                images[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], '*.png')) )
                disparities[cam] = sorted(glob(osp.join(disparity_paths[cam][seq_idx], '*.pfm')) )
                for direction in ['into_future', 'into_past']:
                    flows[cam][direction] = sorted(glob(osp.join(flow_paths[cam][direction][seq_idx], '*.pfm')) )
                    
            self._append_sample(images, disparities, flows, step=self.step)

        print(f"Added {len(self.sample_list) - original_length} from Driving {self.dstype}")
        logging.info(f"Added {len(self.sample_list) - original_length} from Driving {self.dstype}")

    def _add_humans(self):
        """ Add SynthHumans data """
        original_length = len(self.sample_list)
        root = '/large_experiments/p3/replay/datasets/synthetic/pixar_humans_v3_zedstereo/'
        image_paths = defaultdict(list)
        # depth_paths = defaultdict(list)
        
        for cam in ['left','right']:
            image_paths[cam] = sorted( glob(osp.join(root, f'*_{cam}')) )
            # depth_paths[cam] = [ path.replace(self.dstype, 'depth') for path in image_paths[cam] ]
        # print('image_paths',image_paths)
        num_seq = len(image_paths['left'])
        # for each sequence       
        for seq_idx in range(num_seq):
            # print('seq_idx',seq_idx)
            images, depths, flows = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
            for cam in ['left','right']:
                images[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], 'images/*.png')) )
                depths[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], 'depths/*.png')) )
                
            # self._append_sample(images, depths, flows)
            seq_len = len(images['left'])
            for ref_idx in range(0, seq_len-self.sample_len, self.sample_len):
                # print('ref_idx', ref_idx)
                # sample = defaultdict(lambda: defaultdict(list))
                # for cam in ['left','right']:
                #     for idx in range(ref_idx, ref_idx+self.sample_len):
                #         sample['image'][cam].append(images[cam][idx])
                #         sample['depth'][cam].append(depths[cam][idx])
                # sample['flow'] = []
                # self.sample_list.append(sample)

                # if ref_idx + 2*self.sample_len < seq_len:
                #     sample = defaultdict(lambda: defaultdict(list))
                #     for cam in ['left','right']:
                #         for idx in range(ref_idx, ref_idx+2*self.sample_len, 2):
                #             sample['image'][cam].append(images[cam][idx])
                #             sample['depth'][cam].append(depths[cam][idx])
                #     sample['flow'] = []
                #     self.sample_list.append(sample)

                # train on every 3rd frame
                # step = np.random.randint(1,6)
                step = np.random.randint(40,60)
                if ref_idx + step * self.sample_len < seq_len:
                    sample = defaultdict(lambda: defaultdict(list))
                    for cam in ['left','right']:
                        for idx in range(ref_idx, ref_idx+step*self.sample_len, step):
                            sample['image'][cam].append(images[cam][idx])
                            sample['depth'][cam].append(depths[cam][idx])
                    sample['flow'] = []
                    self.sample_list.append(sample)
            # print('self.sample_list',len(self.sample_list))
            # break
        print(f"Added {len(self.sample_list) - original_length} from SynthHumans {self.dstype}")
        logging.info(f"Added {len(self.sample_list) - original_length} from SynthHumans {self.dstype}")


    def _append_sample(self, images, disparities, flows, step=1):
        #  create a sample      
        # print('seq_len',seq_len)
        seq_len = len(images['left'])
        for ref_idx in range(0, seq_len-self.sample_len, step):
            # print('ref_idx', ref_idx)
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ['left','right']:
                for idx in range(ref_idx, ref_idx+self.sample_len):
                    sample['image'][cam].append(images[cam][idx])
                    sample['disparity'][cam].append(disparities[cam][idx])
                for idx in range(ref_idx, ref_idx+self.sample_len-1):
                    sample['flow'][cam].append(flows[cam]['into_future'][idx])
                    sample['inv_flow'][cam].append(flows[cam]['into_past'][idx+1])
            self.sample_list.append(sample)
            # print(f'sample {seq_idx}',sample['image']['left'],sample['disparity']['left'],sample['flow']['left'])
            # print('')
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ['left','right']:
                for idx in range(ref_idx, ref_idx+self.sample_len):
                    sample['image'][cam].append(images[cam][seq_len-idx-1])
                    sample['disparity'][cam].append(disparities[cam][seq_len-idx-1])
                for idx in range(ref_idx, ref_idx+self.sample_len-1):
                    sample['flow'][cam].append(flows[cam]['into_past'][seq_len-idx-1]) 
                    sample['inv_flow'][cam].append(flows[cam]['into_future'][seq_len-idx-2])
            self.sample_list.append(sample)
            # print(f'sample {seq_idx}',sample['image']['left'],sample['disparity']['left'],sample['flow']['left'])
            # print('')
                # for idx in range(ref_idx-sample_distance, ref_idx):
                #     sample['flow'][cam].append(flows[cam]['into_past'][idx])
                # for idx in range(ref_idx, ref_idx+sample_distance):
                #     sample['flow'][cam].append(flows[cam]['into_future'][idx])

class SequenceSintelStereo(StereoSequenceDataset):
    def __init__(self, dstype='clean', aug_params=None, root='/checkpoint/nikitakaraev/2022_mimo/datasets'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)
        self.dstype = dstype
        original_length = len(self.sample_list)
        image_root = osp.join(root, 'sintel_stereo', 'training')
        flow_root = osp.join(root, 'sintel_flow', 'training')
        # print('image_root',image_root)
        # print('flow_root',flow_root)
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        flow_paths = defaultdict(lambda: defaultdict(list))
        
        for cam in ['left','right']:
            image_paths[cam] = sorted( glob(osp.join(image_root, f'{self.dstype}_{cam}/*')) )
            print('image_paths',image_paths)
        cam ='left'
        disparity_paths[cam] = [ path.replace(f'{self.dstype}_{cam}', 'disparities') for path in image_paths[cam] ]
        flow_paths[cam] = sorted( glob(osp.join(flow_root, 'flow/*') ))
        assert len(flow_paths[cam])==len(image_paths[cam])
        # print('disparity_paths',disparity_paths)
        # print('flow_paths',flow_paths)
        num_seq = len(image_paths['left'])
        # for each sequence       
        for seq_idx in range(num_seq):
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ['left','right']:
                sample['image'][cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], '*.png')) )
            cam ='left'
            sample['disparity'][cam] = sorted(glob(osp.join(disparity_paths[cam][seq_idx], '*.png')) )    
            sample['flow'][cam] = sorted(glob(osp.join(flow_paths[cam][seq_idx], '*.flo')) )
            for im1, disp, flow in zip(sample['image'][cam], sample['disparity'][cam], sample['flow'][cam]):
                assert im1.split('/')[-1].split('.')[0] == disp.split('/')[-1].split('.')[0] == flow.split('/')[-1].split('.')[0], (im1.split('/')[-1].split('.')[0], disp.split('/')[-1].split('.')[0], flow.split('/')[-1].split('.')[0])
            self.sample_list.append(sample)
                    
        logging.info(f"Added {len(self.sample_list) - original_length} from FlyingThings {self.dstype}")

class SequenceETH3D(StereoSequenceDataset):
    def __init__(self, aug_params=None, root='/checkpoint/nikitakaraev/2022_mimo/datasets/ETH3D', split='training'):
        super().__init__(aug_params, sparse=True)
        original_length = len(self.sample_list)
        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)
        num_seq = len(image1_list)
        for seq_idx in range(num_seq):
            sample = defaultdict(lambda: defaultdict(list))
            sample['image']['left'] = [image1_list[seq_idx]]
            sample['image']['right'] = [image2_list[seq_idx]]
            sample['disparity']['left'] = [disp_list[seq_idx]]
            self.sample_list.append(sample)
                    
        logging.info(f"Added {len(self.sample_list) - original_length} from ETH3D")

class SequenceMiddlebury(StereoSequenceDataset):
    def __init__(self, aug_params=None, root='/checkpoint/nikitakaraev/2022_mimo/datasets/Middlebury', split='F'):
        super().__init__(aug_params, sparse=True,  reader=frame_utils.readDispMiddlebury)
        original_length = len(self.sample_list)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2014"]
        image_list,disparity_list=[],[]
        if split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
            lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3/official_train.txt")).read_text().splitlines()), lines))
            image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                image_list += [ [img1, img2] ]
                disparity_list += [ disp ]

        num_seq = len(image_list)
        for seq_idx in range(num_seq):
            sample = defaultdict(lambda: defaultdict(list))
            sample['image']['left'] = [image_list[seq_idx][0]]
            sample['image']['right'] = [image_list[seq_idx][1]]
            sample['disparity']['left'] = [disparity_list[seq_idx]]
            self.sample_list.append(sample)
                    
        logging.info(f"Added {len(self.sample_list) - original_length} from FlyingThings")

class SequenceKITTI(StereoSequenceDataset):
    def __init__(self, aug_params=None, root='/checkpoint/nikitakaraev/2022_mimo/datasets/kitti', image_set='training', additional_images=False):
        super(SequenceKITTI, self).__init__(aug_params, sparse=True, reader=self.readDispKITTI)
        assert os.path.exists(root)
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)
        original_length = len(self.sample_list)
        
        for cam in ['left','right']:
            folder = 'image_2' if cam=='left' else 'image_3'
            image_paths[cam] = sorted( glob(osp.join(root, image_set, f'{folder}/*_10.png' )) )
        cam='left'
        disparity_paths[cam] = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) 
        assert len(disparity_paths[cam])==len(image_paths[cam])
            
        num_seq = len(image_paths['left'])
        
        for seq_idx in range(num_seq):
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ['left','right']:
                if additional_images:
                    sample['image'][cam] = sorted(glob(image_paths[cam][seq_idx].replace('_10.png','_*.png') ))
                else:
                    sample['image'][cam] = [image_paths[cam][seq_idx]]
                
            cam ='left'
            if additional_images:
                sample['disparity'][cam] = [(None,disparity_paths[cam][seq_idx])]*10+[disparity_paths[cam][seq_idx]]+[(None,disparity_paths[cam][seq_idx])]*10
                # print(seq_idx,sample['disparity'][cam] )
                # print(sample['image'][cam][10].split('/')[-1].split('.')[0], sample['disparity'][cam][10].split('/')[-1].split('.')[0])
            else:
                sample['disparity'][cam] = [disparity_paths[cam][seq_idx]]
            
            
            for im1, disp in zip(sample['image'][cam], sample['disparity'][cam]):
                if not isinstance(disp, tuple):
                    assert im1.split('/')[-1].split('.')[0] == disp.split('/')[-1].split('.')[0], (im1.split('/')[-1].split('.')[0], disp.split('/')[-1].split('.')[0])
            self.sample_list.append(sample)

        logging.info(f"Added {len(self.sample_list) - original_length} from KITTI")

    def readDispKITTI(self, filename):
        if isinstance(filename, tuple):
            # disp = np.zeros((375, 1242))
            disp = np.zeros_like(cv2.imread(filename[1], cv2.IMREAD_ANYDEPTH) / 256.0)
        else:
            disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
            # print('disp',disp.shape)
        valid = disp > 0.0
        return disp, valid
#         for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
#             self.image_list += [ [img1, img2] ]
#             self.disparity_list += [ disp ]

class SequenceFlyingChairs(StereoSequenceDataset):
    def __init__(self, aug_params=None, split='training', root='/checkpoint/nikitakaraev/2022_mimo/datasets/FlyingChairs_release/data'):
        super(SequenceFlyingChairs, self).__init__(aug_params)
        original_length = len(self.sample_list)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('/checkpoint/nikitakaraev/2022_mimo/datasets/FlyingChairs_release/FlyingChairs_train_val.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                sample = defaultdict(lambda: defaultdict(list))
                for cam in ['left','right']:
                    sample['image'][cam] = [ images[2*i], images[2*i+1] ]
                    sample['flow'][cam] = [flows[i]]
                sample['disparity'] = []

                self.sample_list.append(sample)
                    
        logging.info(f"Added {len(self.sample_list) - original_length} from FlyingChairs")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    # def collate_fn(dataset, batch):
    #     len_batch = len(batch) 
    #     batch = list(filter(lambda x: x is not None, batch))
        
    #     if len_batch > len(batch): # if there are samples missing just use existing members, doesn't work if you reject every sample in a batch
    #         diff = len_batch - len(batch)
    #         for __ in range(diff):
    #             batch.append(dataset[np.random.randint(0, len(dataset))])
    #     return torch.utils.data.dataloader.default_collate(batch)

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip
        
    train_dataset = None
    # for dataset_name in args.train_datasets:
    print('args.train_datasets',args.train_datasets)
    # if 'things' in args.train_datasets:
    add_monkaa = 'monkaa' in args.train_datasets
    add_driving = 'driving' in args.train_datasets
    add_things = 'things' in args.train_datasets
    add_humans = 'humans' in args.train_datasets
    add_falling_things = 'falling_things' in args.train_datasets

    step=2 if add_humans else 1
    new_dataset = None

    if add_monkaa or add_driving or add_things:
        clean_dataset = SequenceSceneFlowDatasets(aug_params, dstype='frames_cleanpass', sample_len=args.sample_len, add_monkaa=add_monkaa, add_driving=add_driving, add_things=add_things, add_humans=False, step=step)
        final_dataset = SequenceSceneFlowDatasets(aug_params, dstype='frames_finalpass', sample_len=args.sample_len, add_monkaa=add_monkaa, add_driving=add_driving, add_things=add_things, add_humans=False, step=step)
        new_dataset = clean_dataset + final_dataset

    if add_humans:
        human_dataset = DynamicStereoDataset(aug_params, root='/large_experiments/p3/replay/datasets/synthetic/replica_animals/dynamic_replica_random_baseline/dataset', split='train', sample_len=args.sample_len)
        if new_dataset is None:
            new_dataset = human_dataset # *4
        else:
            new_dataset = new_dataset + human_dataset # *4

    if add_falling_things:
        falling_dataset = FallingThingsDataset(aug_params, sample_len=args.sample_len)
        if new_dataset is None:
            new_dataset = falling_dataset # *4
        else:
            new_dataset = new_dataset + falling_dataset # *4
    
    logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
    train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    # train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, 20)))
    # if 'chairs' in args.train_datasets:
    #     new_dataset = SequenceFlyingChairs(aug_params)
    #     logging.info(f"Adding {len(new_dataset)} samples from FlyingChairs")
    #     train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    # g = torch.Generator()
    # g.manual_seed(0)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=args.num_workers, drop_last=True)
        # , worker_init_fn=seed_worker, generator=g)
        # , collate_fn=partial(collate_fn, train_dataset))
    # print('workers',int(os.environ.get('SLURM_CPUS_PER_TASK', 6)))
    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

