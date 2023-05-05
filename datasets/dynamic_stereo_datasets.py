# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# # Data loading based on https://github.com/NVIDIA/flownet2-pytorch


import os
import copy
import gzip
import logging
import torch
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
import os.path as osp
from glob import glob

from collections import defaultdict
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.implicitron.dataset.types import (
    FrameAnnotation as ImplicitronFrameAnnotation,
    load_dataclass,
)

from dynamic_stereo.datasets import frame_utils
from dynamic_stereo.evaluation.utils.eval_utils import depth2disparity_scale
from dynamic_stereo.datasets.augmentor import SequenceDispFlowAugmentor


@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    """A dataclass used to load annotations from json."""

    camera_name: Optional[str] = None


class StereoSequenceDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = (
            aug_params.pop("img_pad", None) if aug_params is not None else None
        )
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                raise ValueError("Sparse augmentor is not implemented")
            else:
                self.augmentor = SequenceDispFlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader
        self.depth_reader = self._load_16big_png_depth
        self.is_test = False
        self.sample_list = []
        self.extra_info = []
        self.depth_eps = 1e-5

    def _load_16big_png_depth(self, depth_png):
        with Image.open(depth_png) as depth_pil:
            # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
            # we cast it to uint16, then reinterpret as float16, then cast to float32
            depth = (
                np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depth_pil.size[1], depth_pil.size[0]))
            )
        return depth

    def _get_pytorch3d_camera(
        self, entry_viewpoint, image_size, scale: float
    ) -> PerspectiveCameras:
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

        # now, convert from pixels to PyTorch3D v0.5+ NDC convention
        # if self.image_height is None or self.image_width is None:
        out_size = list(reversed(image_size))

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

    def _get_output_tensor(self, sample):
        output_tensor = defaultdict(list)
        sample_size = len(sample["image"]["left"])
        output_tensor_keys = ["img", "disp", "valid_disp", "mask"]
        add_keys = ["viewpoint", "metadata"]
        for add_key in add_keys:
            if add_key in sample:
                output_tensor_keys.append(add_key)

        for key in output_tensor_keys:
            output_tensor[key] = [[] for _ in range(sample_size)]

        if "viewpoint" in sample:
            viewpoint_left = self._get_pytorch3d_camera(
                sample["viewpoint"]["left"][0],
                sample["metadata"]["left"][0][1],
                scale=1.0,
            )
            viewpoint_right = self._get_pytorch3d_camera(
                sample["viewpoint"]["right"][0],
                sample["metadata"]["right"][0][1],
                scale=1.0,
            )
            depth2disp_scale = depth2disparity_scale(
                viewpoint_left,
                viewpoint_right,
                torch.Tensor(sample["metadata"]["left"][0][1])[None],
            )

        for i in range(sample_size):
            for cam in ["left", "right"]:
                if "mask" in sample and cam in sample["mask"]:
                    mask = frame_utils.read_gen(sample["mask"][cam][i])
                    mask = np.array(mask) / 255.0
                    output_tensor["mask"][i].append(mask)

                if "viewpoint" in sample and cam in sample["viewpoint"]:
                    viewpoint = self._get_pytorch3d_camera(
                        sample["viewpoint"][cam][i],
                        sample["metadata"][cam][i][1],
                        scale=1.0,
                    )
                    output_tensor["viewpoint"][i].append(viewpoint)

                if "metadata" in sample and cam in sample["metadata"]:
                    metadata = sample["metadata"][cam][i]
                    output_tensor["metadata"][i].append(metadata)

                if cam in sample["image"]:

                    img = frame_utils.read_gen(sample["image"][cam][i])
                    img = np.array(img).astype(np.uint8)

                    # grayscale images
                    if len(img.shape) == 2:
                        img = np.tile(img[..., None], (1, 1, 3))
                    else:
                        img = img[..., :3]
                    output_tensor["img"][i].append(img)

                if cam in sample["disparity"]:
                    disp = self.disparity_reader(sample["disparity"][cam][i])
                    if isinstance(disp, tuple):
                        disp, valid_disp = disp
                    else:
                        valid_disp = disp < 512
                    disp = np.array(disp).astype(np.float32)

                    disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)

                    output_tensor["disp"][i].append(disp)
                    output_tensor["valid_disp"][i].append(valid_disp)

                elif "depth" in sample and cam in sample["depth"]:
                    depth = self.depth_reader(sample["depth"][cam][i])

                    depth_mask = depth < self.depth_eps
                    depth[depth_mask] = self.depth_eps

                    disp = depth2disp_scale / depth
                    disp[depth_mask] = 0
                    valid_disp = (disp < 512) * (1 - depth_mask)

                    disp = np.array(disp).astype(np.float32)
                    disp = np.stack([-disp, np.zeros_like(disp)], axis=-1)
                    output_tensor["disp"][i].append(disp)
                    output_tensor["valid_disp"][i].append(valid_disp)

        return output_tensor

    def __getitem__(self, index):
        im_tensor = {"img"}
        sample = self.sample_list[index]
        if self.is_test:
            sample_size = len(sample["image"]["left"])
            im_tensor["img"] = [[] for _ in range(sample_size)]
            for i in range(sample_size):
                for cam in ["left", "right"]:
                    img = frame_utils.read_gen(sample["image"][cam][i])
                    img = np.array(img).astype(np.uint8)[..., :3]
                    img = torch.from_numpy(img).permute(2, 0, 1).float()
                    im_tensor["img"][i].append(img)
            im_tensor["img"] = torch.stack(im_tensor["img"])
            return im_tensor, self.extra_info[index]

        index = index % len(self.sample_list)

        try:
            output_tensor = self._get_output_tensor(sample)
        except:
            logging.warning(f"Exception in loading sample {index}!")
            index = np.random.randint(len(self.sample_list))
            logging.info(f"New index is {index}")
            sample = self.sample_list[index]
            output_tensor = self._get_output_tensor(sample)
        sample_size = len(sample["image"]["left"])

        if self.augmentor is not None:
            output_tensor["img"], output_tensor["disp"] = self.augmentor(
                output_tensor["img"], output_tensor["disp"]
            )
        for i in range(sample_size):
            for cam in (0, 1):
                if cam < len(output_tensor["img"][i]):
                    img = (
                        torch.from_numpy(output_tensor["img"][i][cam])
                        .permute(2, 0, 1)
                        .float()
                    )
                    if self.img_pad is not None:
                        padH, padW = self.img_pad
                        img = F.pad(img, [padW] * 2 + [padH] * 2)
                    output_tensor["img"][i][cam] = img

                if cam < len(output_tensor["disp"][i]):
                    disp = (
                        torch.from_numpy(output_tensor["disp"][i][cam])
                        .permute(2, 0, 1)
                        .float()
                    )

                    if self.sparse:
                        valid_disp = torch.from_numpy(
                            output_tensor["valid_disp"][i][cam]
                        )
                    else:
                        valid_disp = (
                            (disp[0].abs() < 512)
                            & (disp[1].abs() < 512)
                            & (disp[0].abs() != 0)
                        )
                    disp = disp[:1]

                    output_tensor["disp"][i][cam] = disp
                    output_tensor["valid_disp"][i][cam] = valid_disp.float()

                if "mask" in output_tensor and cam < len(output_tensor["mask"][i]):
                    mask = torch.from_numpy(output_tensor["mask"][i][cam]).float()
                    output_tensor["mask"][i][cam] = mask

                if "viewpoint" in output_tensor and cam < len(
                    output_tensor["viewpoint"][i]
                ):
                    viewpoint = output_tensor["viewpoint"][i][cam]
                    output_tensor["viewpoint"][i][cam] = viewpoint

        res = {}
        if "viewpoint" in output_tensor and self.split != "train":
            res["viewpoint"] = output_tensor["viewpoint"]
        if "metadata" in output_tensor and self.split != "train":
            res["metadata"] = output_tensor["metadata"]

        for k, v in output_tensor.items():
            if k != "viewpoint" and k != "metadata":
                for i in range(len(v)):
                    if len(v[i]) > 0:
                        v[i] = torch.stack(v[i])
                if len(v) > 0 and (len(v[0]) > 0):
                    res[k] = torch.stack(v)
        return res

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.sample_list = v * copy_of_self.sample_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.sample_list)


class DynamicReplicaDataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        root="./dynamic_replica_data",
        split="train",
        sample_len=-1,
        only_first_n_samples=-1,
    ):
        super(DynamicReplicaDataset, self).__init__(aug_params)
        self.root = root
        self.sample_len = sample_len
        self.split = split

        frame_annotations_file = f"frame_annotations_{split}.jgz"

        with gzip.open(
            osp.join(root, split, frame_annotations_file), "rt", encoding="utf8"
        ) as zipfile:
            frame_annots_list = load_dataclass(
                zipfile, List[DynamicReplicaFrameAnnotation]
            )
        seq_annot = defaultdict(lambda: defaultdict(list))
        for frame_annot in frame_annots_list:
            seq_annot[frame_annot.sequence_name][frame_annot.camera_name].append(
                frame_annot
            )

        for seq_name in seq_annot.keys():
            try:
                filenames = defaultdict(lambda: defaultdict(list))
                for cam in ["left", "right"]:
                    for framedata in seq_annot[seq_name][cam]:
                        im_path = osp.join(root, split, framedata.image.path)
                        depth_path = osp.join(root, split, framedata.depth.path)
                        mask_path = osp.join(root, split, framedata.mask.path)

                        assert os.path.isfile(im_path), im_path
                        assert os.path.isfile(depth_path), depth_path
                        assert os.path.isfile(mask_path), mask_path

                        filenames["image"][cam].append(im_path)
                        filenames["depth"][cam].append(depth_path)
                        filenames["mask"][cam].append(mask_path)

                        filenames["viewpoint"][cam].append(framedata.viewpoint)
                        filenames["metadata"][cam].append(
                            [framedata.sequence_name, framedata.image.size]
                        )

                        for k in filenames.keys():
                            assert (
                                len(filenames[k][cam])
                                == len(filenames["image"][cam])
                                > 0
                            ), framedata.sequence_name

                seq_len = len(filenames["image"][cam])

                print("seq_len", seq_name, seq_len)
                if split == "train":
                    for ref_idx in range(0, seq_len, 3):
                        step = 1 if self.sample_len == 1 else np.random.randint(1, 6)
                        if ref_idx + step * self.sample_len < seq_len:
                            sample = defaultdict(lambda: defaultdict(list))
                            for cam in ["left", "right"]:
                                for idx in range(
                                    ref_idx, ref_idx + step * self.sample_len, step
                                ):
                                    for k in filenames.keys():
                                        if "mask" not in k:
                                            sample[k][cam].append(
                                                filenames[k][cam][idx]
                                            )

                            self.sample_list.append(sample)
                else:
                    step = self.sample_len if self.sample_len > 0 else seq_len
                    counter = 0

                    for ref_idx in range(0, seq_len, step):
                        sample = defaultdict(lambda: defaultdict(list))
                        for cam in ["left", "right"]:
                            for idx in range(ref_idx, ref_idx + step):
                                for k in filenames.keys():
                                    sample[k][cam].append(filenames[k][cam][idx])

                        self.sample_list.append(sample)
                        counter += 1
                        if only_first_n_samples > 0 and counter >= only_first_n_samples:
                            break
            except Exception as e:
                print(e)
                print("Skipping sequence", seq_name)

        assert len(self.sample_list) > 0, "No samples found"
        print(f"Added {len(self.sample_list)} from Dynamic Replica {split}")
        logging.info(f"Added {len(self.sample_list)} from Dynamic Replica {split}")


class SequenceSceneFlowDataset(StereoSequenceDataset):
    def __init__(
        self,
        aug_params=None,
        root="./datasets",
        dstype="frames_cleanpass",
        sample_len=1,
        things_test=False,
        add_things=True,
        add_monkaa=True,
        add_driving=True,
    ):
        super(SequenceSceneFlowDataset, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype
        self.sample_len = sample_len
        if things_test:
            self._add_things("TEST")
        else:
            if add_things:
                self._add_things("TRAIN")
            if add_monkaa:
                self._add_monkaa()
            if add_driving:
                self._add_driving()

    def _add_things(self, split="TRAIN"):
        """Add FlyingThings3D data"""

        original_length = len(self.sample_list)
        root = osp.join(self.root, "FlyingThings3D")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(
                glob(osp.join(root, self.dstype, split, f"*/*/{cam}/"))
            )
            disparity_paths[cam] = [
                path.replace(self.dstype, "disparity") for path in image_paths[cam]
            ]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(image_paths["left"]))[:40])
        np.random.set_state(state)
        np.random.seed(0)
        num_seq = len(image_paths["left"])

        for seq_idx in range(num_seq):
            if (split == "TEST" and seq_idx in val_idxs) or (
                split == "TRAIN" and not seq_idx in val_idxs
            ):
                images, disparities = defaultdict(list), defaultdict(list)
                for cam in ["left", "right"]:
                    images[cam] = sorted(
                        glob(osp.join(image_paths[cam][seq_idx], "*.png"))
                    )
                    disparities[cam] = sorted(
                        glob(osp.join(disparity_paths[cam][seq_idx], "*.pfm"))
                    )

                self._append_sample(images, disparities)

        assert len(self.sample_list) > 0, "No samples found"
        print(
            f"Added {len(self.sample_list) - original_length} from FlyingThings {self.dstype}"
        )
        logging.info(
            f"Added {len(self.sample_list) - original_length} from FlyingThings {self.dstype}"
        )

    def _add_monkaa(self):
        """Add FlyingThings3D data"""

        original_length = len(self.sample_list)
        root = osp.join(self.root, "Monkaa")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(glob(osp.join(root, self.dstype, f"*/{cam}/")))
            disparity_paths[cam] = [
                path.replace(self.dstype, "disparity") for path in image_paths[cam]
            ]

        num_seq = len(image_paths["left"])

        for seq_idx in range(num_seq):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], "*.png")))
                disparities[cam] = sorted(
                    glob(osp.join(disparity_paths[cam][seq_idx], "*.pfm"))
                )

            self._append_sample(images, disparities)

        assert len(self.sample_list) > 0, "No samples found"
        print(
            f"Added {len(self.sample_list) - original_length} from Monkaa {self.dstype}"
        )
        logging.info(
            f"Added {len(self.sample_list) - original_length} from Monkaa {self.dstype}"
        )

    def _add_driving(self):
        """Add FlyingThings3D data"""

        original_length = len(self.sample_list)
        root = osp.join(self.root, "Driving")
        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(
                glob(osp.join(root, self.dstype, f"*/*/*/{cam}/"))
            )
            disparity_paths[cam] = [
                path.replace(self.dstype, "disparity") for path in image_paths[cam]
            ]

        num_seq = len(image_paths["left"])
        for seq_idx in range(num_seq):
            images, disparities = defaultdict(list), defaultdict(list)
            for cam in ["left", "right"]:
                images[cam] = sorted(glob(osp.join(image_paths[cam][seq_idx], "*.png")))
                disparities[cam] = sorted(
                    glob(osp.join(disparity_paths[cam][seq_idx], "*.pfm"))
                )

            self._append_sample(images, disparities)

        assert len(self.sample_list) > 0, "No samples found"
        print(
            f"Added {len(self.sample_list) - original_length} from Driving {self.dstype}"
        )
        logging.info(
            f"Added {len(self.sample_list) - original_length} from Driving {self.dstype}"
        )

    def _append_sample(self, images, disparities):
        seq_len = len(images["left"])
        for ref_idx in range(0, seq_len - self.sample_len):
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                for idx in range(ref_idx, ref_idx + self.sample_len):
                    sample["image"][cam].append(images[cam][idx])
                    sample["disparity"][cam].append(disparities[cam][idx])
            self.sample_list.append(sample)

            sample = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                for idx in range(ref_idx, ref_idx + self.sample_len):
                    sample["image"][cam].append(images[cam][seq_len - idx - 1])
                    sample["disparity"][cam].append(disparities[cam][seq_len - idx - 1])
            self.sample_list.append(sample)


class SequenceSintelStereo(StereoSequenceDataset):
    def __init__(
        self,
        dstype="clean",
        aug_params=None,
        root="./datasets",
    ):
        super().__init__(
            aug_params, sparse=True, reader=frame_utils.readDispSintelStereo
        )
        self.dstype = dstype
        original_length = len(self.sample_list)
        image_root = osp.join(root, "sintel_stereo", "training")

        image_paths = defaultdict(list)
        disparity_paths = defaultdict(list)

        for cam in ["left", "right"]:
            image_paths[cam] = sorted(
                glob(osp.join(image_root, f"{self.dstype}_{cam}/*"))
            )

        cam = "left"
        disparity_paths[cam] = [
            path.replace(f"{self.dstype}_{cam}", "disparities")
            for path in image_paths[cam]
        ]

        num_seq = len(image_paths["left"])
        # for each sequence
        for seq_idx in range(num_seq):
            sample = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]:
                sample["image"][cam] = sorted(
                    glob(osp.join(image_paths[cam][seq_idx], "*.png"))
                )
            cam = "left"
            sample["disparity"][cam] = sorted(
                glob(osp.join(disparity_paths[cam][seq_idx], "*.png"))
            )
            for im1, disp in zip(sample["image"][cam], sample["disparity"][cam]):
                assert (
                    im1.split("/")[-1].split(".")[0]
                    == disp.split("/")[-1].split(".")[0]
                ), (im1.split("/")[-1].split(".")[0], disp.split("/")[-1].split(".")[0])
            self.sample_list.append(sample)

        logging.info(
            f"Added {len(self.sample_list) - original_length} from SintelStereo {self.dstype}"
        )


def fetch_dataloader(args):
    """Create the data loader for the corresponding trainign set"""

    aug_params = {
        "crop_size": args.image_size,
        "min_scale": args.spatial_scale[0],
        "max_scale": args.spatial_scale[1],
        "do_flip": False,
        "yjitter": not args.noyjitter,
    }
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None

    add_monkaa = "monkaa" in args.train_datasets
    add_driving = "driving" in args.train_datasets
    add_things = "things" in args.train_datasets
    add_dynamic_replica = "dynamic_replica" in args.train_datasets

    new_dataset = None

    if add_monkaa or add_driving or add_things:
        clean_dataset = SequenceSceneFlowDataset(
            aug_params,
            dstype="frames_cleanpass",
            sample_len=args.sample_len,
            add_monkaa=add_monkaa,
            add_driving=add_driving,
            add_things=add_things,
        )

        final_dataset = SequenceSceneFlowDataset(
            aug_params,
            dstype="frames_finalpass",
            sample_len=args.sample_len,
            add_monkaa=add_monkaa,
            add_driving=add_driving,
            add_things=add_things,
        )

        new_dataset = clean_dataset + final_dataset

    if add_dynamic_replica:
        dr_dataset = DynamicReplicaDataset(
            aug_params, split="train", sample_len=args.sample_len
        )
        if new_dataset is None:
            new_dataset = dr_dataset
        else:
            new_dataset = new_dataset + dr_dataset

    logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
    train_dataset = (
        new_dataset if train_dataset is None else train_dataset + new_dataset
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    logging.info("Training with %d image pairs" % len(train_dataset))
    return train_loader
