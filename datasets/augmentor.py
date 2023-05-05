# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
from PIL import Image

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from torchvision.transforms import ColorJitter, functional, Compose


class AdjustGamma(object):
    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = (
            gamma_min,
            gamma_max,
            gain_min,
            gain_max,
        )

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"


class SequenceDispFlowAugmentor:
    def __init__(
        self,
        crop_size,
        min_scale=-0.2,
        max_scale=0.5,
        do_flip=True,
        yjitter=False,
        saturation_range=[0.6, 1.4],
        gamma=[1, 1, 1, 1],
    ):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose(
            [
                ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=saturation_range,
                    hue=0.5 / 3.14,
                ),
                AdjustGamma(*gamma),
            ]
        )
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, seq):
        """Photometric augmentation"""

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            for i in range(len(seq)):
                for cam in (0, 1):
                    seq[i][cam] = np.array(
                        self.photo_aug(Image.fromarray(seq[i][cam])), dtype=np.uint8
                    )
        # symmetric
        else:
            image_stack = np.concatenate(
                [seq[i][cam] for i in range(len(seq)) for cam in (0, 1)], axis=0
            )
            image_stack = np.array(
                self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8
            )
            split = np.split(image_stack, len(seq) * 2, axis=0)
            for i in range(len(seq)):
                seq[i][0] = split[2 * i]
                seq[i][1] = split[2 * i + 1]
        return seq

    def eraser_transform(self, seq, bounds=[50, 100]):
        """Occlusion augmentation"""
        ht, wd = seq[0][0].shape[:2]
        for i in range(len(seq)):
            for cam in (0, 1):
                if np.random.rand() < self.eraser_aug_prob:
                    mean_color = np.mean(seq[0][0].reshape(-1, 3), axis=0)
                    for _ in range(np.random.randint(1, 3)):
                        x0 = np.random.randint(0, wd)
                        y0 = np.random.randint(0, ht)
                        dx = np.random.randint(bounds[0], bounds[1])
                        dy = np.random.randint(bounds[0], bounds[1])
                        seq[i][cam][y0 : y0 + dy, x0 : x0 + dx, :] = mean_color

        return seq

    def spatial_transform(self, img, disp):
        # randomly sample scale
        ht, wd = img[0][0].shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd)
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            for i in range(len(img)):
                for cam in (0, 1):
                    img[i][cam] = cv2.resize(
                        img[i][cam],
                        None,
                        fx=scale_x,
                        fy=scale_y,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    if len(disp[i]) > 0:
                        disp[i][cam] = cv2.resize(
                            disp[i][cam],
                            None,
                            fx=scale_x,
                            fy=scale_y,
                            interpolation=cv2.INTER_LINEAR,
                        )
                        disp[i][cam] = disp[i][cam] * [scale_x, scale_y]

        if self.yjitter:
            y0 = np.random.randint(2, img[0][0].shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img[0][0].shape[1] - self.crop_size[1] - 2)

            for i in range(len(img)):
                y1 = y0 + np.random.randint(-2, 2 + 1)
                img[i][0] = img[i][0][
                    y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
                ]
                img[i][1] = img[i][1][
                    y1 : y1 + self.crop_size[0], x0 : x0 + self.crop_size[1]
                ]
                if len(disp[i]) > 0:
                    disp[i][0] = disp[i][0][
                        y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
                    ]
                    disp[i][1] = disp[i][1][
                        y1 : y1 + self.crop_size[0], x0 : x0 + self.crop_size[1]
                    ]
        else:
            y0 = np.random.randint(0, img[0][0].shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img[0][0].shape[1] - self.crop_size[1])
            for i in range(len(img)):
                for cam in (0, 1):
                    img[i][cam] = img[i][cam][
                        y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
                    ]
                    if len(disp[i]) > 0:
                        disp[i][cam] = disp[i][cam][
                            y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]
                        ]

        return img, disp

    def __call__(self, img, disp):
        img = self.color_transform(img)
        img = self.eraser_transform(img)
        img, disp = self.spatial_transform(img, disp)

        for i in range(len(img)):
            for cam in (0, 1):
                img[i][cam] = np.ascontiguousarray(img[i][cam])
                if len(disp[i]) > 0:
                    disp[i][cam] = np.ascontiguousarray(disp[i][cam])

        return img, disp
