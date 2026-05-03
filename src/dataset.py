#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2024-11-29 16:28:55
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-01-26 05:07:34
FilePath     : /Documents/DeepControlUltra/src/dataset.py
Description  :
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
from torch.utils.data import Dataset
from scipy.spatial import ConvexHull
import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List
from PIL import Image
import glob

__all__ = [
    "apply_bilateral_filter",
    "ROIDataset",
    "ROIOldDataset",
    "RandomContourDataset",
]


def apply_bilateral_filter(
    image: np.ndarray, d: int = 9, sigma_color: int = 75, sigma_space: int = 75
) -> np.ndarray:
    """Apply a bilateral filter to the image."""
    return cv2.bilateralFilter(
        image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )


class ROIDataset(Dataset):
    def __init__(
        self,
        cfg,
        apply_bilateral_filter: bool = False,
        filter_params: Optional[Dict[str, int]] = None,
    ) -> None:
        self.cfg = cfg
        self.apply_bilateral_filter = apply_bilateral_filter
        self.filter_params = (
            filter_params
            if filter_params is not None
            else {"d": 3, "sigma_color": 20, "sigma_space": 20}
        )
        self.images = self.load_data()
        self.fa = torch.sin(
            torch.deg2rad(torch.tensor(cfg.magnet.fa, dtype=torch.float32))
        )

    def load_data(self) -> List:
        images = []
        for image_path in glob.glob(self.cfg.image.dir + "/*.png"):
            if len(images) >= int(self.cfg.train.num_samples):
                break
            image = Image.open(image_path).convert("L")
            image = np.array(image)
            images.append(image)

        return images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_image = self.images[idx].astype(np.float32) / 255.0
        mask = (
            np.load(self.cfg.magnet.mask_path).astype(np.float32)
            if self.cfg.magnet.mask_path is not None
            else (raw_image > 0).astype(np.float32)
        )
        # Apply bilateral filter if required
        if self.apply_bilateral_filter:
            preprocessed_image = apply_bilateral_filter(
                image=raw_image,
                d=self.filter_params["d"],
                sigma_color=self.filter_params["sigma_color"],
                sigma_space=self.filter_params["sigma_space"],
            ).astype(np.float32)
        else:
            preprocessed_image = raw_image
        # Convert to a tensor and add channel dimension
        raw = self.fa * torch.from_numpy(raw_image).unsqueeze(
            0
        )  # Add channel dimension, target flip angle map
        preprocessed = self.fa * torch.from_numpy(preprocessed_image).unsqueeze(
            0
        )  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        return raw.float(), preprocessed.float(), mask.float()


class ROIOldDataset(Dataset):
    def __init__(
        self,
        cfg,
        apply_bilateral_filter: bool = False,
        filter_params: Optional[Dict[str, int]] = None,
    ) -> None:
        self.cfg = cfg
        self.apply_bilateral_filter = apply_bilateral_filter
        self.filter_params = (
            filter_params
            if filter_params is not None
            else {"d": 3, "sigma_color": 20, "sigma_space": 20}
        )
        self.images, self.masks = self.load_data()
        self.fa = torch.sin(
            torch.deg2rad(torch.tensor(cfg.magnet.fa, dtype=torch.float32))
        )

    def load_data(self) -> List:
        images = []
        masks = []
        for image_path in glob.glob(self.cfg.image.dir + "/*.png"):
            if len(images) >= int(self.cfg.train.num_samples):
                break
            image = Image.open(image_path)
            image = np.array(image)
            mask = np.array(image[..., 0])
            roi = np.array(image[..., 1])
            images.append(roi)
            masks.append(mask)

        return images, masks

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_image = self.images[idx].astype(np.float32) / 255.0
        mask = self.masks[idx].astype(np.float32) / 255.0

        # Apply bilateral filter if required
        if self.apply_bilateral_filter:
            preprocessed_image = apply_bilateral_filter(
                image=raw_image,
                d=self.filter_params["d"],
                sigma_color=self.filter_params["sigma_color"],
                sigma_space=self.filter_params["sigma_space"],
            ).astype(np.float32)
        else:
            preprocessed_image = raw_image
        # Convert to a tensor and add channel dimension
        raw = self.fa * torch.from_numpy(raw_image).unsqueeze(
            0
        )  # Add channel dimension, target flip angle map
        preprocessed = torch.from_numpy(preprocessed_image).unsqueeze(
            0
        )  # Add channel dimension
        mask = (
            torch.from_numpy(mask).unsqueeze(0).to(torch.float32)
        )  # Add channel dimension
        return raw, preprocessed, mask


class RandomContourDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 64,
        apply_bilateral_filter: bool = False,
        filter_params: Optional[Dict[str, int]] = None,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.apply_bilateral_filter = apply_bilateral_filter
        self.filter_params = (
            filter_params
            if filter_params is not None
            else {"d": 3, "sigma_color": 20, "sigma_space": 20}
        )
        self.images = self._generate_images()

    def _generate_images(self) -> List:
        images = []
        for _ in range(self.num_samples):
            image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

            # Generate random points for the contour
            num_points = np.random.randint(3, 10)
            points = np.random.rand(num_points, 2) * (self.image_size - 1)

            # Create a convex hull to ensure a closed contour
            hull = ConvexHull(points)
            contour = points[hull.vertices].astype(np.int32)

            # Draw the filled contour
            cv2.fillPoly(image, [contour], 255)

            images.append(image)

        return images

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_image = self.images[idx].astype(np.float32) / 255.0

        # Apply bilateral filter if required
        if self.apply_bilateral_filter:
            preprocessed_image = apply_bilateral_filter(
                raw_image,
                d=self.filter_params["d"],
                sigma_color=self.filter_params["sigma_color"],
                sigma_space=self.filter_params["sigma_space"],
            ).astype(np.float32)
        else:
            preprocessed_image = raw_image
        # Convert to a tensor and add channel dimension
        raw = torch.from_numpy(raw_image).unsqueeze(0)  # Add channel dimension
        preprocessed = torch.from_numpy(preprocessed_image).unsqueeze(
            0
        )  # Add channel dimension
        return raw, preprocessed
