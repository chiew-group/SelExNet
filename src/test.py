#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2024-09-21 01:52:10
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2024-11-29 22:32:15
FilePath     : /Documents/DeepControlV4/src/test.py
Description  : Testing script for the DeepControlV2 model
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm
from scipy import signal
from pytorch_msssim import SSIM
from src.utils import make_if_dont_exist
import os
import cv2
import json
from typing import Tuple, Optional

__all__ = ["Tester"]

SCALE = torch.tensor(9.74e-6, dtype=torch.float32)

class Tester(object):
    def __init__(
        self, cfg: DictConfig, model: nn.Module, bloch_sim: object, device: torch.device
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.bloch_sim = bloch_sim
        self.device = device
        self.test_dir = "data/test"
        self.exp_dir = os.path.join(self.test_dir, self.cfg.exp_name)
        self.sim_img_dir = os.path.join(self.exp_dir, "sim_img")
        self.pulse_dir = os.path.join(self.exp_dir, "pulse")
        self.npz_dir = os.path.join(self.exp_dir, "npz")
        self.phi_map_dir = os.path.join(self.exp_dir, "phi_map")
        self.prepare_dirs()
        self.prepare_grad()
        ckpt = torch.load(
            f"data/train/{self.cfg.exp_name}/model/model.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])

    def prepare_dirs(self) -> None:
        make_if_dont_exist(self.test_dir)
        make_if_dont_exist(self.exp_dir)
        make_if_dont_exist(self.sim_img_dir)
        make_if_dont_exist(self.pulse_dir)
        make_if_dont_exist(self.npz_dir)
        make_if_dont_exist(self.phi_map_dir)
    
    def prepare_grad(self) -> None:
        # Resample gradients to match RF pulse length
        N_sample = int(self.cfg.model.out_dim // self.cfg.magnet.tx)
        grad_x, grad_y = self.bloch_sim.pulse_read()
        grad_x_scipy = signal.resample(grad_x.cpu().numpy(), N_sample)
        grad_y_scipy = signal.resample(grad_y.cpu().numpy(), N_sample)
        self.grad_x_torch = torch.tensor(
            grad_x_scipy,
            dtype=torch.float64,
            device=self.device,
        ).unsqueeze(0)
        self.grad_y_torch = torch.tensor(
            grad_y_scipy,
            dtype=torch.float64,
            device=self.device,
        ).unsqueeze(0)
    
    def run(
        self,
        test_loader: DataLoader,
    ) -> None:
        metrics = []
        metric_dict = {}
        self.model.eval()
        with torch.no_grad():
            with tqdm(
                test_loader,
                unit="batch",
                desc="Inference",
            ) as tdata:
                for batch_idx, (raw, preprocessed, _) in enumerate(tdata, start=1):
                    # Move data to device
                    raw = raw.to(self.device)
                    preprocessed = preprocessed.to(self.device)

                    # Forward pass
                    amp, phase = self.model(preprocessed)
                    # Expand gradients for batch processing if needed
                    grad_x_batch = self.grad_x_torch.expand(
                        amp.shape[0], -1
                    )  # Shape: (B, T)
                    grad_y_batch = self.grad_y_torch.expand(
                        amp.shape[0], -1
                    )

                    # Run Bloch simulation
                    pseudo_img, pseudo_phi = self.bloch_sim.run_simulation(
                        amp,
                        phase,
                        grad_x_batch,
                        grad_y_batch,
                        SCALE.to(self.device),
                        offset=self.cfg.magnet.offset,
                    )

                    # Compute metrics
                    ssim, psnr, dice, iou, nrmse = self.compute_metrics(
                        pseudo_img, raw    # type: ignore
                    )

                    # Update progress bar
                    tdata.set_postfix(
                        {
                            "ssim": f"{ssim:.4f}",
                            "psnr": f"{psnr:.2f}",
                            "dice": f"{dice:.4f}",
                            "iou": f"{iou:.4f}",
                            "nrmse": f"{nrmse:.4f}",
                        }
                    )

                    # Append metrics
                    metrics.append([ssim, psnr, dice, iou, nrmse])
                    metric_dict[f"batch_{batch_idx}"] = {
                        "SSIM": ssim,
                        "PSNR": psnr,
                        "Dice": dice,
                        "IoU": iou,
                        "NRMSE": nrmse,
                    }

                    # Detach and move tensors to CPU for saving
                    old_amp = amp.detach().cpu().numpy()  # Shape: (B, ...)
                    old_phase = phase.detach().cpu().numpy()  # Shape: (B, ...)
                    amp = amp  # Shape: (B, ...)
                    phase = phase  # Shape: (B, ...)

                    # Compute magnitude and phase angles
                    magnitude = torch.sqrt(amp**2 + phase**2)  # Shape: (B, ...)
                    phase_angle = torch.atan2(phase, amp)  # Shape: (B, ...)

                    # Convert pseudo_img to numpy for saving
                    pseudo_img_numpy = pseudo_img.squeeze(1).detach().cpu().numpy()  # Shape: (B, H, W)
                    pseudo_phi_map_numpy = pseudo_phi.squeeze(1).detach().cpu().numpy()  # Shape: (B, H, W)

                    # Iterate over each sample in the batch
                    for i in range(pseudo_img_numpy.shape[0]):
                        global_idx = (batch_idx - 1) * test_loader.batch_size + i
                        img_name = f"{global_idx:05d}.png"
                        pseudo_phi_map_name = f"{global_idx:05d}.npy"
                        raw_name = f"{global_idx:05d}_raw.png"
                        pulse_name = f"{global_idx:05d}.txt"
                        npz_name = f"{global_idx:05d}.npz"

                        # Paths
                        pulse_path = os.path.join(self.pulse_dir, pulse_name)
                        img_path = os.path.join(self.sim_img_dir, img_name)
                        raw_path = os.path.join(self.sim_img_dir, raw_name)
                        npz_path = os.path.join(self.npz_dir, npz_name)
                        phi_map_path = os.path.join(self.phi_map_dir, pseudo_phi_map_name)

                        # Save Pulse Data
                        magnitude_i = magnitude[i].cpu().numpy()
                        phase_angle_i = phase_angle[i].cpu().numpy()
                        with open(pulse_path, "w") as f:
                            f.write(f"{len(magnitude_i)}\n")
                            for x, y in zip(magnitude_i, phase_angle_i):
                                f.write(f"{x:.8f} {y:.8f}\n")

                        # Save Simulated Image
                        self.save_image(pseudo_img_numpy[i], img_path)

                        # Save Raw Image
                        raw_image = raw[i].detach().squeeze(0).cpu().numpy()
                        self.save_image(raw_image, raw_path)

                        # Save NPZ Data
                        np.savez(
                            npz_path,
                            amp=old_amp[i],
                            phase=old_phase[i],
                        )
                        
                        # Save Phi Map
                        np.save(phi_map_path, pseudo_phi_map_numpy[i])

        # After all batches are processed, compute and save mean metrics
        self.save_metrics(metrics, metric_dict)

    def save_metrics(self, metrics: list, metric_dict: dict) -> None:
        metrics_np = np.array(metrics)
        mean_metrics = {
            "SSIM": np.mean(metrics_np[:, 0]),
            "PSNR": np.mean(metrics_np[:, 1]),
            "Dice": np.mean(metrics_np[:, 2]),
            "IoU": np.mean(metrics_np[:, 3]),
            "NRMSE": np.mean(metrics_np[:, 4]),
        }

        print("Mean Metric")
        for key, value in mean_metrics.items():
            print(f"{key}: {value}")

        metric_dict["mean"] = {k: float(v) for k, v in mean_metrics.items()}
        test_root = self.exp_dir
        with open(os.path.join(test_root, "metrics.json"), "w") as ff:
            json.dump(metric_dict, ff, indent=4)

    # Define the compute_metrics function
    def compute_metrics(
        self,
        simulated_image: torch.Tensor,
        target_image: torch.Tensor,
    ) -> Tuple[float, float, float, float, float]:
        smooth = 1e-12
        # SSIM
        ssim = SSIM(
            win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1
        )(simulated_image, target_image).item()

        # PSNR
        max_value = torch.max(target_image).item()
        mse_loss = nn.MSELoss()(simulated_image, target_image)
        psnr = 10 * torch.log10(max_value / mse_loss).item()

        # Dice Coefficient
        intersection = (simulated_image * target_image).sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (
            simulated_image.sum(dim=(2, 3)) + target_image.sum(dim=(2, 3)) + smooth
        )
        dice = dice.mean().item()

        # IoU
        intersection = (simulated_image * target_image).sum(dim=(2, 3))
        union = (
            simulated_image.sum(dim=(2, 3))
            + target_image.sum(dim=(2, 3))
            - intersection
        )
        iou = (intersection + smooth) / (union + smooth)
        iou = iou.mean().item()

        # NRMSE
        nrmse = self.nrmse_torch(simulated_image, target_image, normalizer="range")
        return ssim, psnr, dice, iou, nrmse

    def save_image(self, img: np.ndarray, img_path: str) -> None:
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(img_path, img)

    def nrmse_torch(self, y_pred, y_true, normalizer="range"):
        # Compute RMSE
        rmse = torch.sqrt(nn.MSELoss()(y_pred, y_true))

        # Normalization options
        if normalizer == "range":
            norm_value = torch.max(y_true) - torch.min(y_true)
        elif normalizer == "mean":
            norm_value = torch.mean(y_true)
        elif normalizer == "std":
            norm_value = torch.std(y_true)
        elif normalizer == "l2":
            norm_value = torch.norm(y_true, p=2)
        else:
            raise ValueError("Invalid normalizer type. Use 'range', 'mean', or 'std'.")

        # Compute NRMSE
        nrmse_value = rmse / norm_value
        return nrmse_value.item()
