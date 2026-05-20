#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-03-28 16:55:13
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-04-17 22:14:37
FilePath     : /Documents/paper1/sTx/sTx_RF_SI/test_single.py
Description  :
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import argparse
import os
import torch
from PIL import Image
import numpy as np
from src import SelExNet, BlochSimTorch, SSIM, set_determinism, make_if_dont_exist
import matplotlib
from typing import Tuple
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.io import savemat

matplotlib.use("agg")


SCALE = torch.tensor(1.0, dtype=torch.float32)


def _spatial_dims(x: torch.Tensor) -> tuple[int, ...]:
    """
    Returns the spatial dims of a tensor assumed to be shaped (B, C, ...spatial...)
    Works for 2D, 3D, ... N-D.
    """
    if x.dim() < 3:
        raise ValueError(f"Expected tensor with shape (B,C,...) but got {x.shape}")
    return tuple(range(2, x.dim()))


def downsample_gradient_for_export(
    g: torch.Tensor,
    source_raster: float,
    target_raster: float,
) -> Tuple[torch.Tensor, int]:
    if target_raster <= 0:
        return g, 1
    ratio = float(target_raster) / float(source_raster)
    factor = int(round(ratio))
    if factor < 1 or abs(ratio - factor) > 1e-4:
        raise ValueError(
            "Gradient export raster must be an integer multiple of the adaptation "
            f"raster. Got source={source_raster:.6g}s, target={target_raster:.6g}s."
        )
    if factor == 1:
        return g, factor

    n = g.shape[-1]
    n_out = n // factor
    n_keep = n_out * factor
    if n_keep != n:
        print(
            f"Warning: dropping {n - n_keep} gradient sample(s) for "
            f"{factor}x export downsampling."
        )
    shape = (*g.shape[:-1], n_out, factor)
    return g[..., :n_keep].reshape(shape).mean(dim=-1).contiguous(), factor


def _compute_metrics(
    simulated_image: torch.Tensor,
    target_image: torch.Tensor,
    mask: torch.Tensor = None,
    smooth: float = 1e-8,
) -> Tuple[float, float, float, float, float]:
    dims = _spatial_dims(simulated_image)  # reduce over spatial dims only
    max_val = torch.max(target_image)
    ssim_caller = SSIM(
        window_size=11,
        sigma=1.5,
        in_channels=1,
        L=max_val.item(),
    ).to(simulated_image.device)
    ssim = ssim_caller(simulated_image, target_image).item()

    mse_loss = nn.MSELoss()(simulated_image, target_image)
    psnr = (10 * torch.log10(max_val**2 / (mse_loss + smooth))).item()

    intersection = (simulated_image * target_image).sum(dim=dims)
    dice = (
        (
            (2.0 * intersection + smooth)
            / (simulated_image.sum(dim=dims) + target_image.sum(dim=dims) + smooth)
        )
        .mean()
        .item()
    )

    union = simulated_image.sum(dim=dims) + target_image.sum(dim=dims) - intersection
    iou = ((intersection + smooth) / (union + smooth)).mean().item()

    rmse = torch.sqrt(mse_loss + smooth).item()

    mse_inner = nn.MSELoss()(mask * simulated_image, mask * target_image)
    mse_outer = nn.MSELoss()((1 - mask) * simulated_image, (1 - mask) * target_image)
    rmse_inner = torch.sqrt(mse_inner + smooth).item()
    rmse_outer = torch.sqrt(mse_outer + smooth).item()

    # simulated_image[simulated_image > 0.3] = 1
    num_signal_out = torch.sum(mask - target_image)
    num_signal_in = torch.sum(target_image)
    mean_signal_out = (
        torch.sum((mask - target_image) * simulated_image) / num_signal_out
    )
    mean_signal_in = torch.sum(target_image * simulated_image) / num_signal_in
    print(mean_signal_out.item(), mean_signal_in.item())
    bsr = mean_signal_out.item() / mean_signal_in.item()
    return ssim, psnr, dice, iou, rmse, rmse_inner, rmse_outer, bsr


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str, help="path to config file")
    parser.add_argument("--output_dir", type=str, help="directory to save outputs")
    parser.add_argument("--img_path", type=str, help="path to input image")
    return parser.parse_args()


def main():
    args = parse_command()
    cfg = OmegaConf.load(args.cfg)
    make_if_dont_exist(args.output_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_determinism(cfg.train.deterministic, cfg.train.seed, use_ddp=False)
    bs = BlochSimTorch(
        cfg.image.fov,
        cfg.image.N,
        cfg.magnet.tp,
        cfg.magnet.gamma,
        cfg.magnet.grad_path,
        block_steps=cfg.magnet.block_steps,
    ).to(device)

    si = np.load(cfg.magnet.si_path)
    b0_map_raw = si["b0"]
    b1_map_re = si["b1_real"]
    b1_map_im = si["b1_imag"]
    b0_map = (
        torch.tensor(b0_map_raw, dtype=torch.float32).unsqueeze(0).to(device)
    )  # Add channel dimension [1, 64, 64]
    b1_map_re = (
        torch.tensor(b1_map_re, dtype=torch.float32).unsqueeze(0).to(device)
    )  # Add channel dimension [1, C, 64, 64]
    b1_map_im = (
        torch.tensor(b1_map_im, dtype=torch.float32).unsqueeze(0).to(device)
    )  # Add channel dimension [1, C, 64, 64]
    b0_input = b0_map.unsqueeze(1)  # Add channel dimension [1, 1, 64, 64]
    b1_real_input = b1_map_re.sum(dim=1)  # Sum across channels [1, 64, 64]
    b1_imag_input = b1_map_im.sum(dim=1)  # Sum across channels [1, 64, 64]
    b1_input = torch.hypot(b1_real_input, b1_imag_input).unsqueeze(
        1
    )  # Use magnitude of B1 [1, 1, 64, 64]

    # Load the trained model
    model = SelExNet(cfg).to(device)

    ckpt = torch.load(
        os.path.join("outputs", cfg.exp_name, "model", "model.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt)
    model.eval()

    img = Image.open(args.img_path).convert("L")
    img = np.array(img).astype(np.float32) / 255.0

    raw_image = torch.tensor(img, device=device).unsqueeze(0).unsqueeze(0)
    mask = (
        torch.tensor(np.load(cfg.magnet.mask_path), device=device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    fa = torch.sin(torch.deg2rad(torch.tensor(cfg.magnet.fa, dtype=torch.float32)))
    raw_image = fa * raw_image

    # mask = torch.tensor(mask, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        input_data = torch.cat([raw_image, b0_input, b1_input], dim=1)  # [1, 3, 64, 64]

        re, im, gx, gy, _ = model(input_data)
        re = re * cfg.magnet.rf_scale
        im = im * cfg.magnet.rf_scale
        (
            simulated_image,
            _,
        ) = bs(
            re,
            im,
            gx,
            gy,
            b0_map,
            b1_map_re,
            b1_map_im,
            SCALE,
            offset=cfg.magnet.offset_hz,
        )

        # Compute metrics
        ssim, psnr, dice, iou, rmse, rmse_inner, rmse_outer, bsr = _compute_metrics(
            simulated_image, raw_image, mask=mask
        )
        print(
            f"SSIM: {ssim:.4f}, PSNR: {psnr:.4f}, Dice: {dice:.4f}, IoU: {iou:.4f}, RMSE: {rmse:.4f}, RMSE Inner: {rmse_inner:.4f}, RMSE Outer: {rmse_outer:.4f}, BSR: {bsr:.4f}"
        )
        simulated_image = (
            simulated_image.squeeze().detach().cpu().numpy().astype(np.float32)
        )
        pil_img = Image.fromarray((simulated_image * 255).astype(np.uint8))
        pil_img.save(f"{args.output_dir}/simulated_image.png")
        re = re.squeeze().detach().cpu().numpy()  # [C, 1430]
        im = im.squeeze().detach().cpu().numpy()  # [C, 1430]
        gx_down, _ = downsample_gradient_for_export(gx, 5e-6, 1e-5)
        gy_down, _ = downsample_gradient_for_export(gy, 5e-6, 1e-5)
        sx = torch.gradient(gx_down, spacing=10e-6, axis=-1)[0]
        sy = torch.gradient(gy_down, spacing=10e-6, axis=-1)[0]
        sxy = torch.hypot(sx, sy)
        print(
            "Max gradient slew rates (T/m/s): gx={:.2f}, gy={:.2f}, combined={:.2f}".format(
                torch.max(torch.abs(sx)).item(),
                torch.max(torch.abs(sy)).item(),
                torch.max(torch.abs(sxy)).item(),
            )
        )

        gx_down = gx_down.squeeze().detach().cpu().numpy()  # [1430//factor_x]
        gy_down = gy_down.squeeze().detach().cpu().numpy()  # [1430//factor_y]
        gx_export = np.zeros((gx_down.shape[0],), device=gx_down.device)
        gy_export = np.zeros_like(gx_export)

        gx_export[:] = gx_down
        gy_export[:] = gy_down

        gx_export[-1] = 0.0
        gy_export[-1] = 0.0

        rf_export = (
            (re + 1j * im).reshape((-1, 1), order="F").astype(np.complex64)
        )  # [C*1430, 1]
        gx_export = 1e3 * gx_export  # Convert to mT/m
        gy_export = 1e3 * gy_export  # Convert to mT/m
        grad_export = np.zeros((gx_export.shape[0], 3), dtype=np.float32)
        grad_export[:, 0] = gx_export
        grad_export[:, 1] = gy_export
        savemat(
            f"{args.output_dir}/rf_g.mat",
            {
                "rf": rf_export,
                "grad": grad_export,
            },
            appendmat=False,
        )


if __name__ == "__main__":
    main()
