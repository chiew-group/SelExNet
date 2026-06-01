#!/usr/bin/env python
# coding=utf-8

import os
import datetime
from typing import Tuple
import numpy as np
from scipy import signal
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim import lr_scheduler
from tqdm.auto import tqdm
# from pytorch_msssim import SSIM

from selexnet.utils import (
    plot_progress,
    make_if_dont_exist,
    setup_logger,
    FMGenerator,
    SSIM,
)
from selexnet.ddp import ddp_barrier, ddp_is_initialized

__all__ = ["Trainer"]


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        bloch_sim: object,
        device: torch.device,
        is_main_process: bool = True,
        local_rank: int = 0,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.bloch_sim = bloch_sim
        self.device = device
        self.is_main_process = is_main_process
        self.local_rank = local_rank
        self.logger = None

        # DDP state
        self.use_ddp = ddp_is_initialized()
        self.rank = dist.get_rank() if self.use_ddp else 0
        self.world_size = dist.get_world_size() if self.use_ddp else 1

        # Training state
        self.start = 0
        self.train_losses = []
        self.valid_losses = []
        self.best_val_loss = float("inf")

        # Optional grad clip (default disabled)
        # In YAML: train.grad_clip_norm: 0.0  (or omit)
        self.grad_clip_norm = float(getattr(cfg.train, "grad_clip_norm", 0.0) or 0.0)

        # Losses
        self.iou = BinaryJaccardLossND(smooth=1e-8, per_channel=True)
        
        # Target FA (for conditional IoU loss)
        self.target_fa = torch.sin(torch.deg2rad(torch.tensor(float(getattr(cfg.magnet, "fa", 90.0)))))

        # B0 / B1 maps (kept FP32)
        self.rf_scale = torch.tensor(
            self.cfg.magnet.rf_scale, device=device, dtype=torch.float32
        )
        if not self.cfg.magnet.use_fm_generator:
            si = np.load(cfg.magnet.si_path)
            self.b0_map = torch.tensor(si["b0"], dtype=torch.float32).unsqueeze(
                0
            )  # [1,H,W]
            self.b1_map_re = torch.tensor(si["b1_real"], dtype=torch.float32).unsqueeze(
                0
            )  # [1,Tx,H,W]
            self.b1_map_im = torch.tensor(si["b1_imag"], dtype=torch.float32).unsqueeze(
                0
            )  # [1,Tx,H,W]
        else:
            self.fm_tr_gen = FMGenerator(
                cfg,
                mode="train",
            )
            self.fm_val_gen = FMGenerator(
                cfg,
                mode="valid",
            )
            # -------------------------
            # Field-map caching (for robust training)
            # -------------------------
            self.fm_resample_every_epochs = int(
                getattr(cfg.train, "fm_resample_every_epochs", 1)
            )
            self._epoch_tr_fm_cache = None  # (B0, B1_re, B1_im) for training epoch
            self._val_fm_cache = (
                None  # (B0, B1_re, B1_im) fixed for whole training run (rank0)
            )

        # Directories
        self.train_dir = "outputs"
        self.exp_dir = os.path.join(self.train_dir, self.cfg.exp_name)
        self.log_dir = os.path.join(self.exp_dir, "log")
        self.model_dir = os.path.join(self.exp_dir, "model")
        self.plot_dir = os.path.join(self.exp_dir, "plot")
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")

        if not self.cfg.train.joint:
            self._prepare_G()

    # -------------------------
    # Setup
    # -------------------------
    def _prepare_G(self):
        # Resample gradients to match RF pulse length
        N_sample = int(self.cfg.model.rf_output_dim)
        grad_x, grad_y, grad_z = self.bloch_sim.pulse_read()
        if grad_x.shape[0] != N_sample:
            grad_x_scipy = signal.resample(grad_x.cpu().numpy(), N_sample)
            grad_y_scipy = signal.resample(grad_y.cpu().numpy(), N_sample)
            grad_z_scipy = signal.resample(grad_z.cpu().numpy(), N_sample)
        else:
            grad_x_scipy = grad_x.cpu().numpy()
            grad_y_scipy = grad_y.cpu().numpy()
            grad_z_scipy = grad_z.cpu().numpy()
        self.gx = torch.tensor(
            grad_x_scipy / 1000.0,
            dtype=torch.float32,
            device=self.device,
        )
        self.gy = torch.tensor(
            grad_y_scipy / 1000.0,
            dtype=torch.float32,
            device=self.device,
        )
        self.gz = torch.tensor(
            grad_z_scipy / 1000.0,
            dtype=torch.float32,
            device=self.device,
        )

    def _prepare_dir(self):
        if not self.is_main_process:
            return
        for d in [
            self.train_dir,
            self.exp_dir,
            self.log_dir,
            self.model_dir,
            self.plot_dir,
            self.checkpoint_dir,
        ]:
            make_if_dont_exist(d)

    def _prepare_logger(self, resume: bool = False):
        if not self.is_main_process:
            return
        log_name = (
            "training_log_"
            + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            + ".log"
        )
        self.logger = setup_logger("SelExNet", os.path.join(self.log_dir, log_name))
        self._log(f"{'Resume' if resume else 'Start'} {self.cfg.experiment} Training")
        self._log(
            f"Using device: {self.device} | DDP={self.use_ddp} rank={self.rank}/{self.world_size} | "
            f"Deterministic={self.cfg.train.deterministic} | "
            f"TF32={torch.backends.cuda.matmul.allow_tf32} | "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark} | "
            f"cudnn.deterministic={torch.backends.cudnn.deterministic}"
        )

    # -------------------------
    # Loop
    # -------------------------
    def run(self, train_loader: DataLoader, valid_loader: DataLoader):
        for epoch in range(self.start, self.cfg.train.epochs):
            # DDP shuffle control (train only)
            if self.use_ddp and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            if self.is_main_process:
                self._log(f"Epoch {epoch + 1}/{self.cfg.train.epochs}")

            if self.cfg.magnet.use_fm_generator:
                if (
                    self._epoch_tr_fm_cache is None
                    or (epoch - self.start) % self.fm_resample_every_epochs == 0
                ):
                    self._refresh_epoch_field_maps()

                    if self.is_main_process:
                        self._log(
                            f"[FM] Resampled training field maps at epoch {epoch + 1}"
                        )

            self.model.train()
            epoch_losses = []

            loader = self._tqdm_wrap(
                train_loader,
                unit="batch",
                desc=f"Epoch {epoch + 1}/{self.cfg.train.epochs}",
            )

            for raw, preprocessed, mask in loader:
                self.optimizer.zero_grad(set_to_none=True)

                raw = raw.to(self.device, non_blocking=True)
                preprocessed = preprocessed.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                if not self.cfg.magnet.use_fm_generator:
                    assert (
                        self.b0_map is not None
                        and self.b1_map_re is not None
                        and self.b1_map_im is not None
                    )
                    B0 = self.b0_map.expand(raw.size(0), -1, -1).to(
                        self.device, non_blocking=True
                    )  # [B,H,W]
                    B1_re = self.b1_map_re.expand(raw.size(0), -1, -1, -1).to(
                        self.device, non_blocking=True
                    )  # [B,Tx,H,W]
                    B1_im = self.b1_map_im.expand(raw.size(0), -1, -1, -1).to(
                        self.device, non_blocking=True
                    )  # [B,Tx,H,W]
                    B1 = torch.hypot(B1_re.sum(dim=1), B1_im.sum(dim=1))  # [B,H,W]
                else:
                    # ---- Epoch-level field-map reuse ----
                    bs = raw.size(0)
                    B0, B1_re, B1_im = self._get_epoch_field_maps(bs)

                    # Derived magnitude map (same as before)
                    B1 = torch.hypot(B1_re.sum(dim=1), B1_im.sum(dim=1))  # [B,H,W]

                model_input = torch.cat(
                    [
                        preprocessed,
                        B0.unsqueeze(1),
                        B1.unsqueeze(1),
                    ],
                    dim=1,
                )

                re, im, gx, gy, gz = self.model(model_input)
                if gx is None and gy is None and gz is None:
                    gx = self.gx
                    gy = self.gy
                    gz = self.gz

                pseudo_img, _ = self.bloch_sim(
                    re,
                    im,
                    gx,
                    gy,
                    B0,
                    B1_re,
                    B1_im,
                    self.rf_scale,
                    offset=self.cfg.magnet.offset_hz,
                )

                loss, cycle, psnr, ssim, iou = self._criterion(
                    pseudo_img,
                    raw,
                    re,
                    im,
                    gx,
                    gy,
                )

                loss.backward()

                # Optional grad clip (disabled by default)
                if self.grad_clip_norm > 0:
                    m = (
                        self.model.module
                        if hasattr(self.model, "module")
                        else self.model
                    )
                    torch.nn.utils.clip_grad_norm_(
                        m.parameters(), max_norm=self.grad_clip_norm
                    )

                self.optimizer.step()

                epoch_losses.append(float(loss.item()))

                if self.is_main_process and hasattr(loader, "set_postfix"):
                    loader.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "cycle": f"{float(cycle):.4f}",
                            "psnr": f"{float(psnr):.4f}",
                            "ssim": f"{float(ssim):.4f}",
                            "iou": f"{float(iou):.4f}",
                        }
                    )

            avg_train_loss = self._reduce_mean(float(np.mean(epoch_losses)))
            self.train_losses.append([epoch + 1, avg_train_loss])
            self._log(f"Training Loss: {avg_train_loss:.4f}")

            self._validate(epoch, valid_loader)

            self._log(f"lr: {self.scheduler.optimizer.param_groups[0]['lr']:.3e}")
            self._save_checkpoint(epoch)

            self._save(
                plot_progress,
                self.logger,
                self.plot_dir,
                self.train_losses,
                self.valid_losses,
                "loss",
            )

            torch.cuda.empty_cache()

    # -------------------------
    # Validation (rank0 only)
    # -------------------------
    def _validate(self, epoch: int, valid_loader: DataLoader):
        if (epoch + 1) % self.cfg.train.val_iter != 0:
            return

        avg_val_loss = None

        if self.is_main_process:
            self.model.eval()
            epoch_losses = []
            totals = dict(ssim=0.0, psnr=0.0, dice=0.0, iou=0.0, rmse=0.0)

            with torch.no_grad():
                loader = tqdm(valid_loader, desc="Validation")  # rank0 only
                for raw, preprocessed, _ in loader:
                    raw = raw.to(self.device, non_blocking=True)
                    preprocessed = preprocessed.to(self.device, non_blocking=True)

                    if not self.cfg.magnet.use_fm_generator:
                        assert (
                            self.b0_map is not None
                            and self.b1_map_re is not None
                            and self.b1_map_im is not None
                        )
                        B0 = self.b0_map.expand(raw.size(0), -1, -1).to(
                            self.device, non_blocking=True
                        )  # [B,H,W]
                        B1_re = self.b1_map_re.expand(raw.size(0), -1, -1, -1).to(
                            self.device, non_blocking=True
                        )  # [B,Tx,H,W]
                        B1_im = self.b1_map_im.expand(raw.size(0), -1, -1, -1).to(
                            self.device, non_blocking=True
                        )  # [B,Tx,H,W]
                        B1 = torch.hypot(B1_re.sum(dim=1), B1_im.sum(dim=1))  # [B,H,W]
                    else:
                        assert self.fm_val_gen is not None
                        # Sample ONCE for entire training run (stable validation physics)
                        if self._val_fm_cache is None:
                            # choose a fixed batch size for the cache
                            bs_full = int(self.cfg.train.batch_size)
                            self._val_fm_cache = self.fm_val_gen._sample_new_map(
                                batch_size=bs_full, device=self.device
                            )

                        B0_v, B1r_v, B1i_v = self._val_fm_cache

                        bs = raw.size(0)
                        B0 = B0_v[:bs]
                        B1_re = B1r_v[:bs]
                        B1_im = B1i_v[:bs]

                        B1 = torch.hypot(B1_re.sum(dim=1), B1_im.sum(dim=1))  # [B,H,W]

                    model_input = torch.cat(
                        [
                            preprocessed,
                            B0.unsqueeze(1),
                            B1.unsqueeze(1),
                        ],
                        dim=1,
                    )

                    re, im, gx, gy, gz = self.model(model_input)
                    if gx is None and gy is None and gz is None:
                        gx = self.gx
                        gy = self.gy
                        gz = self.gz

                    pseudo_img, _ = self.bloch_sim(
                        re,
                        im,
                        gx,
                        gy,
                        B0,
                        B1_re,
                        B1_im,
                        self.rf_scale,
                        offset=self.cfg.magnet.offset_hz,
                    )

                    loss, *_ = self._criterion(pseudo_img, raw, re, im, gx, gy)
                    epoch_losses.append(float(loss.item()))

                    ssim, psnr, dice, iou, rmse = self._compute_metrics(pseudo_img, raw)
                    totals["ssim"] += ssim
                    totals["psnr"] += psnr
                    totals["dice"] += dice
                    totals["iou"] += iou
                    totals["rmse"] += rmse

            avg_val_loss = float(np.mean(epoch_losses))
            self.valid_losses.append([epoch + 1, avg_val_loss])

            n = max(1, len(valid_loader))
            self._log(
                f"Validation Loss: {avg_val_loss:.4f} | "
                f"SSIM={totals['ssim'] / n:.4f}, PSNR={totals['psnr'] / n:.4f}, "
                f"DICE={totals['dice'] / n:.4f}, IoU={totals['iou'] / n:.4f}, RMSE={totals['rmse'] / n:.4f}"
            )

            # best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self._log(f"Saving Best Model with Loss: {self.best_val_loss:.4f}")
                self._save(
                    torch.save,
                    (
                        self.model.module.state_dict()
                        if hasattr(self.model, "module")
                        else self.model.state_dict()
                    ),
                    os.path.join(self.model_dir, "model.pt"),
                )

            # scheduler step on rank0
            self.scheduler.step(avg_val_loss)

        # Make all ranks wait so they stay in-sync by epoch
        if self.use_ddp:
            ddp_barrier(self.local_rank)

    # -------------------------
    # Save Checkpoint
    # -------------------------

    def _save_checkpoint(self, epoch: int):
        if not self.is_main_process:
            return
        ckpt = {
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.valid_losses,
            "model_state_dict": (
                self.model.module.state_dict()
                if hasattr(self.model, "module")
                else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(ckpt, os.path.join(self.checkpoint_dir, "ckpt.pt"))

    # -------------------------
    # Loss / Metrics
    # -------------------------
    def _criterion(self, pred, target, re, im, gx, gy):
        img_loss, cycle, psnr, ssim = self._image_reconstruction_loss(pred, target)
        iou_loss = self.iou(pred, target)
        pulse_loss = self._combined_regularization(
            re,
            im,
            self.cfg.train.lambda_rf_re_amp,
            self.cfg.train.lambda_rf_im_amp,
            self.cfg.train.lambda_rf_re_smooth,
            self.cfg.train.lambda_rf_im_smooth,
        )
        grad_loss = self._gradient_amplitude_penalty(
            gx, gy
        ) + self._gradient_slew_rate_penalty(gx, gy)
        total = img_loss + pulse_loss + grad_loss
        
        # IoU is meaningful when the target is a binary-ish mask scaled by sin(FA).
        # Compare against the known target maximum instead of fragile float equality.
        if abs(self.target_fa - 1.0) < 1e-4:
            total = total + self.cfg.train.lambda_iou * iou_loss

        return total, cycle.item(), psnr.item(), ssim.item(), iou_loss.item()

    def _compute_metrics(
        self,
        simulated_image: torch.Tensor,
        target_image: torch.Tensor,
        smooth: float = 1e-8,
    ) -> Tuple[float, float, float, float, float]:
        dims = _spatial_dims(simulated_image)  # reduce over spatial dims only
        max_val = torch.max(target_image)
        ssim_caller = SSIM(
            window_size=11,
            sigma=1.5,
            in_channels=1,
            L=max_val.item(),
        ).to(self.device)
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

        union = (
            simulated_image.sum(dim=dims) + target_image.sum(dim=dims) - intersection
        )
        iou = ((intersection + smooth) / (union + smooth)).mean().item()

        rmse = torch.sqrt(mse_loss + smooth).item()
        return ssim, psnr, dice, iou, rmse

    def _psnr_loss(
        self, pseudo_img: torch.Tensor, target_img: torch.Tensor, smooth: float = 1e-8
    ) -> torch.Tensor:
        max_val = torch.max(target_img)
        mse = F.mse_loss(pseudo_img, target_img)
        psnr = 10 * torch.log10(max_val**2 / (mse + smooth))
        return 1.0 / (psnr + smooth)

    def _ssim_loss(
        self, pseudo_img: torch.Tensor, target_img: torch.Tensor
    ) -> torch.Tensor:
        max_val = torch.max(target_img).item()
        ssim_caller = SSIM(window_size=11, sigma=1.5, in_channels=1, L=max_val).to(
            self.device
        )
        return 1.0 - ssim_caller(pseudo_img, target_img)

    def _image_reconstruction_loss(
        self, pseudo_img: torch.Tensor, target_img: torch.Tensor, smooth=1e-8
    ):
        cycle_loss = torch.sqrt(F.mse_loss(pseudo_img, target_img) + smooth)
        psnr = self._psnr_loss(pseudo_img, target_img, smooth=smooth)
        ssim = self._ssim_loss(pseudo_img, target_img)
        total_loss = (
            self.cfg.train.lambda_rmse * cycle_loss
            + self.cfg.train.lambda_ssim * ssim
            + self.cfg.train.lambda_psnr * psnr
        )
        return total_loss, cycle_loss, psnr, ssim

    def _combined_regularization(
        self,
        re: torch.Tensor,  # [B,Tx,T]
        im: torch.Tensor,  # [B,Tx,T]
        lambda_re_pulses: float = 0.1,
        lambda_im_pulses: float = 0.1,
        lambda_re_smooth: float = 0.1,
        lambda_im_smooth: float = 0.01,
    ) -> torch.Tensor:
        re_loss = lambda_re_pulses * self._magnitude_pulse_regularization(re)
        im_loss = lambda_im_pulses * self._magnitude_pulse_regularization(im)
        smoothness_re_loss = lambda_re_smooth * self._smoothness_pulse_regularization(
            re
        )
        smoothness_im_loss = lambda_im_smooth * self._smoothness_pulse_regularization(
            im
        )
        return re_loss + im_loss + smoothness_re_loss + smoothness_im_loss

    def _smoothness_pulse_regularization(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(torch.pow(input_tensor[..., 1:] - input_tensor[..., :-1], 2))

    def _magnitude_pulse_regularization(self, input_tensor: torch.Tensor):
        return torch.mean(torch.pow(input_tensor, 2))

    def _gradient_amplitude_penalty(self, g_x, g_y, lambda_grad=5e-1):
        beta = 50.0
        eps = 1e-6
        g_max = torch.tensor(
            self.cfg.magnet.gmax, device=g_x.device, dtype=torch.float32
        )

        g_x_smooth = torch.sqrt(g_x**2 + eps)
        g_y_smooth = torch.sqrt(g_y**2 + eps)

        penalty_x = F.softplus(beta * (g_x_smooth - g_max)) / beta
        penalty_y = F.softplus(beta * (g_y_smooth - g_max)) / beta
        loss_grad_amp = 0.5 * (penalty_x.mean() + penalty_y.mean())
        return lambda_grad * loss_grad_amp

    def _gradient_slew_rate_penalty(self, g_x, g_y, lambda_slew=5e-1):
        beta = 50.0
        eps = 1e-6
        out_dim = g_x.shape[-1]
        dt = self.cfg.magnet.tp / (out_dim - 1)
        slew_max = torch.tensor(
            self.cfg.magnet.smax, device=g_x.device, dtype=torch.float32
        )

        slew_x = torch.gradient(g_x, spacing=dt, dim=-1)[0]
        slew_y = torch.gradient(g_y, spacing=dt, dim=-1)[0]

        slew_x_smooth = torch.sqrt(slew_x**2 + eps)
        slew_y_smooth = torch.sqrt(slew_y**2 + eps)

        penalty_x = F.softplus(beta * (slew_x_smooth - slew_max)) / beta
        penalty_y = F.softplus(beta * (slew_y_smooth - slew_max)) / beta
        loss_grad_slew = 0.5 * (penalty_x.mean() + penalty_y.mean())
        return lambda_slew * loss_grad_slew

    # -------------------------
    # Helpers
    # -------------------------

    def _log(self, msg: str):
        if self.is_main_process and self.logger is not None:
            self.logger.info(msg)

    def _tqdm_wrap(self, loader, **kwargs):
        return tqdm(loader, **kwargs) if self.is_main_process else loader

    def _save(self, fn, *args, **kwargs):
        if self.is_main_process:
            fn(*args, **kwargs)

    def _reduce_mean(self, value: float) -> float:
        if not self.use_ddp:
            return value
        t = torch.tensor(value, device=self.device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= self.world_size
        return float(t.item())

    def _refresh_epoch_field_maps(self):
        """
        Sample ONE set of field maps for the whole epoch (training).
        We sample at full batch size, and slice per-iteration to match the current batch size.
        """
        assert self.fm_tr_gen is not None
        bs_full = int(self.cfg.train.batch_size)
        B0, B1_re, B1_im = self.fm_tr_gen._sample_new_map(
            batch_size=bs_full, device=self.device
        )
        self._epoch_tr_fm_cache = (B0, B1_re, B1_im)

    def _get_epoch_field_maps(self, bs: int):
        """
        Return (B0, B1_re, B1_im) with batch size = bs by slicing epoch cache.
        Works for last smaller batch too.
        """
        if self._epoch_tr_fm_cache is None:
            self._refresh_epoch_field_maps()

        B0_e, B1r_e, B1i_e = self._epoch_tr_fm_cache

        if bs > B0_e.size(0):
            # Should not happen if we sampled at cfg.train.batch_size >= bs
            raise ValueError(f"Requested bs={bs} but cached bs={B0_e.size(0)}")

        return B0_e[:bs], B1r_e[:bs], B1i_e[:bs]


def _spatial_dims(x: torch.Tensor) -> tuple[int, ...]:
    """
    Returns the spatial dims of a tensor assumed to be shaped (B, C, ...spatial...)
    Works for 2D, 3D, ... N-D.
    """
    if x.dim() < 3:
        raise ValueError(f"Expected tensor with shape (B,C,...) but got {x.shape}")
    return tuple(range(2, x.dim()))


class BinaryJaccardLossND(nn.Module):
    """
    Soft (differentiable) IoU/Jaccard loss for N-D images/volumes.

    Assumptions:
        - y_pred and y_true are in [0,1] (not necessarily binary)
        - shapes are (B, C, ...spatial...), typically C=1
    """

    def __init__(self, smooth: float = 1e-8, per_channel: bool = False):
        super().__init__()
        self.smooth = float(smooth)
        self.per_channel = bool(per_channel)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: pred {y_pred.shape} vs true {y_true.shape}"
            )

        dims = _spatial_dims(y_pred)  # reduce over spatial dims only

        intersection = (y_pred * y_true).sum(dim=dims)  # (B,C)
        union = y_pred.sum(dim=dims) + y_true.sum(dim=dims) - intersection  # (B,C)

        iou = (intersection + self.smooth) / (union + self.smooth)  # (B,C)

        if self.per_channel:
            return 1.0 - iou.mean()  # average over B and C
        else:
            return (
                1.0 - iou.mean(dim=1).mean()
            )  # mean over C then over B (robust if C>1)
