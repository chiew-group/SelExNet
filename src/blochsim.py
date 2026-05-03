#!/usr/bin/env python
# coding=utf-8
"""
Optimized differentiable Bloch simulator with checkpointed time blocks (C3).

Key goals:
- Avoid materializing (B, Ny, Nx, T) tensors (B1 real/imag/mag/phase).
- Keep physics in FP32.
- Reduce autograd memory by checkpointing time integration in blocks.

This implementation is intended for training (autograd enabled). For inference,
you can disable checkpointing for maximum speed.
"""

from __future__ import annotations
import math
from typing import Sequence, Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

__all__ = ["BlochSimTorch"]


class BlochSimTorch(nn.Module):
    def __init__(
        self,
        fov: Sequence[float],
        N: Sequence[int],
        tp: float,
        gamma: float,
        grad_path: str,
        *,
        block_steps: int = 32,
        use_checkpoint: bool = True,
        use_reentrant: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        # Geometry / constants
        self.x_fov, self.y_fov = float(fov[0]), float(fov[1])
        self.Nx, self.Ny = int(N[0]), int(N[1])
        self.grad_path = grad_path

        # Simulation constants (stored as buffers so they move with .to(device))
        self.register_buffer("M0", torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32))
        self.register_buffer("tp", torch.tensor(float(tp), dtype=torch.float32))
        self.register_buffer("gamma", torch.tensor(float(gamma), dtype=torch.float32))
        self.register_buffer("pi", torch.tensor(math.pi, dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(float(eps), dtype=torch.float32))

        # Spatial grids (buffers)
        x_pos = torch.linspace(
            -self.x_fov / 2, self.x_fov / 2, self.Nx, dtype=torch.float32
        )
        y_pos = torch.linspace(
            -self.y_fov / 2, self.y_fov / 2, self.Ny, dtype=torch.float32
        )
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")
        self.register_buffer("x_grid_expand", x_grid.unsqueeze(0))  # (1, Ny, Nx)
        self.register_buffer("y_grid_expand", y_grid.unsqueeze(0))  # (1, Ny, Nx)

        # Checkpoint configuration
        self.block_steps = int(block_steps)
        self.use_checkpoint = bool(use_checkpoint)
        self.use_reentrant = bool(use_reentrant)

    # ------------------------
    # Helpers (FP32 physics)
    # ------------------------
    def rotate_x(
        self, My: torch.Tensor, Mz: torch.Tensor, angle: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        My_new = My * cos_angle - Mz * sin_angle
        Mz_new = My * sin_angle + Mz * cos_angle
        return My_new, Mz_new

    def rotate_y(
        self, Mx: torch.Tensor, Mz: torch.Tensor, angle: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        Mx_new = Mx * cos_angle + Mz * sin_angle
        Mz_new = -Mx * sin_angle + Mz * cos_angle
        return Mx_new, Mz_new

    def rotate_z(
        self, Mx: torch.Tensor, My: torch.Tensor, angle: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        Mx_new = Mx * cos_angle - My * sin_angle
        My_new = Mx * sin_angle + My * cos_angle
        return Mx_new, My_new

    def safe_atan2(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient-safe atan2 using radius normalization.
        Equivalent to the original TorchScript implementation.
        """
        eps = self.eps.to(dtype=y.dtype, device=y.device)
        inv_r = torch.rsqrt(x * x + y * y + eps * eps)
        return torch.atan2(y * inv_r, x * inv_r)

    # ------------------------
    # Grad file reader
    # ------------------------
    def pulse_read(self) -> Tuple[torch.Tensor, torch.Tensor]:
        g_x = []
        g_y = []

        with open(self.grad_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if len(lines) < 3:
            raise ValueError("Gradient file too short")

        # Line 0: number of points (can be ignored, but sanity-check is useful)
        n_pts = int(lines[0].split()[0])

        # Line 1: physical gradient limits
        parts = lines[1].split()
        if len(parts) < 2:
            raise ValueError("Gradient max line malformed")
        Gx_max = float(parts[0])
        Gy_max = float(parts[1])

        # Remaining lines: normalized gradients
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            gx_norm = float(parts[0])
            gy_norm = float(parts[1])
            g_x.append(Gx_max * gx_norm)
            g_y.append(Gy_max * gy_norm)

        gx = torch.tensor(g_x, dtype=torch.float32)
        gy = torch.tensor(g_y, dtype=torch.float32)

        if gx.numel() != n_pts:
            print(
                f"⚠️ Warning: gradient points mismatch "
                f"(file says {n_pts}, read {gx.numel()})"
            )

        return gx, gy

    # ------------------------
    # Main forward
    # ------------------------
    def forward(
        self,
        re: torch.Tensor,
        im: torch.Tensor,
        grad_x: torch.Tensor,
        grad_y: torch.Tensor,
        B0: torch.Tensor,
        coil_real: torch.Tensor,
        coil_imag: torch.Tensor,
        scaling_factor: torch.Tensor,
        offset: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            re, im: (B, T) real/imag RF waveform
            grad_x, grad_y: (B, T)
            scaling_factor: scalar or (B,) or broadcastable
            B0: (B, Ny, Nx) or broadcastable
            coil_real, coil_imag: (B, Ny, Nx)

        Returns:
            Mxy_abs: (B, 1, Ny, Nx)
            Mxy_phi: (B, 1, Ny, Nx)
        """
        # Force FP32 physics regardless of upstream AMP
        dtype = torch.float32
        device = re.device

        # Ensure shapes
        B = re.shape[0]
        C = re.shape[1]
        T = re.shape[2]

        Ny, Nx = self.Ny, self.Nx

        # grads can be (B,T) or (T,)
        grad_x = grad_x.to(dtype=dtype)
        grad_y = grad_y.to(dtype=dtype)

        scaling_factor = scaling_factor.to(dtype=dtype, device=device)
        B0 = B0.to(dtype=dtype)
        coil_real = coil_real.to(dtype=dtype)
        coil_imag = coil_imag.to(dtype=dtype)

        # scalar offset per batch
        offset_t = torch.full((B,), float(offset), dtype=dtype, device=device)

        # Time step
        dt = (self.tp.to(dtype=dtype, device=device)) / float(T)
        gamma_dt = self.gamma.to(dtype=dtype, device=device) * dt

        # Spatial grids (broadcasted)
        x_grid = self.x_grid_expand.to(device=device, dtype=dtype).expand(
            B, -1, -1
        )  # (B,Ny,Nx)
        y_grid = self.y_grid_expand.to(device=device, dtype=dtype).expand(B, -1, -1)

        # Initial magnetization
        Mx = torch.full(
            (B, Ny, Nx), float(self.M0[0].item()), dtype=dtype, device=device
        )
        My = torch.full(
            (B, Ny, Nx), float(self.M0[1].item()), dtype=dtype, device=device
        )
        Mz = torch.full(
            (B, Ny, Nx), float(self.M0[2].item()), dtype=dtype, device=device
        )

        # Helper to get grad at step m in shape (B,1,1)
        def _grad_step(g: torch.Tensor, m: int) -> torch.Tensor:
            if g.dim() == 2:
                return g[:, m].view(B, 1, 1)
            else:
                return g[m].view(1, 1, 1).expand(B, 1, 1)

        # One integration block (will be checkpointed)
        def _run_block(
            Mx_in,
            My_in,
            Mz_in,
            re_in,
            im_in,
            grad_x_in,
            grad_y_in,
            start: int,
            end: int,
        ):
            Mx_blk, My_blk, Mz_blk = Mx_in, My_in, Mz_in
            for m in range(start, end):
                # RF at step m (broadcast to (B,Ny,Nx))
                re_m = re_in[:, :, m].view(B, C, 1, 1)
                im_m = im_in[:, :, m].view(B, C, 1, 1)

                # Complex B1 from coil sensitivity (no (..,T) materialization)
                b1_re_m_c = re_m * coil_real - im_m * coil_imag
                b1_im_m_c = re_m * coil_imag + im_m * coil_real

                b1_re_m = b1_re_m_c.sum(dim=1)  # (B,Ny,Nx)
                b1_im_m = b1_im_m_c.sum(dim=1)  # (B,Ny,Nx)

                # Magnitude and phase
                b1_mag = scaling_factor.view(-1, 1, 1) * torch.hypot(b1_re_m, b1_im_m)
                phi_m = self.safe_atan2(b1_im_m, b1_re_m)

                # Grad offsets
                gx_m = _grad_step(grad_x_in, m)
                gy_m = _grad_step(grad_y_in, m)
                offset_x = gx_m * x_grid
                offset_y = gy_m * y_grid

                # total off-resonance field (includes B0 + gradient + user offset)
                total_offset = (
                    B0
                    + offset_x
                    + offset_y
                    + offset_t.view(B, 1, 1) * 2 * self.pi / self.gamma
                )

                # Effective field and rotation angles
                B1_eff = torch.hypot(b1_mag, total_offset)
                alpha = self.safe_atan2(total_offset, b1_mag)
                theta = gamma_dt * B1_eff

                # Rotations (same order as your original)
                Mx_blk, My_blk = self.rotate_z(Mx_blk, My_blk, phi_m)
                Mx_blk, Mz_blk = self.rotate_y(Mx_blk, Mz_blk, alpha)
                # Mx_blk, My_blk = self.rotate_z(Mx_blk, My_blk, theta)
                My_blk, Mz_blk = self.rotate_x(My_blk, Mz_blk, theta)
                Mx_blk, Mz_blk = self.rotate_y(Mx_blk, Mz_blk, -alpha)
                Mx_blk, My_blk = self.rotate_z(Mx_blk, My_blk, -phi_m)

            return Mx_blk, My_blk, Mz_blk

        # Whether we actually checkpoint (only meaningful when grads are needed)
        need_grad = (
            re.requires_grad
            or im.requires_grad
            or grad_x.requires_grad
            or grad_y.requires_grad
        )
        do_ckpt = (
            self.use_checkpoint and self.training and need_grad and self.block_steps > 0
        )

        # Run in blocks
        bs = self.block_steps if self.block_steps > 0 else T
        for start in range(0, T, bs):
            end = min(start + bs, T)

            if do_ckpt:

                def _blk_fn(
                    Mx_t, My_t, Mz_t, re_t, im_t, gx_t, gy_t, start_i=start, end_i=end
                ):
                    return _run_block(
                        Mx_t, My_t, Mz_t, re_t, im_t, gx_t, gy_t, start_i, end_i
                    )

                Mx, My, Mz = checkpoint(
                    _blk_fn, Mx, My, Mz, re, im, grad_x, grad_y, use_reentrant=True
                )

            else:
                # 🔑 SAME math, SAME signature, just without checkpoint
                Mx, My, Mz = _run_block(Mx, My, Mz, re, im, grad_x, grad_y, start, end)

        Mxy_abs = torch.hypot(Mx, My).unsqueeze(1)  # (B,1,Ny,Nx)
        Mxy_phi = self.safe_atan2(My, Mx).unsqueeze(1)
        return Mxy_abs, Mxy_phi
