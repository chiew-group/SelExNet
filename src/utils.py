#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-03-03 19:57:27
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-04-10 18:06:34
FilePath     : /Documents/paper1/sTx/sTx_RF_SI/src/utils.py
Description  :
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import shutil
import logging
from typing import Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import pi
import math
from torch_pca import PCA
import warnings

FILTER = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d,
}

plt.switch_backend("agg")

__all__ = [
    "resume_training",
    "plot_progress",
    "setup_logger",
    "make_if_dont_exist",
    "GaussianMixture",
    "FMGenerator",
    "SSIM",
]


def resume_training(
    cfg,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    resume: bool,
    is_main_process: bool,
):
    start_epoch = 0
    best_val_loss = float("inf")
    train_losses, valid_losses = [], []
    resume_update = resume

    ckpt_path = os.path.join("outputs", cfg.exp_name, "checkpoint", "ckpt.pt")

    if resume:
        if os.path.exists(ckpt_path):
            if is_main_process:
                print(f"Resuming from {ckpt_path}")

            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_loss = float(checkpoint["best_val_loss"])
            train_losses = checkpoint["train_losses"]
            valid_losses = checkpoint["val_losses"]
        else:
            if is_main_process:
                print(f"No checkpoint found at {ckpt_path}, training from scratch.")
            resume_update = False

    return (
        model,
        optimizer,
        scheduler,
        resume_update,
        start_epoch,
        best_val_loss,
        train_losses,
        valid_losses,
    )


def plot_progress(
    logger: logging.Logger,
    save_dir: str,
    train_loss: Sequence[Sequence[Union[int, float]]],
    val_loss: Sequence[Sequence[Union[int, float]]],
    name: str,
) -> None:
    """
    Should probably by improved
    :return:
    """
    assert len(train_loss) != 0
    train_loss = np.array(train_loss)
    try:
        font = {"weight": "normal", "size": 18}

        matplotlib.rc("font", **font)

        fig = plt.figure(figsize=(30, 24))
        ax = fig.add_subplot(111)
        ax.plot(train_loss[:, 0], train_loss[:, 1], color="b", ls="-", label="loss_tr")
        if len(val_loss) != 0:
            val_loss = np.array(val_loss)
            ax.plot(val_loss[:, 0], val_loss[:, 1], color="r", ls="-", label="loss_val")

        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.set_title(name)
        fig.savefig(os.path.join(save_dir, name + ".png"))
        plt.cla()
        plt.close(fig)
    except Exception as e:
        logger.info(f"failed to plot {name} training progress: {e}")


def setup_logger(
    logger_name: str, log_file: str, level: int = logging.INFO
) -> logging.Logger:
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    log_setup.setLevel(level)
    log_setup.propagate = False
    if not log_setup.handlers:
        fileHandler = logging.FileHandler(log_file, mode="w")
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        log_setup.addHandler(fileHandler)
        log_setup.addHandler(streamHandler)

    return log_setup


def make_if_dont_exist(folder_path: str, overwrite: bool = False):
    if os.path.exists(folder_path):
        if not overwrite:
            print(f"{folder_path} exists, no overwrite here.")
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path, ignore_errors=True)
            os.makedirs(folder_path)
    else:
        os.makedirs(folder_path)
        print(f"{folder_path} created!")


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Memory-friendly batch quadratic form helper.
    mat_a: (n, k, 1, d)
    mat_b: (1, k, d, d)
    return: (n, k, 1, d)
    """
    res = torch.zeros(mat_a.shape, device=mat_a.device, dtype=mat_a.dtype)
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)  # (n, d)
        mat_b_i = mat_b[0, i, :, :].squeeze()  # (d, d)
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    return res


def calculate_matmul(mat_a, mat_b):
    """
    mat_a: (n, k, 1, d)
    mat_b: (n, k, d, 1)
    return: (n, k, 1, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


# -----------------------------
# Gaussian Mixture Model (fixed)
# -----------------------------
class GaussianMixture(torch.nn.Module):
    """
    EM-fit Gaussian Mixture Model in PyTorch.

    Supports covariance_type in {"full", "diag"}.

    IMPORTANT FIXES vs your version:
    1) diag log-likelihood was wrong (used 1/sqrt(var) instead of 1/var; wrong logdet)
    2) safe param updates (never replace nn.Parameter; use .data.copy_)
    3) rollback uses clones, not references
    4) variance flooring and stable logdet / quadratic forms
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        covariance_type: str = "full",
        eps: float = 1.0e-6,
        init_params: str = "kmeans",
        mu_init: torch.Tensor | None = None,
        var_init: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_components = int(n_components)
        self.n_features = int(n_features)
        self.covariance_type = str(covariance_type)
        self.eps = float(eps)
        self.init_params = str(init_params)

        assert self.covariance_type in ["full", "diag"]
        assert self.init_params in ["kmeans", "random"]

        self.mu_init = mu_init
        self.var_init = var_init

        self.log_likelihood = torch.tensor(float("-inf"))
        self.params_fitted = False

        self._init_params()

    def _init_params(self):
        device = self.mu_init.device if self.mu_init is not None else None
        dtype = self.mu_init.dtype if self.mu_init is not None else None

        if self.mu_init is not None:
            assert self.mu_init.size() == (1, self.n_components, self.n_features)
            self.mu = torch.nn.Parameter(
                self.mu_init.detach().clone(), requires_grad=False
            )
        else:
            self.mu = torch.nn.Parameter(
                torch.randn(
                    1, self.n_components, self.n_features, device=device, dtype=dtype
                ),
                requires_grad=False,
            )

        if self.covariance_type == "diag":
            if self.var_init is not None:
                assert self.var_init.size() == (1, self.n_components, self.n_features)
                var0 = self.var_init.detach().clone()
            else:
                var0 = torch.ones(
                    1, self.n_components, self.n_features, device=device, dtype=dtype
                )
            self.var = torch.nn.Parameter(var0, requires_grad=False)
        else:
            if self.var_init is not None:
                assert self.var_init.size() == (
                    1,
                    self.n_components,
                    self.n_features,
                    self.n_features,
                )
                var0 = self.var_init.detach().clone()
            else:
                eye = torch.eye(self.n_features, device=device, dtype=dtype).view(
                    1, 1, self.n_features, self.n_features
                )
                var0 = eye.repeat(1, self.n_components, 1, 1)
            self.var = torch.nn.Parameter(var0, requires_grad=False)

        self.pi = torch.nn.Parameter(
            torch.full(
                (1, self.n_components, 1),
                1.0 / self.n_components,
                device=device,
                dtype=dtype,
            ),
            requires_grad=False,
        )

        self.params_fitted = False

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
        # Accept (n,d) or (n,1,d)
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x

    def fit(
        self,
        x: torch.Tensor,
        delta: float = 1e-3,
        n_iter: int = 100,
        warm_start: bool = False,
    ):
        """
        EM optimization.

        x: (n, d) or (n, 1, d)
        """
        x = self._ensure_3d(x)

        if (not warm_start) and self.params_fitted:
            self._init_params()

        # init by kmeans if requested
        if self.init_params == "kmeans" and self.mu_init is None:
            self.mu.data.copy_(self.get_kmeans_mu(x, n_centers=self.n_components))

        # EM loop
        ll_old = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)

        for _ in range(int(n_iter)):
            # Save old params for rollback (CLONE!)
            mu_old = self.mu.data.clone()
            var_old = self.var.data.clone()
            pi_old = self.pi.data.clone()

            self.__em(x)
            ll_new = self.__score(x, as_average=True)

            if torch.isnan(ll_new) or torch.isinf(ll_new):
                # re-init if blow up
                self._init_params()
                if self.init_params == "kmeans":
                    self.mu.data.copy_(
                        self.get_kmeans_mu(x, n_centers=self.n_components)
                    )
                continue

            # Convergence check
            if (ll_new - ll_old).abs() < delta:
                self.params_fitted = True
                self.log_likelihood = ll_new.detach()
                return

            # If likelihood decreases, rollback (use saved clones)
            if ll_new < ll_old:
                self.mu.data.copy_(mu_old)
                self.var.data.copy_(var_old)
                self.pi.data.copy_(pi_old)
                self.params_fitted = True
                self.log_likelihood = ll_old.detach()
                return

            ll_old = ll_new
            self.log_likelihood = ll_new.detach()

        self.params_fitted = True

    def sample(self, n: int = 1):
        """
        Sample from the fitted model.
        returns:
            x: (n, d)
            y: (n,)
        """
        n = int(n)
        probs = self.pi.squeeze(-1).squeeze(0)  # (k,)
        counts = (
            torch.distributions.multinomial.Multinomial(total_count=n, probs=probs)
            .sample()
            .to(torch.int64)
        )

        x_all = []
        y_all = []

        for k in range(self.n_components):
            ck = int(counts[k].item())
            if ck <= 0:
                continue

            if self.covariance_type == "diag":
                var_k = torch.clamp(self.var[0, k], min=self.eps)
                x_k = self.mu[0, k] + torch.randn(
                    ck, self.n_features, device=self.mu.device, dtype=self.mu.dtype
                ) * torch.sqrt(var_k)
            else:
                cov_k = self.var[0, k]
                cov_k = cov_k + self.eps * torch.eye(
                    self.n_features, device=cov_k.device, dtype=cov_k.dtype
                )
                dist = torch.distributions.multivariate_normal.MultivariateNormal(
                    self.mu[0, k], cov_k
                )
                x_k = dist.sample((ck,))

            x_all.append(x_k)
            y_all.append(torch.full((ck,), k, device=self.mu.device, dtype=torch.int64))

        x = (
            torch.cat(x_all, dim=0)
            if len(x_all)
            else torch.empty(
                (0, self.n_features), device=self.mu.device, dtype=self.mu.dtype
            )
        )
        y = (
            torch.cat(y_all, dim=0)
            if len(y_all)
            else torch.empty((0,), device=self.mu.device, dtype=torch.int64)
        )
        return x, y

    def score_samples(self, x: torch.Tensor):
        x = self._ensure_3d(x)
        return self.__score(x, as_average=False)

    # -----------------------------
    # Core likelihood (fixed)
    # -----------------------------
    def _estimate_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log N(x|mu,var) per component.
        Output: (n, k, 1)
        """
        x = self._ensure_3d(x)

        if self.covariance_type == "diag":
            # Correct diag Gaussian:
            # -0.5*( d log(2pi) + sum log(var) + sum (x-mu)^2/var )
            mu = self.mu  # (1,k,d)
            var = torch.clamp(self.var, min=self.eps)  # (1,k,d)
            inv_var = 1.0 / var

            log_det = torch.sum(torch.log(var), dim=2, keepdim=True)  # (1,k,1)
            quad = torch.sum((x - mu) ** 2 * inv_var, dim=2, keepdim=True)  # (n,k,1)

            return -0.5 * (self.n_features * np.log(2.0 * pi) + log_det + quad)

        # full covariance
        mu = self.mu
        cov = self.var
        eye = torch.eye(self.n_features, device=cov.device, dtype=cov.dtype).view(
            1, 1, self.n_features, self.n_features
        )
        cov = cov + self.eps * eye

        n = x.shape[0]
        out = torch.empty((n, self.n_components, 1), device=x.device, dtype=x.dtype)

        log_2pi = self.n_features * np.log(2.0 * pi)

        # Component-wise cholesky solve (stable, avoids explicit inverse)
        for k in range(self.n_components):
            L = torch.linalg.cholesky(cov[0, k])  # (d,d)
            diff = x[:, 0, :] - mu[0, k, :]  # (n,d)
            y = torch.linalg.solve_triangular(L, diff.T, upper=False).T  # (n,d)
            quad = torch.sum(y * y, dim=1, keepdim=True)  # (n,1)
            log_det = 2.0 * torch.log(torch.diagonal(L)).sum()  # scalar
            out[:, k, :] = -0.5 * (log_2pi + log_det + quad)

        return out

    def _e_step(self, x: torch.Tensor):
        x = self._ensure_3d(x)
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(
            torch.clamp(self.pi, min=self.eps)
        )
        log_prob_norm = torch.logsumexp(
            weighted_log_prob, dim=1, keepdim=True
        )  # (n,1,1)
        log_resp = weighted_log_prob - log_prob_norm
        return torch.mean(log_prob_norm), log_resp

    def _m_step(self, x: torch.Tensor, log_resp: torch.Tensor):
        x = self._ensure_3d(x)
        resp = torch.exp(log_resp)  # (n,k,1)

        Nk = torch.sum(resp, dim=0, keepdim=True) + self.eps  # (1,k,1)
        pi = Nk / x.shape[0]  # (1,k,1)

        mu = torch.sum(resp * x, dim=0, keepdim=True) / Nk  # (1,k,d)

        if self.covariance_type == "diag":
            # var = E[(x-mu)^2]
            x2 = torch.sum(resp * x * x, dim=0, keepdim=True) / Nk
            mu2 = mu * mu
            xmu = torch.sum(resp * x * mu, dim=0, keepdim=True) / Nk
            var = x2 - 2 * xmu + mu2
            var = torch.clamp(var, min=self.eps)
            return pi, mu, var

        # full covariance
        n = x.shape[0]
        var = torch.empty(
            (1, self.n_components, self.n_features, self.n_features),
            device=x.device,
            dtype=x.dtype,
        )
        eye = torch.eye(self.n_features, device=x.device, dtype=x.dtype)

        for k in range(self.n_components):
            diff = x[:, 0, :] - mu[0, k, :]  # (n,d)
            # weighted covariance: sum_n resp_nk * diff_n diff_n^T / Nk
            w = resp[:, k, 0].view(n, 1)  # (n,1)
            cov_k = (diff * w).T @ diff / Nk[0, k, 0]
            cov_k = cov_k + self.eps * eye
            var[0, k] = cov_k

        return pi, mu, var

    def __em(self, x: torch.Tensor):
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)
        self.pi.data.copy_(pi)
        self.mu.data.copy_(mu)
        self.var.data.copy_(var)

    def __score(self, x: torch.Tensor, as_average: bool = True):
        x = self._ensure_3d(x)
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(
            torch.clamp(self.pi, min=self.eps)
        )
        per_sample = torch.logsumexp(weighted_log_prob, dim=1)  # (n,1)
        if as_average:
            return per_sample.mean()
        return per_sample.squeeze(-1).squeeze(-1)

    def get_kmeans_mu(
        self,
        x: torch.Tensor,
        n_centers: int,
        init_times: int = 20,
        min_delta: float = 1e-3,
    ):
        """
        Lightweight kmeans init (same idea as your code) but returns (1,k,d)
        """
        x = self._ensure_3d(x).squeeze(1)  # (n,d)
        x_min, x_max = x.min(), x.max()
        denom = torch.clamp(x_max - x_min, min=self.eps)
        xn = (x - x_min) / denom

        best_cost = float("inf")
        best_center = None

        n = xn.shape[0]
        for _ in range(int(init_times)):
            idx = torch.randperm(n, device=x.device)[:n_centers]
            center = xn[idx].clone()  # (k,d)

            for _it in range(50):
                dists = torch.cdist(xn, center)  # (n,k)
                labels = torch.argmin(dists, dim=1)  # (n,)
                center_old = center.clone()

                for c in range(n_centers):
                    mask = labels == c
                    if mask.any():
                        center[c] = xn[mask].mean(dim=0)

                delta = torch.norm(center - center_old, dim=1).max()
                if delta < min_delta:
                    break

            dists = torch.cdist(xn, center)
            labels = torch.argmin(dists, dim=1)

            cost = 0.0
            for c in range(n_centers):
                mask = labels == c
                if mask.any():
                    cost += torch.norm(xn[mask] - center[c], p=2, dim=1).mean().item()

            if cost < best_cost:
                best_cost = cost
                best_center = center

        # de-normalize back to original scale
        mu = best_center * denom + x_min
        return mu.unsqueeze(0)  # (1,k,d)


# ---------------------------------
# PCA + GMM wrapper for B0/B1 maps
# ---------------------------------
class FMGenerator:
    """
    Loads flattened B0/B1 maps from a .npz, standardizes them,
    fits PCA and GMM in PCA score space, then samples new maps.

    Notes:
    - Sampling only: wrap sampling in torch.no_grad() in your training loop.
    - Tesla-scale stable due to standardization + (optional) float64 PCA fit.

    Expected arrays in npz:
        b0_{mode}:      (N, Nx, Ny)
        b1_real_{mode}: (N, C, Nx, Ny)
        b1_imag_{mode}: (N, C, Nx, Ny)
    """

    def __init__(
        self,
        cfg,
        mode: str = "train",
        desired_variance: float = 0.95,
        standardize_eps: float = 1e-8,
        gmm_eps: float = 1e-6,
    ):
        self.cfg = cfg
        self.mode = mode
        self.desired_variance = np.float64(desired_variance)
        self.internal_dtype = torch.float64
        self.output_dtype = torch.float32
        self.standardize_eps = np.float64(standardize_eps)
        self.gmm_eps = np.float64(gmm_eps)

        # load & standardize
        self._load_data()

        # choose components
        self._find_num_component()

        # fit PCA+GMM
        self._init_pca_gmm()

    # -----------------------------
    # Standardization helpers
    # -----------------------------
    def _standardize(self, x: torch.Tensor):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False)
        std = torch.clamp(std, min=self.standardize_eps)
        return (x - mean) / std, mean, std

    @staticmethod
    def _destandardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return x * std + mean

    # -----------------------------
    # Load data
    # -----------------------------
    def _load_data(self):
        numpy_data = np.load(self.cfg.magnet.si_path)

        b0 = numpy_data[f"b0_{self.mode}"]  # (N, Nx, Ny)
        b1r = numpy_data[f"b1_real_{self.mode}"]  # (N, C, Nx, Ny)
        b1i = numpy_data[f"b1_imag_{self.mode}"]  # (N, C, Nx, Ny)

        b0 = (
            torch.tensor(b0, dtype=self.internal_dtype)
            .contiguous()
            .view(b0.shape[0], -1)
        )
        b1r = (
            torch.tensor(b1r, dtype=self.internal_dtype)
            .contiguous()
            .view(b1r.shape[0], -1)
        )
        b1i = (
            torch.tensor(b1i, dtype=self.internal_dtype)
            .contiguous()
            .view(b1i.shape[0], -1)
        )

        # Standardize for Tesla stability
        self.b0, self.b0_mean, self.b0_std = self._standardize(b0)
        self.b1_real, self.b1r_mean, self.b1r_std = self._standardize(b1r)
        self.b1_imag, self.b1i_mean, self.b1i_std = self._standardize(b1i)

    # -----------------------------
    # Determine PCA components
    # -----------------------------
    def _find_num_component(self):
        # Fit PCA with n_components=None to get full spectrum
        pca_b0 = PCA(n_components=None, svd_solver="full")
        pca_b1r = PCA(n_components=None, svd_solver="full")
        pca_b1i = PCA(n_components=None, svd_solver="full")

        # Fit in float64 for Tesla-scale stability
        pca_b0_fit = pca_b0.fit(self.b0)
        pca_b1r_fit = pca_b1r.fit(self.b1_real)
        pca_b1i_fit = pca_b1i.fit(self.b1_imag)

        c0 = torch.cumsum(pca_b0_fit.explained_variance_ratio_, dim=0)
        c1 = torch.cumsum(pca_b1r_fit.explained_variance_ratio_, dim=0)
        c2 = torch.cumsum(pca_b1i_fit.explained_variance_ratio_, dim=0)

        dv = self.desired_variance
        self.b0_num_components = int(
            torch.nonzero(c0 >= dv, as_tuple=False)[0].item() + 1
        )
        self.b1_real_num_components = int(
            torch.nonzero(c1 >= dv, as_tuple=False)[0].item() + 1
        )
        self.b1_imag_num_components = int(
            torch.nonzero(c2 >= dv, as_tuple=False)[0].item() + 1
        )

    # -----------------------------
    # Fit PCA + GMM
    # -----------------------------
    def _init_pca_gmm(self):
        # PCA models
        b0_pca = PCA(n_components=self.b0_num_components, svd_solver="full")
        b1r_pca = PCA(n_components=self.b1_real_num_components, svd_solver="full")
        b1i_pca = PCA(n_components=self.b1_imag_num_components, svd_solver="full")

        # Fit/transform in float64, store PCA objects
        b0_scores = b0_pca.fit_transform(self.b0)
        b1r_scores = b1r_pca.fit_transform(self.b1_real)
        b1i_scores = b1i_pca.fit_transform(self.b1_imag)

        # GMMs in score space
        b0_gmm = GaussianMixture(
            n_components=2,
            n_features=self.b0_num_components,
            covariance_type="diag",
            eps=self.gmm_eps,
            init_params="kmeans",
        )
        b1r_gmm = GaussianMixture(
            n_components=3,
            n_features=self.b1_real_num_components,
            covariance_type="diag",
            eps=self.gmm_eps,
            init_params="kmeans",
        )
        b1i_gmm = GaussianMixture(
            n_components=3,
            n_features=self.b1_imag_num_components,
            covariance_type="diag",
            eps=self.gmm_eps,
            init_params="kmeans",
        )

        # Fit (no gradients needed)
        with torch.no_grad():
            b0_gmm.fit(b0_scores)
            b1r_gmm.fit(b1r_scores)
            b1i_gmm.fit(b1i_scores)

        self.pca_group = {"B0": b0_pca, "B1_real": b1r_pca, "B1_imag": b1i_pca}
        self.gmm_group = {"B0": b0_gmm, "B1_real": b1r_gmm, "B1_imag": b1i_gmm}

    # -----------------------------
    # Sampling new maps
    # -----------------------------
    def _sample_new_map(self, batch_size: int, device: torch.device | None = None):
        """
        Returns:
            B0_maps:      (B, Nx, Ny)
            B1_real_maps: (B, C, Nx, Ny)
            B1_imag_maps: (B, C, Nx, Ny)
        """
        B = int(batch_size)
        Nx, Ny = self.cfg.image.N
        tx = self.cfg.magnet.tx

        device = device if device is not None else torch.device("cpu")

        B0_maps = torch.zeros(B, Nx, Ny, dtype=self.internal_dtype, device=device)
        B1_real_maps = torch.zeros(
            B, tx, Nx, Ny, dtype=self.internal_dtype, device=device
        )
        B1_imag_maps = torch.zeros(
            B, tx, Nx, Ny, dtype=self.internal_dtype, device=device
        )

        # Ensure models are on the sampling device
        # (PCA objects are CPU-ish; GMM parameters must be on device)
        for key in self.gmm_group:
            self.gmm_group[key].to(device)

        with torch.no_grad():
            for i in range(B):
                for key in ["B0", "B1_real", "B1_imag"]:
                    pca = self.pca_group[key]
                    gmm = self.gmm_group[key]

                    z, _ = gmm.sample(n=1)  # (1, D) on device
                    # PCA inverse_transform may run on CPU depending on torch_pca implementation.
                    # To be robust, do PCA inversion on CPU, then bring back.
                    pca_dtype = pca.components_.dtype  # ALWAYS correct
                    z_cpu = z.detach().to("cpu", dtype=pca_dtype)
                    x_cpu = pca.inverse_transform(z_cpu)
                    x = x_cpu.to(device=device, dtype=self.internal_dtype)

                    # De-standardize back to original physical unit space
                    if key == "B0":
                        x = self._destandardize(
                            x, self.b0_mean.to(device), self.b0_std.to(device)
                        )
                        B0_maps[i] = x.reshape(Nx, Ny)
                    elif key == "B1_real":
                        x = self._destandardize(
                            x, self.b1r_mean.to(device), self.b1r_std.to(device)
                        )
                        B1_real_maps[i] = x.reshape(tx, Nx, Ny)
                    else:
                        x = self._destandardize(
                            x, self.b1i_mean.to(device), self.b1i_std.to(device)
                        )
                        B1_imag_maps[i] = x.reshape(tx, Nx, Ny)

        B0_maps = B0_maps.to(dtype=self.output_dtype)
        B1_real_maps = B1_real_maps.to(dtype=self.output_dtype)
        B1_imag_maps = B1_imag_maps.to(dtype=self.output_dtype)

        return B0_maps, B1_real_maps, B1_imag_maps


class GaussianFilter(nn.Module):
    def __init__(
        self,
        data_dim,
        window_size,
        in_channels,
        sigma,
        padding=None,
        ensemble_kernel=True,
    ):
        """Gaussian Filer for 1D, 2D or 3D data (3D/4D/5D tensor)

        Args:
            data_dim (int, optional): The dimension of the data.
            window_size (int or Tuple[int], optional): The window size of the gaussian filter.
            in_channels (int, optional): The number of channels of the 4d tensor.
            sigma (float or Tuple[float], optional): The sigma of the gaussian filter.
            padding (int or Tuple[int], optional): The padding of the gaussian filter. Defaults to None. If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
            ensemble_kernel (bool, optional): Whether to fuse the two cascaded 1d kernel into a 2d kernel. Defaults to True.
        """
        super().__init__()
        if data_dim not in [1, 2, 3]:
            raise ValueError(f"data_dim must be 1, 2 or 3, but got {data_dim}.")
        self.data_dim = data_dim
        self.filter = FILTER[self.data_dim]

        if isinstance(window_size, int):
            window_size = [window_size] * self.data_dim
        if not all([w % 2 == 1 for w in window_size]):
            raise ValueError(f"Window size must be odd, but got {window_size}.")
        self.window_size = window_size

        if padding is None:
            padding = [w // 2 for w in window_size]
        if isinstance(padding, int):
            padding = [padding] * self.data_dim
        self.padding = padding

        if isinstance(sigma, (float, int)):
            sigma = [sigma] * self.data_dim
        self.sigma2 = [s**2 for s in sigma]

        assert (
            len(self.window_size)
            == len(self.padding)
            == len(self.sigma2)
            == self.data_dim
        )
        kernels = [
            self._get_gaussian_window1d(w, s2)
            for w, s2 in zip(self.window_size, self.sigma2)
        ]

        self.ensemble_kernel = ensemble_kernel
        if self.ensemble_kernel:
            kernels = self._get_gaussian_windowNd(kernels)
            kernels = kernels.reshape(1, 1, *self.window_size).repeat_interleave(
                repeats=in_channels, dim=0
            )
            self.register_buffer(name="gaussian_window", tensor=kernels)
        else:
            for dim_idx, kernel in enumerate(kernels, start=2):
                base_shape = [1, 1] + [1] * self.data_dim
                base_shape[dim_idx] = -1
                kernel = kernel.reshape(*base_shape).repeat_interleave(
                    repeats=in_channels, dim=0
                )
                if dim_idx == 2:
                    name = "gaussian_window"
                else:
                    name = f"gaussian_window_{dim_idx}"
                self.register_buffer(name=name, tensor=kernel)

    @staticmethod
    def _get_gaussian_window1d(window_size, sigma2):
        x = torch.arange(-(window_size // 2), window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        w = w / w.sum()
        return w

    def _get_gaussian_windowNd(self, gaussian_windows_1d):
        for dim_idx, kernel in enumerate(gaussian_windows_1d, start=2):
            base_shape = [1, 1] + [1] * self.data_dim
            base_shape[dim_idx] = -1
            kernel = kernel.reshape(*base_shape)
            if dim_idx == 2:
                w = kernel
            else:
                w = w * kernel
        return w

    def __repr__(self):
        base_str = (
            f"{self.__class__.__name__} with Kernel: {self.gaussian_window.shape}"
        )
        if not self.ensemble_kernel:
            for dim_idx in range(3, self.data_dim + 2):
                kernel = self.get_buffer(f"gaussian_window_{dim_idx}")
                base_str += f", {kernel.shape}"
        return base_str

    def forward(self, x):
        if self.ensemble_kernel:
            # ensemble kernel: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/3add4532d3f633316cba235da1c69e90f0dfb952/pytorch_ssim/__init__.py#L11-L15
            x = self.filter(
                input=x,
                weight=self.gaussian_window,
                stride=1,
                padding=self.padding,
                groups=x.shape[1],
            )
        else:
            # splitted kernel: https://github.com/VainF/pytorch-msssim/blob/2398f4db0abf44bcd3301cfadc1bf6c94788d416/pytorch_msssim/ssim.py#L48
            for i, d in enumerate(x.shape[2:], start=2):
                if d >= self.window_size[i - 2]:
                    w = self.get_buffer(
                        target="gaussian_window" if i == 2 else f"gaussian_window_{i}"
                    )
                    x = self.filter(
                        input=x,
                        weight=w,
                        stride=1,
                        padding=self.padding,
                        groups=x.shape[1],
                    )
                else:
                    warnings.warn(
                        f"Skipping Gaussian Smoothing at dimension {i} for x: {x.shape} and window size: {self.window_size}"
                    )
        return x


class SSIM(nn.Module):
    def __init__(
        self,
        window_size=11,
        in_channels=1,
        sigma=1.5,
        *,
        K1=0.01,
        K2=0.03,
        L=1,
        keep_batch_dim=False,
        data_dim=2,
        return_log=False,
        return_msssim=False,
        padding=None,
        ensemble_kernel=True,
    ):
        """Calculate the mean SSIM (MSSIM) between two 4D tensors.

        Args:
            window_size (int or Tuple[int], optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float or Tuple[float], optional): The sigma of the gaussian filter. Defaults to 1.5.
            K1 (float, optional): K1 of MSSIM. Defaults to 0.01.
            K2 (float, optional): K2 of MSSIM. Defaults to 0.03.
            L (int, optional): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
            keep_batch_dim (bool, optional): Whether to keep the batch dim. Defaults to False.
            data_dim (int, optional): The dimension of the data. Defaults to 2, which means a 2d image (4d tensor).
            return_log (bool, optional): Whether to return the logarithmic form. Defaults to False.
            return_msssim (bool, optional): Whether to return the MS-SSIM score. Defaults to False, which will return the original MSSIM score.
            padding (int or Tuple[int], optional): The padding of the gaussian filter. Defaults to None. If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
            ensemble_kernel (bool, optional): Whether to fuse the two cascaded 1d kernel into a 2d kernel. Defaults to True.

        ```
            # setting 0: for 4d float tensors with the data range [0, 1] and 1 channel
            ssim_caller = SSIM().cuda()
            # setting 1: for 4d float tensors with the data range [0, 1] and 3 channel
            ssim_caller = SSIM(in_channels=3).cuda()
            # setting 2: for 4d float tensors with the data range [0, 255] and 3 channel
            ssim_caller = SSIM(L=255, in_channels=3).cuda()
            # setting 3: for 4d float tensors with the data range [0, 255] and 3 channel, and return the logarithmic form
            ssim_caller = SSIM(L=255, in_channels=3, return_log=True).cuda()
            # setting 4: for 4d float tensors with the data range [0, 1] and 1 channel,return the logarithmic form, and keep the batch dim
            ssim_caller = SSIM(return_log=True, keep_batch_dim=True).cuda()
            # setting 5: for 4d float tensors with the data range [0, 1] and 1 channel, padding=0 and the splitted kernels.
            ssim_caller = SSIM(return_log=True, keep_batch_dim=True, padding=0, ensemble_kernel=False).cuda()

            # two 4d tensors
            x = torch.randn(3, 1, 100, 100).cuda()
            y = torch.randn(3, 1, 100, 100).cuda()
            ssim_score_0 = ssim_caller(x, y)
            # or in the fp16 mode (we have fixed the computation progress into the float32 mode to avoid the unexpected result)
            with torch.amp.autocast(enabled=True):
                ssim_score_1 = ssim_caller(x, y)
            assert torch.isclose(ssim_score_0, ssim_score_1)
        ```

        Reference:
        [1] SSIM: Wang, Zhou et al. “Image quality assessment: from error visibility to structural similarity.” IEEE Transactions on Image Processing 13 (2004): 600-612.
        [2] MS-SSIM: Wang, Zhou et al. “Multi-scale structural similarity for image quality assessment.” (2003).
        """
        super().__init__()
        self.data_dim = data_dim
        self.window_size = window_size
        self.C1 = (K1 * L) ** 2  # equ 7 in ref1
        self.C2 = (K2 * L) ** 2  # equ 7 in ref1
        self.keep_batch_dim = keep_batch_dim
        self.return_log = return_log
        self.return_msssim = return_msssim
        if self.return_msssim and self.return_log:
            raise ValueError("return_log only support return_msssim=False")
        if self.return_msssim and self.data_dim < 2:
            raise ValueError("return_msssim only support data_dim>=2")

        self.gaussian_filter = GaussianFilter(
            data_dim=self.data_dim,
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            padding=padding,
            ensemble_kernel=ensemble_kernel,
        )

    def forward(self, x, y):
        """Calculate the mean SSIM (MSSIM) between two 3d/4d/5d tensors.

        Args:
            x (Tensor): 3d/4d/5d tensor
            y (Tensor): 3d/4d/5d tensor

        Returns:
            Tensor: MSSIM or MS-SSIM
        """
        assert x.shape == y.shape, f"x: {x.shape} and y: {y.shape} must be the same"
        assert x.ndim == self.data_dim + 2, (
            f"x: {x.ndim} and y: {y.ndim} must be {self.data_dim + 2}d tensors"
        )
        if x.type() != self.gaussian_filter.gaussian_window.type():
            x = x.type_as(self.gaussian_filter.gaussian_window)
        if y.type() != self.gaussian_filter.gaussian_window.type():
            y = y.type_as(self.gaussian_filter.gaussian_window)

        if self.return_msssim:
            return self.msssim(x, y)
        else:
            return self.ssim(x, y)

    def ssim(self, x, y):
        ssim, _ = self._ssim(x, y)
        if self.return_log:
            # https://github.com/xuebinqin/BASNet/blob/56393818e239fed5a81d06d2a1abfe02af33e461/pytorch_ssim/__init__.py#L81-L83
            ssim = ssim - ssim.min()
            ssim = ssim / ssim.max()
            ssim = -torch.log(ssim + 1e-8)

        if self.keep_batch_dim:
            return ssim.flatten(1).mean(-1)
        else:
            return ssim.mean()

    def msssim(self, x, y):
        ms_components = []
        for i, w in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
            ssim, cs = self._ssim(x, y)

            if self.keep_batch_dim:
                ssim = ssim.flatten(1).mean(-1)
                cs = cs.flatten(1).mean(-1)
            else:
                ssim = ssim.mean()
                cs = cs.mean()

            if i == 4:
                ms_components.append(ssim**w)
            else:
                ms_components.append(cs**w)
                bs, *c, h, w = x.shape
                padding = [s % 2 for s in (h, w)]  # spatial padding
                if len(c) > 1:
                    # only pooling in the spatial domain
                    x = x.reshape(bs, -1, h, w)
                    y = y.reshape(bs, -1, h, w)
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)
                if len(c) > 1:
                    x = x.reshape(bs, *c, h // 2, w // 2)
                    y = y.reshape(bs, *c, h // 2, w // 2)
        msssim = math.prod(ms_components)  # equ 7 in ref2
        return msssim

    def _ssim(self, x, y):
        mu_x = self.gaussian_filter(x)  # equ 14
        mu_y = self.gaussian_filter(y)  # equ 14
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x  # equ 15
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y  # equ 15
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y  # equ 16

        A1 = 2 * mu_x * mu_y + self.C1
        A2 = 2 * sigma_xy + self.C2
        B1 = mu_x * mu_x + mu_y * mu_y + self.C1
        B2 = sigma2_x + sigma2_y + self.C2

        # equ 12, 13 in ref1
        l = A1 / B1
        cs = A2 / B2
        ssim = l * cs
        return ssim, cs


# def calculate_matmul_n_times(n_components, mat_a, mat_b):
#     """
#     Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
#     Bypasses torch.matmul to reduce memory footprint.
#     args:
#         mat_a:      torch.Tensor (n, k, 1, d)
#         mat_b:      torch.Tensor (1, k, d, d)
#     """
#     res = torch.zeros(mat_a.shape).to(mat_a.device)

#     for i in range(n_components):
#         mat_a_i = mat_a[:, i, :, :].squeeze(-2)
#         mat_b_i = mat_b[0, i, :, :].squeeze()
#         res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

#     return res


# def calculate_matmul(mat_a, mat_b):
#     """
#     Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
#     Bypasses torch.matmul to reduce memory footprint.
#     args:
#         mat_a:      torch.Tensor (n, k, 1, d)
#         mat_b:      torch.Tensor (n, k, d, 1)
#     """
#     assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
#     return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


# class BMapGenerator(object):
#     def __init__(self, cfg, mode="train"):
#         self.cfg = cfg
#         self.dtype = torch.float32
#         self.mode = mode
#         self.load_data()
#         self.find_num_component()
#         self.init_pca_gmm()

#     def load_data(self):
#         numpy_data = np.load(self.cfg.magnet.bmap_path)
#         b0 = numpy_data[f"b0_{self.mode}"]  # shape (N, Nx, Ny)
#         b1_real = numpy_data[f"b1_real_{self.mode}"]  # shape (N, C, Nx, Ny)
#         b1_imag = numpy_data[f"b1_imag_{self.mode}"]  # shape (N, C, Nx, Ny)

#         N = b0.shape[0]
#         N1 = b1_real.shape[0]

#         b0_tensor = torch.tensor(b0, dtype=self.dtype)
#         b1_real_tensor = torch.tensor(b1_real, dtype=self.dtype)
#         b1_imag_tensor = torch.tensor(b1_imag, dtype=self.dtype)
#         b0_tensor = b0_tensor.contiguous().view(N, -1)
#         b1_real_tensor = b1_real_tensor.contiguous().view(N1, -1)
#         b1_imag_tensor = b1_imag_tensor.contiguous().view(N1, -1)

#         self.b0 = b0_tensor
#         self.b1_real = b1_real_tensor
#         self.b1_imag = b1_imag_tensor

#     def init_pca_gmm(self):
#         """
#         description : initialize pca and gmm
#         param        {*} cfg
#         return       {([pca_b0, pca_b1_real, pca_b1_imag], [gmm_b0, gmm_b1_real, gmm_b1_imag])}
#         """

#         b0_pca = PCA(n_components=self.b0_num_components, svd_solver="full")
#         b1_real_pca = PCA(n_components=self.b1_real_num_components, svd_solver="full")
#         b1_imag_pca = PCA(n_components=self.b1_imag_num_components, svd_solver="full")

#         b0_feat_map = b0_pca.fit_transform(self.b0)
#         b1_real_feat_map = b1_real_pca.fit_transform(self.b1_real)
#         b1_imag_feat_map = b1_imag_pca.fit_transform(self.b1_imag)

#         b0_gmm = GaussianMixture(n_components=2, n_features=self.b0_num_components)
#         b1_real_gmm = GaussianMixture(
#             n_components=3,
#             n_features=self.b1_real_num_components,
#             covariance_type="diag",
#         )
#         b1_imag_gmm = GaussianMixture(
#             n_components=3,
#             n_features=self.b1_imag_num_components,
#             covariance_type="diag",
#         )

#         b0_gmm.fit(b0_feat_map)
#         b1_real_gmm.fit(b1_real_feat_map)
#         b1_imag_gmm.fit(b1_imag_feat_map)

#         self.pca_group = {"B0": b0_pca, "B1_real": b1_real_pca, "B1_imag": b1_imag_pca}
#         self.gmm_group = {"B0": b0_gmm, "B1_real": b1_real_gmm, "B1_imag": b1_imag_gmm}

#     def find_num_component(self):
#         pca_b0 = PCA(n_components=None, svd_solver="full")
#         pca_b1_real = PCA(n_components=None, svd_solver="full")
#         pca_b1_imag = PCA(n_components=None, svd_solver="full")

#         pca_b0_feat = pca_b0.fit(self.b0)
#         pca_b1_real_feat = pca_b1_real.fit(self.b1_real)
#         pca_b1_imag_feat = pca_b1_imag.fit(self.b1_imag)

#         b0_cumulative_explained_variance = torch.cumsum(
#             pca_b0_feat.explained_variance_ratio_, dim=0
#         )
#         b1_real_cumulative_explained_variance = torch.cumsum(
#             pca_b1_real_feat.explained_variance_ratio_, dim=0
#         )
#         b1_imag_cumulative_explained_variance = torch.cumsum(
#             pca_b1_imag_feat.explained_variance_ratio_, dim=0
#         )
#         # Desired explained variance
#         desired_variance = 0.95

#         # Find the number of components using torch.where or torch.nonzero
#         self.b0_num_components = (
#             torch.nonzero(
#                 b0_cumulative_explained_variance >= desired_variance, as_tuple=False
#             )[0].item()
#             + 1
#         )
#         self.b1_real_num_components = (
#             torch.nonzero(
#                 b1_real_cumulative_explained_variance >= desired_variance,
#                 as_tuple=False,
#             )[0].item()
#             + 1
#         )
#         self.b1_imag_num_components = (
#             torch.nonzero(
#                 b1_imag_cumulative_explained_variance >= desired_variance,
#                 as_tuple=False,
#             )[0].item()
#             + 1
#         )

#     def sample_new_map(self, bacth_size):
#         Nx, Ny = self.cfg.image.N
#         tx = 4
#         B0_maps = torch.zeros(bacth_size, Nx, Ny, dtype=self.dtype)
#         B1_real_maps = torch.zeros(bacth_size, tx, Nx, Ny, dtype=self.dtype)
#         B1_imag_maps = torch.zeros(bacth_size, tx, Nx, Ny, dtype=self.dtype)
#         for i in range(bacth_size):
#             for (pca_k, pca_v), (gmm_k, gmm_v) in zip(
#                 self.pca_group.items(), self.gmm_group.items()
#             ):
#                 sample_pca, _ = gmm_v.sample()
#                 # sample_pca: shape (1, D)
#                 reconstructed = pca_v.inverse_transform(sample_pca)

#                 if pca_k == "B0" and gmm_k == "B0":
#                     Bmap = reconstructed.reshape(Nx, Ny)
#                     B0_maps[i] = Bmap
#                 else:
#                     Bmap = reconstructed.reshape(tx, Nx, Ny)
#                     if gmm_k == "B1_real":
#                         B1_real_maps[i] = Bmap
#                     else:
#                         B1_imag_maps[i] = Bmap

#         B1_real_maps_pp, B1_imag_maps_pp = self.post_process(B1_real_maps, B1_imag_maps)
#         return 1e-6 * B0_maps, B1_real_maps_pp, B1_imag_maps_pp

#     def post_process(self, b1_real, b1_imag):
#         N, C, Nx, Ny = b1_real.shape
#         b1_real = b1_real.view(N * C, 1, Nx, Ny)
#         b1_imag = b1_imag.view(N * C, 1, Nx, Ny)
#         conv = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         conv.weight.data = torch.ones(1, 1, 3, 3) / 9
#         b1_real_pp = conv(b1_real)
#         b1_imag_pp = conv(b1_imag)
#         b1_real_pp = b1_real_pp.view(N, C, Nx, Ny) / 5
#         b1_imag_pp = b1_imag_pp.view(N, C, Nx, Ny) / 5
#         return b1_real_pp, b1_imag_pp


# class GaussianMixture(torch.nn.Module):
#     """
#     Fits a mixture of k=1,..,K Gaussians to the input data (K is supplied via n_components).
#     Input tensors are expected to be flat with dimensions (n: number of samples, d: number of features).
#     The model then extends them to (n, 1, d).
#     The model parametrization (mu, sigma) is stored as (1, k, d),
#     probabilities are shaped (n, k, 1) if they relate to an individual sample,
#     or (1, k, 1) if they assign membership probabilities to one of the mixture components.
#     """

#     def __init__(
#         self,
#         n_components,
#         n_features,
#         covariance_type="full",
#         eps=1.0e-6,
#         init_params="kmeans",
#         mu_init=None,
#         var_init=None,
#     ):
#         """
#         Initializes the model and brings all tensors into their required shape.
#         The class expects data to be fed as a flat tensor in (n, d).
#         The class owns:
#             x:               torch.Tensor (n, 1, d)
#             mu:              torch.Tensor (1, k, d)
#             var:             torch.Tensor (1, k, d) or (1, k, d, d)
#             pi:              torch.Tensor (1, k, 1)
#             covariance_type: str
#             eps:             float
#             init_params:     str
#             log_likelihood:  float
#             n_components:    int
#             n_features:      int
#         args:
#             n_components:    int
#             n_features:      int
#         options:
#             mu_init:         torch.Tensor (1, k, d)
#             var_init:        torch.Tensor (1, k, d) or (1, k, d, d)
#             covariance_type: str
#             eps:             float
#             init_params:     str
#         """
#         super(GaussianMixture, self).__init__()

#         self.n_components = n_components
#         self.n_features = n_features

#         self.mu_init = mu_init
#         self.var_init = var_init
#         self.eps = eps

#         self.log_likelihood = -np.inf

#         self.covariance_type = covariance_type
#         self.init_params = init_params

#         assert self.covariance_type in ["full", "diag"]
#         assert self.init_params in ["kmeans", "random"]

#         self._init_params()

#     def _init_params(self):
#         if self.mu_init is not None:
#             assert self.mu_init.size() == (1, self.n_components, self.n_features), (
#                 "Input mu_init does not have required tensor dimensions (1, %i, %i)"
#                 % (self.n_components, self.n_features)
#             )
#             # (1, k, d)
#             self.mu = torch.nn.Parameter(self.mu_init, requires_grad=False)
#         else:
#             self.mu = torch.nn.Parameter(
#                 torch.randn(1, self.n_components, self.n_features), requires_grad=False
#             )

#         if self.covariance_type == "diag":
#             if self.var_init is not None:
#                 # (1, k, d)
#                 assert self.var_init.size() == (
#                     1,
#                     self.n_components,
#                     self.n_features,
#                 ), (
#                     "Input var_init does not have required tensor dimensions (1, %i, %i)"
#                     % (self.n_components, self.n_features)
#                 )
#                 self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
#             else:
#                 self.var = torch.nn.Parameter(
#                     torch.ones(1, self.n_components, self.n_features),
#                     requires_grad=False,
#                 )
#         elif self.covariance_type == "full":
#             if self.var_init is not None:
#                 # (1, k, d, d)
#                 assert self.var_init.size() == (
#                     1,
#                     self.n_components,
#                     self.n_features,
#                     self.n_features,
#                 ), (
#                     "Input var_init does not have required tensor dimensions (1, %i, %i, %i)"
#                     % (self.n_components, self.n_features, self.n_features)
#                 )
#                 self.var = torch.nn.Parameter(self.var_init, requires_grad=False)
#             else:
#                 self.var = torch.nn.Parameter(
#                     torch.eye(self.n_features)
#                     .reshape(1, 1, self.n_features, self.n_features)
#                     .repeat(1, self.n_components, 1, 1),
#                     requires_grad=False,
#                 )

#         # (1, k, 1)
#         self.pi = torch.nn.Parameter(
#             torch.Tensor(1, self.n_components, 1), requires_grad=False
#         ).fill_(1.0 / self.n_components)
#         self.params_fitted = False

#     def check_size(self, x):
#         if len(x.size()) == 2:
#             # (n, d) --> (n, 1, d)
#             x = x.unsqueeze(1)

#         return x

#     def bic(self, x):
#         """
#         Bayesian information criterion for a batch of samples.
#         args:
#             x:      torch.Tensor (n, d) or (n, 1, d)
#         returns:
#             bic:    float
#         """
#         x = self.check_size(x)
#         n = x.shape[0]

#         # Free parameters for covariance, means and mixture components
#         free_params = (
#             self.n_features * self.n_components
#             + self.n_features
#             + self.n_components
#             - 1
#         )

#         bic = -2.0 * self.__score(
#             x, as_average=False
#         ).mean() * n + free_params * np.log(n)

#         return bic

#     def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
#         """
#         Fits model to the data.
#         args:
#             x:          torch.Tensor (n, d) or (n, k, d)
#         options:
#             delta:      float
#             n_iter:     int
#             warm_start: bool
#         """
#         if not warm_start and self.params_fitted:
#             self._init_params()

#         x = self.check_size(x)

#         if self.init_params == "kmeans" and self.mu_init is None:
#             mu = self.get_kmeans_mu(x, n_centers=self.n_components)
#             self.mu.data = mu

#         i = 0
#         j = np.inf

#         while (i <= n_iter) and (j >= delta):
#             log_likelihood_old = self.log_likelihood
#             mu_old = self.mu
#             var_old = self.var

#             self.__em(x)
#             self.log_likelihood = self.__score(x)

#             if torch.isinf(self.log_likelihood.abs()) or torch.isnan(
#                 self.log_likelihood
#             ):
#                 device = self.mu.device
#                 # When the log-likelihood assumes unbound values, reinitialize model
#                 self.__init__(
#                     self.n_components,
#                     self.n_features,
#                     covariance_type=self.covariance_type,
#                     mu_init=self.mu_init,
#                     var_init=self.var_init,
#                     eps=self.eps,
#                 )
#                 for p in self.parameters():
#                     p.data = p.data.to(device)
#                 if self.init_params == "kmeans":
#                     (self.mu.data,) = self.get_kmeans_mu(x, n_centers=self.n_components)

#             i += 1
#             j = self.log_likelihood - log_likelihood_old

#             if j <= delta:
#                 # When score decreases, revert to old parameters
#                 self.__update_mu(mu_old)
#                 self.__update_var(var_old)

#         self.params_fitted = True

#     def predict(self, x, probs=False):
#         """
#         Assigns input data to one of the mixture components by evaluating the likelihood under each.
#         If probs=True returns normalized probabilities of class membership.
#         args:
#             x:          torch.Tensor (n, d) or (n, 1, d)
#             probs:      bool
#         returns:
#             p_k:        torch.Tensor (n, k)
#             (or)
#             y:          torch.LongTensor (n)
#         """
#         x = self.check_size(x)

#         weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

#         if probs:
#             p_k = torch.exp(weighted_log_prob)
#             return torch.squeeze(p_k / (p_k.sum(1, keepdim=True)))
#         else:
#             return torch.squeeze(
#                 torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor)
#             )

#     def predict_proba(self, x):
#         """
#         Returns normalized probabilities of class membership.
#         args:
#             x:          torch.Tensor (n, d) or (n, 1, d)
#         returns:
#             y:          torch.LongTensor (n)
#         """
#         return self.predict(x, probs=True)

#     def sample(self, n=1):
#         """
#         Samples from the model.
#         args:
#             n:          int
#         returns:
#             x:          torch.Tensor (n, d)
#             y:          torch.Tensor (n)
#         """
#         counts = torch.distributions.multinomial.Multinomial(
#             total_count=n, probs=self.pi.squeeze()
#         ).sample()
#         x = torch.empty(0, device=counts.device)
#         y = torch.cat(
#             [
#                 torch.full([int(sample)], j, device=counts.device)
#                 for j, sample in enumerate(counts)
#             ]
#         )

#         # Only iterate over components with non-zero counts
#         for k in np.arange(self.n_components)[counts > 0]:
#             if self.covariance_type == "diag":
#                 x_k = self.mu[0, k] + torch.randn(
#                     int(counts[k]), self.n_features, device=x.device
#                 ) * torch.sqrt(self.var[0, k])
#             elif self.covariance_type == "full":
#                 d_k = torch.distributions.multivariate_normal.MultivariateNormal(
#                     self.mu[0, k], self.var[0, k]
#                 )
#                 x_k = torch.stack([d_k.sample() for _ in range(int(counts[k]))])

#             x = torch.cat((x, x_k), dim=0)

#         return x, y

#     def score_samples(self, x):
#         """
#         Computes log-likelihood of samples under the current model.
#         args:
#             x:          torch.Tensor (n, d) or (n, 1, d)
#         returns:
#             score:      torch.LongTensor (n)
#         """
#         x = self.check_size(x)

#         score = self.__score(x, as_average=False)
#         return score

#     def _estimate_log_prob(self, x):
#         """
#         Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
#         args:
#             x:            torch.Tensor (n, d) or (n, 1, d)
#         returns:
#             log_prob:     torch.Tensor (n, k, 1)
#         """
#         x = self.check_size(x)

#         if self.covariance_type == "full":
#             mu = self.mu
#             var = self.var

#             precision = torch.inverse(var)
#             d = x.shape[-1]

#             log_2pi = d * np.log(2.0 * pi)

#             log_det = self._calculate_log_det(precision)

#             x_mu_T = (x - mu).unsqueeze(-2)
#             x_mu = (x - mu).unsqueeze(-1)

#             x_mu_T_precision = calculate_matmul_n_times(
#                 self.n_components, x_mu_T, precision
#             )
#             x_mu_T_precision_x_mu = calculate_matmul(x_mu_T_precision, x_mu)

#             return -0.5 * (log_2pi - log_det + x_mu_T_precision_x_mu)

#         elif self.covariance_type == "diag":
#             mu = self.mu
#             prec = torch.rsqrt(self.var)

#             log_p = torch.sum(
#                 (mu * mu + x * x - 2 * x * mu) * prec, dim=2, keepdim=True
#             )
#             log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)

#             return -0.5 * (self.n_features * np.log(2.0 * pi) + log_p - log_det)

#     def _calculate_log_det(self, var):
#         """
#         Calculate log determinant in log space, to prevent overflow errors.
#         Args:
#             var: torch.Tensor (1, k, d, d) - Covariance matrices
#         """
#         log_det = torch.empty(size=(self.n_components,)).to(var.device)

#         for k in range(self.n_components):
#             epsilon = 1e-6 * torch.mean(
#                 torch.diagonal(var[0, k])
#             )  # Scale epsilon by matrix values
#             # Regularize covariance matrix by adding epsilon * Identity matrix
#             regularized_var = var[0, k] + epsilon * torch.eye(
#                 var.shape[-1], device=var.device
#             )

#             try:
#                 # Compute Cholesky decomposition
#                 chol = torch.linalg.cholesky(regularized_var)
#                 log_det[k] = 2 * torch.log(torch.diagonal(chol)).sum()
#             except RuntimeError as e:
#                 print(f"Cholesky decomposition failed for component {k}: {e}")
#                 log_det[k] = torch.tensor(
#                     float("-inf"), device=var.device
#                 )  # Assign -inf in case of failure

#         return log_det.unsqueeze(-1)

#     def _e_step(self, x):
#         """
#         Computes log-responses that indicate the (logarithmic) posterior belief (sometimes called responsibilities) that a data point was generated by one of the k mixture components.
#         Also returns the mean of the mean of the logarithms of the probabilities (as is done in sklearn).
#         This is the so-called expectation step of the EM-algorithm.
#         args:
#             x:              torch.Tensor (n, d) or (n, 1, d)
#         returns:
#             log_prob_norm:  torch.Tensor (1)
#             log_resp:       torch.Tensor (n, k, 1)
#         """
#         x = self.check_size(x)

#         weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)

#         log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
#         log_resp = weighted_log_prob - log_prob_norm

#         return torch.mean(log_prob_norm), log_resp

#     def _m_step(self, x, log_resp):
#         """
#         From the log-probabilities, computes new parameters pi, mu, var (that maximize the log-likelihood). This is the maximization step of the EM-algorithm.
#         args:
#             x:          torch.Tensor (n, d) or (n, 1, d)
#             log_resp:   torch.Tensor (n, k, 1)
#         returns:
#             pi:         torch.Tensor (1, k, 1)
#             mu:         torch.Tensor (1, k, d)
#             var:        torch.Tensor (1, k, d)
#         """
#         x = self.check_size(x)

#         resp = torch.exp(log_resp)

#         pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
#         mu = torch.sum(resp * x, dim=0, keepdim=True) / pi

#         if self.covariance_type == "full":
#             eps = (torch.eye(self.n_features) * self.eps).to(x.device)
#             var = (
#                 torch.sum(
#                     (x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2))
#                     * resp.unsqueeze(-1),
#                     dim=0,
#                     keepdim=True,
#                 )
#                 / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1)
#                 + eps
#             )

#         elif self.covariance_type == "diag":
#             x2 = (resp * x * x).sum(0, keepdim=True) / pi
#             mu2 = mu * mu
#             xmu = (resp * mu * x).sum(0, keepdim=True) / pi
#             var = x2 - 2 * xmu + mu2 + self.eps

#         pi = pi / x.shape[0]

#         return pi, mu, var

#     def __em(self, x):
#         """
#         Performs one iteration of the expectation-maximization algorithm by calling the respective subroutines.
#         args:
#             x:          torch.Tensor (n, 1, d)
#         """
#         _, log_resp = self._e_step(x)
#         pi, mu, var = self._m_step(x, log_resp)

#         self.__update_pi(pi)
#         self.__update_mu(mu)
#         self.__update_var(var)

#     def __score(self, x, as_average=True):
#         """
#         Computes the log-likelihood of the data under the model.
#         args:
#             x:                  torch.Tensor (n, 1, d)
#             sum_data:           bool
#         returns:
#             score:              torch.Tensor (1)
#             (or)
#             per_sample_score:   torch.Tensor (n)

#         """
#         weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
#         per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)

#         if as_average:
#             return per_sample_score.mean()
#         else:
#             return torch.squeeze(per_sample_score)

#     def __update_mu(self, mu):
#         """
#         Updates mean to the provided value.
#         args:
#             mu:         torch.FloatTensor
#         """
#         assert mu.size() in [
#             (self.n_components, self.n_features),
#             (1, self.n_components, self.n_features),
#         ], (
#             "Input mu does not have required tensor dimensions (%i, %i) or (1, %i, %i)"
#             % (self.n_components, self.n_features, self.n_components, self.n_features)
#         )

#         if mu.size() == (self.n_components, self.n_features):
#             self.mu = mu.unsqueeze(0)
#         elif mu.size() == (1, self.n_components, self.n_features):
#             self.mu.data = mu

#     def __update_var(self, var):
#         """
#         Updates variance to the provided value.
#         args:
#             var:        torch.FloatTensor
#         """
#         if self.covariance_type == "full":
#             assert var.size() in [
#                 (self.n_components, self.n_features, self.n_features),
#                 (1, self.n_components, self.n_features, self.n_features),
#             ], (
#                 "Input var does not have required tensor dimensions (%i, %i, %i) or (1, %i, %i, %i)"
#                 % (
#                     self.n_components,
#                     self.n_features,
#                     self.n_features,
#                     self.n_components,
#                     self.n_features,
#                     self.n_features,
#                 )
#             )

#             if var.size() == (self.n_components, self.n_features, self.n_features):
#                 self.var = var.unsqueeze(0)
#             elif var.size() == (1, self.n_components, self.n_features, self.n_features):
#                 self.var.data = var

#         elif self.covariance_type == "diag":
#             assert var.size() in [
#                 (self.n_components, self.n_features),
#                 (1, self.n_components, self.n_features),
#             ], (
#                 "Input var does not have required tensor dimensions (%i, %i) or (1, %i, %i)"
#                 % (
#                     self.n_components,
#                     self.n_features,
#                     self.n_components,
#                     self.n_features,
#                 )
#             )

#             if var.size() == (self.n_components, self.n_features):
#                 self.var = var.unsqueeze(0)
#             elif var.size() == (1, self.n_components, self.n_features):
#                 self.var.data = var

#     def __update_pi(self, pi):
#         """
#         Updates pi to the provided value.
#         args:
#             pi:         torch.FloatTensor
#         """
#         assert pi.size() in [(1, self.n_components, 1)], (
#             "Input pi does not have required tensor dimensions (%i, %i, %i)"
#             % (1, self.n_components, 1)
#         )

#         self.pi.data = pi

#     def get_kmeans_mu(self, x, n_centers, init_times=50, min_delta=1e-3):
#         """
#         Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
#         The algorithm is repeated init_times often, after which the best centerpoint is returned.
#         args:
#             x:            torch.FloatTensor (n, d) or (n, 1, d)
#             init_times:   init
#             min_delta:    int
#         """
#         if len(x.size()) == 3:
#             x = x.squeeze(1)
#         x_min, x_max = x.min(), x.max()
#         x = (x - x_min) / (x_max - x_min)

#         min_cost = np.inf

#         for i in range(init_times):
#             tmp_center = x[
#                 np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False),
#                 ...,
#             ]
#             l2_dis = torch.norm(
#                 (x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2
#             )
#             l2_cls = torch.argmin(l2_dis, dim=1)

#             cost = 0
#             for c in range(n_centers):
#                 cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

#             if cost < min_cost:
#                 min_cost = cost
#                 center = tmp_center

#         delta = np.inf

#         while delta > min_delta:
#             l2_dis = torch.norm(
#                 (x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2
#             )
#             l2_cls = torch.argmin(l2_dis, dim=1)
#             center_old = center.clone()

#             for c in range(n_centers):
#                 center[c] = x[l2_cls == c].mean(dim=0)

#             delta = torch.norm((center_old - center), dim=1).max()

#         return center.unsqueeze(0) * (x_max - x_min) + x_min
