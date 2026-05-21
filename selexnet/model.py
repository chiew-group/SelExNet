#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2024-09-21 01:52:10
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-03-12 00:14:32
FilePath     : /Documents/sTx_B0_1/src/model.py
Description  : DeepControlV2 network architecture
I Love IU
Copyright (c) 2024 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

DEFAULT_NUM_GROUP = 32


# Include the get_num_groups function as defined earlier
def get_num_groups(num_channels, default_num_groups=32):
    num_groups = min(default_num_groups, num_channels)
    while num_groups > 0:
        if num_channels % num_groups == 0:
            return num_groups
        num_groups -= 1
    return 1  # Fallback to 1 if no divisor is found


def _get_norm_layer(num_channels, norm_type, eps=1e-8, is_fc=False):
    if is_fc:
        return nn.RMSNorm(num_channels, eps=eps)
    if norm_type == "group":
        num_groups = get_num_groups(num_channels, DEFAULT_NUM_GROUP)
        return nn.GroupNorm(num_groups, num_channels, eps=eps, affine=True)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, eps=eps, affine=True)
    elif norm_type == "batch":
        return nn.BatchNorm2d(num_channels, eps=eps, affine=True)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")


def _get_act_layer(act_type):
    if act_type == "swish":
        return MemoryEfficientSwish()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    elif act_type == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


def generate_spiral_trajectory_gradients(params, cfg):
    """
    Generates a spiral trajectory based on parameters.

    Args:
        num_points (int): Number of time points.
        k_max (float): Maximum k-space radius.
        params (dict): Contains 'n_turns', 'alpha', 'beta'.

    Returns:
        kx, ky (torch.Tensor): K-space trajectories.
    """
    n_turns = params[:, 0].unsqueeze(1) * float(cfg.magnet.ktraj.n_turns[0]) + float(
        cfg.magnet.ktraj.n_turns[1]
    )
    alpha = params[:, 1].unsqueeze(1) * float(cfg.magnet.ktraj.alpha[0]) + float(
        cfg.magnet.ktraj.alpha[1]
    )
    beta = params[:, 2].unsqueeze(1) * float(cfg.magnet.ktraj.beta[0]) + float(
        cfg.magnet.ktraj.beta[1]
    )
    kmax_factor = params[:, 3].unsqueeze(1) * float(
        cfg.magnet.ktraj.kmax_factor[0]
    ) + float(cfg.magnet.ktraj.kmax_factor[1])
    T = cfg.magnet.tp
    num_points = cfg.model.rf_output_dim // 2
    gamma = cfg.magnet.gamma / (2 * torch.pi)  # rad⋅s^-1⋅T^-1
    fov = cfg.image.fov  # m
    N = cfg.image.N  # pixels
    dt = T / num_points
    alpha = alpha.expand(-1, num_points)
    beta = beta.expand(-1, num_points)

    t = torch.linspace(0, T, num_points, device=params.device).unsqueeze(0)
    t_normalized = t / T  # Normalize to [0, 1]

    # Radial component with variable density
    k_max = (
        torch.sqrt(
            torch.tensor(cfg.magnet.tx, dtype=params.dtype, device=params.device)
        )
        * N[0]
    ) / (kmax_factor * 2.0 * fov[0])  # 1/m
    r = k_max * (1.0 - torch.pow(t_normalized, alpha))

    # Angular component with variable speed
    theta = 2.0 * torch.pi * n_turns * torch.pow(t_normalized, beta)

    # Cartesian coordinates
    kx = r * torch.cos(theta)  # 1/m
    ky = r * torch.sin(theta)  # 1/m

    dkx = torch.gradient(kx, spacing=dt, dim=-1)[0]  # rad/m/s
    dky = torch.gradient(ky, spacing=dt, dim=-1)[0]  # rad/m/s

    # Compute gradients
    Gx = dkx / gamma  # T/m
    Gy = dky / gamma  # T/m
    Gx = Gx.repeat_interleave(repeats=2, dim=-1)  # [B, T*2]
    Gy = Gy.repeat_interleave(repeats=2, dim=-1)  # [B, T*2]
    Gz = torch.zeros_like(Gx).requires_grad_(False)  # T/m

    return Gx, Gy, Gz


# Memory-efficient Swish activation function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = ctx.saved_tensors[0]
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_outputs * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
        return grad_input


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


# Bottleneck block with GroupNorm and optional dropout
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        norm="group",
        act="swish",
        dropout_rate=0.1,
    ):
        super(Bottleneck, self).__init__()
        width = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.norm1 = _get_norm_layer(width, norm)
        self.act1 = _get_act_layer(act)

        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.norm2 = _get_norm_layer(width, norm)
        self.act2 = _get_act_layer(act)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.norm3 = _get_norm_layer(out_channels, norm)

        self.act3 = _get_act_layer(act)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act3(out)

        return out


# Modified DCNV2 model with GroupNorm and dropout
class RFNet(nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_dim=2000,
        tx=8,
        dropout=0.5,
        norm="group",
        act="swish",
        block=[3, 4, 6, 3],
    ):
        super(RFNet, self).__init__()
        self.in_channels = 64
        self.tx = tx

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(
            input_dim, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.norm1 = _get_norm_layer(self.in_channels, norm)
        self.act1 = _get_act_layer(act)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define layers using the Bottleneck block
        self.layer1 = self._make_layer(
            Bottleneck,
            64,
            blocks=block[0],
            dropout=dropout,
            stride=1,
            norm=norm,
            act=act,
        )
        self.layer2 = self._make_layer(
            Bottleneck,
            128,
            blocks=block[1],
            dropout=dropout,
            stride=2,
            norm=norm,
            act=act,
        )
        self.layer3 = self._make_layer(
            Bottleneck,
            256,
            blocks=block[2],
            dropout=dropout,
            stride=2,
            norm=norm,
            act=act,
        )
        self.layer4 = self._make_layer(
            Bottleneck,
            512,
            blocks=block[3],
            dropout=dropout,
            stride=1,
            norm=norm,
            act=act,
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers for amplitude and phase
        self.dropout = nn.Dropout(dropout)
        self.fc_amp = nn.Linear(512 * Bottleneck.expansion, output_dim * tx)
        self.fc_phase = nn.Linear(512 * Bottleneck.expansion, output_dim * tx)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self, block, planes, blocks, dropout=0.5, stride=1, norm="group", act="swish"
    ):
        downsample = None
        out_channels = planes * block.expansion

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _get_norm_layer(out_channels, norm),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                norm=norm,
                act=act,
                dropout_rate=dropout,
            )
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    norm=norm,
                    act=act,
                    dropout_rate=dropout,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        # Initial layers
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling and fully connected layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        # Real and Imaginary outputs
        re = self.fc_amp(x)
        im = self.fc_phase(x)

        re_out = re.view(B, self.tx, -1)
        im_out = im.view(B, self.tx, -1)

        return re_out, im_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class GradNet(nn.Module):
    def __init__(
        self,
        cfg,
        input_dim=1,
        output_dim=4,
        dropout=0.5,
        norm="group",
        act="swish",
        block=[3, 4, 6, 3],
    ):
        super(GradNet, self).__init__()
        self.cfg = cfg
        self.in_channels = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(
            input_dim, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.norm1 = _get_norm_layer(self.in_channels, norm)
        self.act1 = _get_act_layer(act)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define layers using the Bottleneck block
        self.layer1 = self._make_layer(
            Bottleneck,
            64,
            blocks=block[0],
            dropout=dropout,
            stride=1,
            norm=norm,
            act=act,
        )
        self.layer2 = self._make_layer(
            Bottleneck,
            128,
            blocks=block[1],
            dropout=dropout,
            stride=2,
            norm=norm,
            act=act,
        )
        self.layer3 = self._make_layer(
            Bottleneck,
            256,
            blocks=block[2],
            dropout=dropout,
            stride=2,
            norm=norm,
            act=act,
        )
        self.layer4 = self._make_layer(
            Bottleneck,
            512,
            blocks=block[3],
            dropout=dropout,
            stride=1,
            norm=norm,
            act=act,
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers for amplitude and phase
        self.dropout = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(512 * Bottleneck.expansion, 512)
        self.fc_norm1 = _get_norm_layer(512, norm, is_fc=True)
        self.fc_act1 = _get_act_layer(act)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_norm2 = _get_norm_layer(128, norm, is_fc=True)
        self.fc_act2 = _get_act_layer(act)
        self.fc_3 = nn.Linear(128, 32)
        self.fc_norm3 = _get_norm_layer(32, norm, is_fc=True)
        self.fc_act3 = _get_act_layer(act)
        self.fc_grad = nn.Linear(32, output_dim)
        self.fc_grad_act = nn.Sigmoid()

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self, block, planes, blocks, dropout=0.5, stride=1, norm="group", act="swish"
    ):
        downsample = None
        out_channels = planes * block.expansion

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _get_norm_layer(out_channels, norm),
            )

        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                norm=norm,
                act=act,
                dropout_rate=dropout,
            )
        )
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    norm=norm,
                    act=act,
                    dropout_rate=dropout,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling and fully connected layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        # Real and Imaginary outputs
        x = self.fc_1(x)
        x = self.fc_norm1(x)
        x = self.fc_act1(x)
        x = self.fc_2(x)
        x = self.fc_norm2(x)
        x = self.fc_act2(x)
        x = self.fc_3(x)
        x = self.fc_norm3(x)
        x = self.fc_act3(x)
        x = self.fc_grad(x)
        ktraj = self.fc_grad_act(x)

        gx, gy, gz = generate_spiral_trajectory_gradients(ktraj, self.cfg)

        return gx, gy, gz

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(
                m, (nn.GroupNorm, nn.InstanceNorm2d, nn.BatchNorm2d, nn.LayerNorm)
            ):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SelExNet(nn.Module):
    def __init__(self, cfg):
        super(SelExNet, self).__init__()
        self.rfnet = RFNet(
            input_dim=cfg.model.in_dim,
            output_dim=cfg.model.rf_output_dim,
            tx=cfg.magnet.tx,
            dropout=cfg.train.dropout,
            norm=cfg.model.norm,
            act=cfg.model.act,
            block=cfg.model.block,
        )
        if cfg.train.joint:
            self.gnet = GradNet(
                cfg,
                input_dim=cfg.model.in_dim,
                output_dim=cfg.model.grad_output_dim,
                dropout=cfg.train.dropout,
                norm=cfg.model.norm,
                act=cfg.model.act,
                block=cfg.model.block,
            )

    def forward(self, x):
        rf_re, rf_im = self.rfnet(x)
        if not hasattr(self, "gnet"):
            return rf_re, rf_im, None, None, None
        gx, gy, gz = self.gnet(x)
        return rf_re, rf_im, gx, gy, gz
