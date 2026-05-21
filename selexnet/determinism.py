#!/usr/bin/env python
# coding=utf-8

import os
import torch
import torch.distributed as dist

__all__ = [
    "set_determinism",
    "auto_num_workers",
]


def auto_num_workers(world_size: int, max_workers: int = 8):
    cpu = os.cpu_count()
    if cpu is None:
        return 2
    return max(2, min(max_workers, cpu // world_size))


def set_seed(seed: int, use_ddp: bool):
    # Different seed per-rank is OK; DDP sampler handles epoch shuffling separately.
    if use_ddp and dist.is_initialized():
        seed = seed + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_determinism(
    deterministic: bool,
    seed: int,
    use_ddp: bool = False,
):
    """
    Configure full training determinism.

    If deterministic=True:
        - Disables TF32
        - Disables cuDNN benchmark
        - Enables deterministic algorithms
        - Fixes all RNG seeds

    If deterministic=False:
        - Enables fast paths (TF32 optional)
    """

    # -------------------------
    # Seeds
    # -------------------------
    set_seed(seed, use_ddp)

    # -------------------------
    # Backend control
    # -------------------------

    # TF32 and cuDNN benchmark
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if deterministic:
        # cuDNN
        torch.backends.cudnn.deterministic = True

        # Enforce deterministic ops
        torch.use_deterministic_algorithms(True)

        # Avoid nondeterministic workspace reuse
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    else:
        # Fast mode
        torch.backends.cudnn.deterministic = False

        # TF32 left to user preference
        torch.use_deterministic_algorithms(False)
