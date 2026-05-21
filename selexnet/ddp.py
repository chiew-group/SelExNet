#!/usr/bin/env python
# coding=utf-8

import os
import torch
import torch.distributed as dist

__all__ = [
    "ddp_setup",
    "ddp_is_initialized",
    "ddp_barrier",
    "_is_main_process",
]


def ddp_setup() -> int:
    """Initialize NCCL DDP. Returns local_rank."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )
    return local_rank


def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_barrier(local_rank: int | None = None):
    """
    Safe DDP barrier.
    - Uses device_ids to avoid NCCL warnings.
    - No-op if DDP is not initialized.
    """
    if not ddp_is_initialized():
        return

    if local_rank is None:
        dist.barrier()
    else:
        dist.barrier(device_ids=[local_rank])


def _is_main_process() -> bool:
    return not ddp_is_initialized() or dist.get_rank() == 0
