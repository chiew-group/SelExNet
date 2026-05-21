#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from omegaconf import OmegaConf

from selexnet import (
    SelExNet,
    BlochSimTorch,
    Trainer,
    ROIDataset,
    ddp_setup,
    ddp_barrier,
    _is_main_process,
    set_determinism,
    auto_num_workers,
    resume_training,
)


def _load_tester():
    try:
        from selexnet.test import Tester
    except ModuleNotFoundError as exc:
        if exc.name == "src.test":
            raise ModuleNotFoundError(
                "Test mode requires src/test.py with a Tester implementation."
            ) from exc
        raise
    return Tester


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str, help="path to config file")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--test", action="store_true", help="run test only")
    parser.add_argument(
        "--ddp", action="store_true", help="enable multi-GPU distributed training"
    )
    # in parse_command()
    parser.add_argument(
        "--freeze",
        type=str,
        default="none",
        choices=["none", "rfnet", "gnet"],
        help="freeze a submodule",
    )
    parser.add_argument(
        "--finetune_weights", type=str, help="path to finetune checkpoint"
    )
    return parser.parse_args()


def freeze_submodule(model: torch.nn.Module, name: str):
    if name == "rfnet":
        assert hasattr(model, "rfnet")
        for p in model.rfnet.parameters():
            p.requires_grad_(False)
    elif name == "gnet":
        # only exists if cfg.train.joint is True
        if not hasattr(model, "gnet"):
            raise ValueError(
                "Requested freezing gnet, but cfg.train.joint=False (no gnet)."
            )
        for p in model.gnet.parameters():
            p.requires_grad_(False)
    elif name == "none":
        pass
    else:
        raise ValueError(f"Unknown freeze target: {name}")


def main():
    args = parse_command()
    cfg = OmegaConf.load(args.cfg)

    use_ddp = args.ddp
    is_main_process = True
    local_rank = 0

    if use_ddp:
        local_rank = ddp_setup()
        is_main_process = _is_main_process()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    set_determinism(cfg.train.deterministic, cfg.train.seed, use_ddp)

    resume = args.resume

    # -------------------------
    # Bloch simulator (FP32)
    # -------------------------
    bs = BlochSimTorch(
        cfg.image.fov,
        cfg.image.N,
        cfg.magnet.tp,
        cfg.magnet.gamma,
        cfg.magnet.grad_path,
        block_steps=cfg.magnet.block_steps,
    ).to(device)

    # -------------------------
    # Model, Optimizer, Scheduler
    # -------------------------
    model = SelExNet(cfg).to(device)

    # ---- apply freeze ----

    freeze_submodule(model, args.freeze)

    # (optional) print a sanity summary on rank0
    if is_main_process:
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[Freeze={args.freeze}] trainable params: {n_train}/{n_total} ({100 * n_train / n_total:.2f}%)"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.train.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=10, min_lr=1e-6
    )

    if args.finetune_weights is not None:
        if os.path.exists(args.finetune_weights):
            if is_main_process:
                print(f"Loading finetune checkpoint from {args.finetune_weights}")
            ckpt_path = args.finetune_weights
        else:
            if is_main_process:
                raise FileNotFoundError(
                    f"Model checkpoint {args.finetune_weights} not found"
                )

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
        model.eval()

    # ! IMPORTANT: resume/load must happen BEFORE DDP wrapping
    (
        model,
        optimizer,
        scheduler,
        resume_update,
        start_epoch,
        best_val_loss,
        train_losses,
        valid_losses,
    ) = resume_training(cfg, model, optimizer, scheduler, resume, is_main_process)

    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # -------------------------
    # Dataset / split
    # -------------------------
    dataset = ROIDataset(cfg, apply_bilateral_filter=False)

    train_ratio = 0.5
    valid_ratio = 0.5
    train_size = int(train_ratio * len(dataset))
    valid_size = int(valid_ratio * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset,
        [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(17),
    )

    # -------------------------
    # Samplers / loaders
    # -------------------------
    world_size = dist.get_world_size() if (use_ddp and dist.is_initialized()) else 1
    per_rank_batch = max(1, cfg.train.batch_size // world_size)
    num_workers = 0 if cfg.train.deterministic else auto_num_workers(world_size)
    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=17
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_batch,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    # Validation: simplest + stable = run on rank0 only inside Trainer
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=per_rank_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # -------------------------
    # Train / Test
    # -------------------------
    if args.test:
        Tester = _load_tester()
        tester = Tester(cfg, model, bs, device)  # type: ignore
        tester.run(test_loader)
        if use_ddp:
            ddp_barrier(local_rank)
            dist.destroy_process_group()
        return

    trainer = Trainer(
        cfg,
        model,
        optimizer,
        scheduler,
        bs,
        device,
        is_main_process=is_main_process,
        local_rank=local_rank,
    )

    trainer.start = start_epoch
    trainer.best_val_loss = best_val_loss
    trainer.train_losses = train_losses
    trainer.valid_losses = valid_losses

    # rank0 creates dirs & logger, then all ranks wait
    if is_main_process:
        trainer._prepare_dir()
        trainer._prepare_logger(resume=resume_update)
    if use_ddp:
        ddp_barrier(local_rank)
    trainer.run(train_loader, valid_loader)

    if use_ddp:
        ddp_barrier(local_rank)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
