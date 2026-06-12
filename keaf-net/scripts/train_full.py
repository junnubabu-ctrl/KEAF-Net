"""
Full KEAF-Net training launcher (production engine).

Single GPU / CPU:
    python -m scripts.train_full --config configs/keafnet_okvqa.yaml

Multi-GPU (DDP), e.g. 2 GPUs:
    torchrun --nproc_per_node=2 -m scripts.train_full --config configs/keafnet_okvqa.yaml

Smoke test on synthetic data:
    python -m scripts.train_full --synthetic --epochs 1 --cpu
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from keafnet.data import SyntheticVQADataset, collate
from keafnet.models import KEAFConfig, KEAFNet
from keafnet.training.trainer import Trainer

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def load_config(path, overrides):
    cfg = dict(epochs=20, batch_size=64, lr=1e-4, lr_min=1e-6, weight_decay=0.01,
               grad_clip=1.0, grad_accum=1, loo_sample_size=10, seed=42,
               num_workers=4, val_every=1, amp=True, tensorboard=False,
               ckpt_dir="checkpoints", log_dir="runs")
    if path and _HAS_YAML and os.path.exists(path):
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def build_loaders(cfg, synthetic, rank, world):
    if synthetic:
        train = SyntheticVQADataset(n=cfg.get("train_n", 256), seed=cfg["seed"])
        val = SyntheticVQADataset(n=cfg.get("val_n", 64), seed=cfg["seed"] + 1)
    else:  # pragma: no cover
        from keafnet.data import KnowledgeVQADataset
        train = KnowledgeVQADataset(cfg["train_annotations"], cfg["features_dir"],
                                    num_answers=cfg["num_answers"])
        val = KnowledgeVQADataset(cfg["val_annotations"], cfg["features_dir"],
                                  num_answers=cfg["num_answers"])
    tsamp = DistributedSampler(train, world, rank) if world > 1 else None
    vsamp = DistributedSampler(val, world, rank, shuffle=False) if world > 1 else None
    tl = DataLoader(train, batch_size=cfg["batch_size"], shuffle=(tsamp is None),
                    sampler=tsamp, num_workers=cfg["num_workers"], collate_fn=collate)
    vl = DataLoader(val, batch_size=cfg["batch_size"], shuffle=False,
                    sampler=vsamp, num_workers=cfg["num_workers"], collate_fn=collate)
    return tl, vl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    rank, world, local = setup_ddp()
    cfg = load_config(args.config, {"epochs": args.epochs})
    torch.manual_seed(cfg["seed"] + rank)

    device = "cpu" if args.cpu or not torch.cuda.is_available() else f"cuda:{local}"
    if rank == 0:
        print(f"[train_full] device={device} world_size={world} synthetic={args.synthetic}")

    num_answers = cfg.get("num_answers", 3129)
    model = KEAFNet(KEAFConfig(num_answers=num_answers,
                               pretrained=not args.synthetic)).to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.startswith("cuda") else None)

    tl, vl = build_loaders(cfg, args.synthetic, rank, world)
    trainer = Trainer(model, tl, vl, cfg, device=device, rank=rank, world_size=world)
    if args.resume:
        trainer.resume(args.resume)
    best = trainer.fit()
    if rank == 0:
        print(f"[done] best val acc = {best*100:.2f}%")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
