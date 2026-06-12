"""
KEAF-Net training entry point.

Usage:
    python -m scripts.train --config configs/keafnet_okvqa.yaml
    python -m scripts.train --synthetic            # smoke test, no real data

Implements the training objective L = L_VQA + lambda * L_AKF with AdamW, a
cosine LR schedule, gradient clipping, and periodic validation.

Reference: KEAF-Net, Section 5 (Implementation Details).
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from keafnet.data import SyntheticVQADataset, collate
from keafnet.models import KEAFConfig, KEAFNet
from keafnet.utils import accuracy_from_logits

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def load_config(path: str | None) -> dict:
    defaults = dict(
        epochs=20, batch_size=64, lr=1e-4, weight_decay=0.01,
        warmup_steps=500, grad_clip=1.0, loo_sample_size=10,
        seed=42, num_workers=0, val_every=1, ckpt_dir="checkpoints",
    )
    if path and _HAS_YAML and os.path.exists(path):
        with open(path) as f:
            defaults.update(yaml.safe_load(f) or {})
    return defaults


def build_loaders(cfg: dict, synthetic: bool):
    if synthetic:
        train = SyntheticVQADataset(n=cfg.get("train_n", 256), seed=cfg["seed"])
        val = SyntheticVQADataset(n=cfg.get("val_n", 64), seed=cfg["seed"] + 1)
    else:  # pragma: no cover - requires real features on disk
        from keafnet.data import KnowledgeVQADataset
        train = KnowledgeVQADataset(cfg["train_annotations"], cfg["features_dir"])
        val = KnowledgeVQADataset(cfg["val_annotations"], cfg["features_dir"])
    tl = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], collate_fn=collate)
    vl = DataLoader(val, batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=cfg["num_workers"], collate_fn=collate)
    return tl, vl


def to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    accs = []
    for batch in loader:
        batch = to_device(batch, device)
        logits, _ = model(batch)
        accs.append(accuracy_from_logits(logits, batch["answer_scores"]))
    return sum(accs) / max(len(accs), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="use random data to smoke-test the full loop")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    torch.manual_seed(cfg["seed"])

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[train] device={device} synthetic={args.synthetic}")

    model_cfg = KEAFConfig(pretrained=not args.synthetic)
    model = KEAFNet(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params: {n_params/1e6:.1f}M total, {n_train/1e6:.1f}M trainable")

    tl, vl = build_loaders(cfg, args.synthetic)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    total_steps = cfg["epochs"] * max(len(tl), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps,
                                                       eta_min=1e-6)

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    best = -1.0
    for epoch in range(cfg["epochs"]):
        model.train()
        for step, batch in enumerate(tl):
            batch = to_device(batch, device)
            loss, logs = model.compute_loss(batch, cfg["loo_sample_size"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            if step % 10 == 0:
                print(f"  e{epoch} s{step} loss={logs['loss']:.4f} "
                      f"vqa={logs['l_vqa']:.4f} akf={logs['l_akf']:.4f}")

        if (epoch + 1) % cfg["val_every"] == 0:
            acc = evaluate(model, vl, device)
            print(f"[val] epoch {epoch} acc={acc:.4f}")
            if acc > best:
                best = acc
                torch.save({"model": model.state_dict(), "cfg": model_cfg.__dict__},
                           os.path.join(cfg["ckpt_dir"], "best.pt"))
                print(f"[ckpt] saved best.pt (acc={acc:.4f})")

    print(f"[done] best val acc={best:.4f}")


if __name__ == "__main__":
    main()
