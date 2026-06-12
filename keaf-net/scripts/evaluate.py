"""
KEAF-Net evaluation entry point.

Usage:
    python -m scripts.evaluate --ckpt checkpoints/best.pt --config configs/keafnet_okvqa.yaml
    python -m scripts.evaluate --synthetic   # smoke test

Reports VQA soft accuracy on the evaluation split (OK-VQA uses the 5,046-question
validation set as the standard test set).

Reference: KEAF-Net, Section 5.
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from keafnet.data import SyntheticVQADataset, collate
from keafnet.models import KEAFConfig, KEAFNet
from keafnet.utils import accuracy_from_logits


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        cfg = KEAFConfig(**state["cfg"])
        model = KEAFNet(cfg).to(device)
        model.load_state_dict(state["model"])
        print(f"[eval] loaded {args.ckpt}")
    else:
        model = KEAFNet(KEAFConfig(pretrained=not args.synthetic)).to(device)
        print("[eval] no checkpoint; evaluating randomly initialised model")

    if args.synthetic or not args.ckpt:
        ds = SyntheticVQADataset(n=128, seed=123)
    else:  # pragma: no cover
        from keafnet.data import KnowledgeVQADataset
        import yaml
        with open(args.config) as f:
            c = yaml.safe_load(f)
        ds = KnowledgeVQADataset(c["val_annotations"], c["features_dir"])

    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
    model.eval()
    accs = []
    for batch in loader:
        batch = to_device(batch, device)
        logits, _ = model(batch)
        accs.append(accuracy_from_logits(logits, batch["answer_scores"]))
    acc = sum(accs) / max(len(accs), 1)
    print(f"[eval] VQA soft accuracy = {acc*100:.2f}%")


if __name__ == "__main__":
    main()
