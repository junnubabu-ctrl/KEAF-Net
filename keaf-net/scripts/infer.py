"""
KEAF-Net single-example inference.

Demonstrates a forward pass and shows the AKF keep/discard decisions and the
MHSR per-hop attention, which together make the reasoning interpretable.

Usage:
    python -m scripts.infer --synthetic
    python -m scripts.infer --ckpt checkpoints/best.pt --features path/to/example.pt
"""
from __future__ import annotations

import argparse

import torch

from keafnet.data import SyntheticVQADataset, collate
from keafnet.models import KEAFConfig, KEAFNet


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--features", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        cfg = KEAFConfig(**state["cfg"])
        model = KEAFNet(cfg).to(device)
        model.load_state_dict(state["model"])
    else:
        model = KEAFNet(KEAFConfig(pretrained=not args.synthetic)).to(device)
    model.eval()

    if args.features:
        example = torch.load(args.features, map_location="cpu")
        batch = collate([example])
    else:
        batch = collate([SyntheticVQADataset(n=1)[0]])

    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    logits, aux = model(batch)
    pred = logits.argmax(dim=-1).item()
    kept = int(aux["keep"][0].sum().item())
    total = int(batch["triplet_mask"][0].sum().item()) if "triplet_mask" in batch else aux["keep"].shape[1]

    print(f"[infer] predicted answer index : {pred}")
    print(f"[infer] AKF kept {kept}/{total} candidate triplets")
    print(f"[infer] MHSR hops               : {len(aux['attn'])}")
    for h, a in enumerate(aux["attn"]):
        top = a[0].topk(min(3, a.shape[1]))
        idxs = top.indices.tolist()
        vals = [round(v, 3) for v in top.values.tolist()]
        print(f"         hop {h+1} top nodes: {list(zip(idxs, vals))}")


if __name__ == "__main__":
    main()
