"""Evaluate a trained checkpoint and dump predictions.

    python scripts/evaluate.py --config configs/okvqa.yaml \
        --checkpoint outputs/keaf_net_seed42_best.pt --split val
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from keaf_net.config import load_config
from keaf_net.data import AnswerVocab, VQADataset, list_collate
from keaf_net.engine import Evaluator
from keaf_net.models import KEAFNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KEAF-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="where to write predictions json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab = AnswerVocab.load(cfg.data.answer_vocab)
    cfg.model.num_answers = len(vocab)

    model = KEAFNet(cfg.model)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    model.to(args.device)

    dataset = VQADataset(cfg.data.data_root, cfg.data.feature_root, cfg.data.knowledge_cache,
                         cfg.data.dataset, args.split, vocab)
    loader = DataLoader(dataset, batch_size=cfg.optim.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers, collate_fn=list_collate)

    result = Evaluator(model, vocab, device=args.device).evaluate(loader)
    if result["accuracy"] is not None:
        print(f"{cfg.data.dataset}/{args.split} accuracy = {100 * result['accuracy']:.2f}")

    out = args.out or os.path.join(cfg.output_dir, f"predictions_{cfg.data.dataset}_{args.split}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result["predictions"], fh, indent=2)
    print(f"wrote {len(result['predictions'])} predictions to {out}")


if __name__ == "__main__":
    main()
