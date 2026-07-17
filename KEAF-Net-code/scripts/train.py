"""Train KEAF-Net (Algorithm 2).

Example
-------
    python scripts/train.py --config configs/okvqa.yaml --seed 42

The three reported runs use ``--seed {42, 123, 2024}``; the tables report
mean +/- std over the three checkpoints.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from keaf_net.config import load_config
from keaf_net.data import AnswerVocab, VQADataset, list_collate, move_sample
from keaf_net.engine import Evaluator, Trainer
from keaf_net.models import KEAFNet
from keaf_net.utils import set_seed


def build_loaders(cfg):
    vocab = AnswerVocab.load(cfg.data.answer_vocab)
    common = dict(root=cfg.data.data_root, feature_root=cfg.data.feature_root,
                  knowledge_root=cfg.data.knowledge_cache, dataset=cfg.data.dataset)
    train = VQADataset(**common, split="train", vocab=vocab)
    val = VQADataset(**common, split="val", vocab=vocab)
    loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=cfg.optim.batch_size, shuffle=shuffle,
        num_workers=cfg.data.num_workers, collate_fn=list_collate)
    return vocab, loader(train, True), loader(val, False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KEAF-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.device = args.device
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    vocab, train_loader, val_loader = build_loaders(cfg)
    cfg.model.num_answers = len(vocab)
    model = KEAFNet(cfg.model)
    trainer = Trainer(model, cfg, device=cfg.device)
    evaluator = Evaluator(model, vocab, device=cfg.device)

    total_steps = cfg.optim.epochs * max(1, len(train_loader))
    best_acc, patience, step = -1.0, 0, 0
    for epoch in range(cfg.optim.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.optim.epochs}")
        for batch in progress:
            trainer.set_lr(trainer.lr_at(step / total_steps))
            log = trainer.train_step([move_sample(s, cfg.device) for s in batch])
            progress.set_postfix(loss=f"{log.total:.3f}", akf=f"{log.akf:.3f}", kept=f"{log.kept:.2f}")
            step += 1

        acc = evaluator.evaluate(val_loader)["accuracy"]
        print(f"[epoch {epoch + 1}] val accuracy = {acc:.4f}" if acc is not None else "[epoch] val done")
        if acc is not None and acc > best_acc:
            best_acc, patience = acc, 0
            torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                        "accuracy": acc, "epoch": epoch},
                       os.path.join(cfg.output_dir, f"{cfg.experiment}_seed{cfg.seed}_best.pt"))
        else:
            patience += 1
            if patience >= cfg.optim.early_stop_patience:
                print(f"early stopping at epoch {epoch + 1}; best val accuracy {best_acc:.4f}")
                break


if __name__ == "__main__":
    main()
