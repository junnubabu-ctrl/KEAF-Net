"""Training engine (Algorithm 2).

The only non-standard part of the loop is the sampled leave-one-out (LOO)
estimate of triplet relevance (Section 3.4). For each batch we:

1. run the full forward pass and keep the per-sample answer loss;
2. for up to ``|S|`` retained triplets per sample, re-run *only* the reasoning
   path with that triplet removed and measure the change in answer loss
   ``d_j = L_VQA(K \\ {j}) - L_VQA(K)`` (Eq. 11), under ``no_grad`` because the
   target is stop-gradient anyway;
3. turn the differences into soft targets and regress the AKF scores onto them
   (Eq. 12-13).

This keeps the LOO overhead to ``|S|`` reasoning passes per batch rather than
one per candidate triplet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim import AdamW

from ..config import Config
from ..models import KEAFNet
from .loss import vqa_bce


@dataclass
class StepOutput:
    total: float
    vqa: float
    akf: float
    kept: float       # mean fraction of triplets retained by the filter


class Trainer:
    def __init__(self, model: KEAFNet, cfg: Config, device: torch.device | str = "cpu"):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = torch.device(device)
        self.opt = AdamW(
            model.parameters(), lr=cfg.optim.lr,
            betas=cfg.optim.betas, weight_decay=cfg.optim.weight_decay,
        )
        self._step = 0

    # --------------------------------------------------------------- LOO loss
    def _akf_loss(self, out: dict, target: torch.Tensor, loss_full: torch.Tensor) -> torch.Tensor:
        """Retrieval-consistency loss from sampled LOO differences."""
        batch = out["batch"]
        k_index, scores, gate = out["knowledge_index"], out["akf_scores"], out["gate"]
        node_keep = out["node_keep"].detach()
        s_max = self.cfg.model.akf_loo_samples

        # score position for each knowledge node id
        pos_of = scores.new_full((batch.num_nodes,), -1, dtype=torch.long)
        pos_of[k_index] = torch.arange(k_index.numel(), device=scores.device)

        # retained triplets grouped per sample
        retained = (gate.detach() > 0).nonzero(as_tuple=False).squeeze(-1)
        if retained.numel() == 0:
            return scores.new_zeros(())
        retained_nodes = k_index[retained]
        retained_batch = batch.node_batch[retained_nodes]

        # build a [B, s_max] table of sampled retained node ids (-1 = empty slot)
        table = retained_nodes.new_full((batch.num_graphs, s_max), -1)
        for b in range(batch.num_graphs):
            nodes = retained_nodes[retained_batch == b]
            if nodes.numel() == 0:
                continue
            perm = torch.randperm(nodes.numel(), device=nodes.device)[:s_max]
            table[b, : perm.numel()] = nodes[perm]

        sampled_scores, sampled_targets = [], []
        for r in range(s_max):
            col = table[:, r]
            active = col >= 0
            if not active.any():
                continue
            keep_r = node_keep.clone()
            keep_r[col[active]] = 0.0
            with torch.no_grad():
                logits_r, _ = self.model.reason(batch, keep_r)
                loss_r = vqa_bce(logits_r, target, reduce=False)
            delta = loss_r[active] - loss_full[active]            # Eq. (11)
            sampled_scores.append(scores[pos_of[col[active]]])
            sampled_targets.append(delta)

        if not sampled_scores:
            return scores.new_zeros(())
        return self.model.akf.loss(torch.cat(sampled_scores), torch.cat(sampled_targets))

    # --------------------------------------------------------------- one step
    def compute_losses(self, samples) -> tuple[torch.Tensor, StepOutput]:
        cfg = self.cfg
        out = self.model(samples)
        target = out["batch"].answer_target
        if target is None:
            raise ValueError("answer_target is required for training")

        vqa = vqa_bce(out["logits"], target)
        loss_full = vqa_bce(out["logits"].detach(), target, reduce=False)
        akf = self._akf_loss(out, target, loss_full)
        total = vqa + cfg.optim.akf_loss_weight * akf

        kept = (out["gate"].detach() > 0).float().mean() if out["gate"].numel() else torch.zeros(())
        log = StepOutput(float(total.detach()), float(vqa.detach()), float(akf.detach()), float(kept))
        return total, log

    def train_step(self, samples) -> StepOutput:
        self.model.train()
        total, log = self.compute_losses(samples)
        (total / self.cfg.optim.grad_accum_steps).backward()
        self._step += 1
        if self._step % self.cfg.optim.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return log

    # --------------------------------------------------------------- schedule
    def lr_at(self, progress: float) -> float:
        """Cosine schedule with linear warm-up; ``progress`` in [0, 1]."""
        o = self.cfg.optim
        if progress < o.warmup_ratio:
            return o.lr * progress / max(o.warmup_ratio, 1e-8)
        t = (progress - o.warmup_ratio) / max(1 - o.warmup_ratio, 1e-8)
        return o.min_lr + 0.5 * (o.lr - o.min_lr) * (1 + math.cos(math.pi * t))

    def set_lr(self, lr: float) -> None:
        for group in self.opt.param_groups:
            group["lr"] = lr
