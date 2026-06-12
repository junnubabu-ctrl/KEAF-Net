"""
Production training engine for KEAF-Net.

Adds, on top of the minimal `scripts/train.py`:
  * mixed-precision (AMP) training,
  * gradient accumulation,
  * checkpoint save/resume (model + optimizer + scheduler + step),
  * optional Distributed Data Parallel (DDP) across GPUs,
  * TensorBoard logging (optional),
  * a vectorized LOO estimator for the AKF supervision.

Use via `scripts/train_full.py`.

Reference: KEAF-Net, Section 5.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:  # pragma: no cover
    _HAS_TB = False


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_acc: float = -1.0


def vectorized_loo_deltas(model, batch, v, t, q_cls, k, sample_size: int = 10):
    """Vectorized leave-one-out delta estimation.

    Instead of looping over triplets, we build a batch of masked copies (one per
    sampled triplet) and run a single forward pass. For B examples and S samples
    this evaluates B*(S+1) graphs at once.
    """
    soft = batch["answer_scores"]
    triplet_mask = batch.get("triplet_mask")
    boxes = batch.get("boxes")
    b, p, d = k.shape
    device = k.device

    base_logits, _ = model.reason(v, t, q_cls, k, triplet_mask, boxes)
    base_loss = F.binary_cross_entropy_with_logits(
        base_logits, soft, reduction="none").mean(dim=1)  # (B,)

    idx = torch.randint(0, p, (sample_size,), device=device)
    delta = torch.zeros(b, p, device=device)

    # Expand each example S times with a different triplet zeroed.
    v_e = v.repeat_interleave(sample_size, 0)
    t_e = t.repeat_interleave(sample_size, 0)
    q_e = q_cls.repeat_interleave(sample_size, 0)
    soft_e = soft.repeat_interleave(sample_size, 0)
    mask_e = triplet_mask.repeat_interleave(sample_size, 0) if triplet_mask is not None else None
    boxes_e = boxes.repeat_interleave(sample_size, 0) if boxes is not None else None

    k_e = k.repeat_interleave(sample_size, 0).clone()  # (B*S, P, d)
    rows = torch.arange(b * sample_size, device=device)
    drop_idx = idx.repeat(b)  # which triplet to drop in each expanded row
    k_e[rows, drop_idx] = 0.0

    logits_e, _ = model.reason(v_e, t_e, q_e, k_e, mask_e, boxes_e)
    loss_e = F.binary_cross_entropy_with_logits(
        logits_e, soft_e, reduction="none").mean(dim=1)  # (B*S,)
    loss_e = loss_e.view(b, sample_size)
    base_rep = base_loss.unsqueeze(1)
    d_vals = loss_e - base_rep  # (B, S)
    delta[torch.arange(b, device=device).unsqueeze(1), idx.unsqueeze(0).expand(b, -1)] = d_vals
    return delta


class Trainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader,
                 cfg: dict, device: str = "cuda", rank: int = 0,
                 world_size: int = 1) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg["weight_decay"])
        total = cfg["epochs"] * max(len(train_loader), 1)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=total, eta_min=cfg.get("lr_min", 1e-6))
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.get("amp", True)
                                                and device == "cuda")
        self.accum = cfg.get("grad_accum", 1)
        self.state = TrainState()
        self.writer = (SummaryWriter(cfg.get("log_dir", "runs"))
                       if _HAS_TB and self.is_main and cfg.get("tensorboard", False)
                       else None)

    # ------------------------------------------------------------ checkpoint

    def save(self, path: str, acc: float) -> None:
        if not self.is_main:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        raw = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({
            "model": raw.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": self.state.__dict__,
            "cfg": raw.cfg.__dict__,
        }, path)

    def resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        raw = self.model.module if hasattr(self.model, "module") else self.model
        raw.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.sched.load_state_dict(ckpt["sched"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.state = TrainState(**ckpt["state"])
        if self.is_main:
            print(f"[resume] from {path} @ epoch {self.state.epoch}")

    # ------------------------------------------------------------ loops

    def _to_device(self, batch):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def train_epoch(self) -> None:
        self.model.train()
        raw = self.model.module if hasattr(self.model, "module") else self.model
        for step, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                v, t, q_cls, k = raw.encode(batch)
                logits, aux = raw.reason(v, t, q_cls, k,
                                         batch.get("triplet_mask"), batch.get("boxes"))
                l_vqa = raw.vqa_loss(logits, batch["answer_scores"])
                with torch.no_grad():
                    delta = vectorized_loo_deltas(raw, batch, v, t, q_cls, k,
                                                  self.cfg["loo_sample_size"])
                targets = raw.akf.soft_targets_from_loo(delta)
                l_akf = raw.akf.loss(aux["alpha"], targets, batch.get("triplet_mask"))
                loss = (l_vqa + raw.cfg.loss_lambda * l_akf) / self.accum

            self.scaler.scale(loss).backward()
            if (step + 1) % self.accum == 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.cfg["grad_clip"])
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                self.sched.step()
                self.state.global_step += 1

            if self.is_main and step % 20 == 0:
                print(f"  e{self.state.epoch} s{step} "
                      f"loss={loss.item()*self.accum:.4f} "
                      f"vqa={l_vqa.item():.4f} akf={l_akf.item():.4f}")
                if self.writer:
                    gs = self.state.global_step
                    self.writer.add_scalar("loss/total", loss.item() * self.accum, gs)
                    self.writer.add_scalar("loss/vqa", l_vqa.item(), gs)
                    self.writer.add_scalar("loss/akf", l_akf.item(), gs)

    @torch.no_grad()
    def validate(self) -> float:
        from ..utils import accuracy_from_logits
        self.model.eval()
        accs = []
        for batch in self.val_loader:
            batch = self._to_device(batch)
            logits, _ = self.model(batch)
            accs.append(accuracy_from_logits(logits, batch["answer_scores"]))
        return sum(accs) / max(len(accs), 1)

    def fit(self) -> float:
        for epoch in range(self.state.epoch, self.cfg["epochs"]):
            self.state.epoch = epoch
            if hasattr(self.train_loader, "sampler") and hasattr(
                    self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            self.train_epoch()
            if (epoch + 1) % self.cfg.get("val_every", 1) == 0:
                acc = self.validate()
                if self.is_main:
                    print(f"[val] epoch {epoch} acc={acc*100:.2f}%")
                    if self.writer:
                        self.writer.add_scalar("val/acc", acc, epoch)
                    self.save(os.path.join(self.cfg["ckpt_dir"], "last.pt"), acc)
                    if acc > self.state.best_acc:
                        self.state.best_acc = acc
                        self.save(os.path.join(self.cfg["ckpt_dir"], "best.pt"), acc)
                        print(f"[ckpt] new best {acc*100:.2f}%")
        return self.state.best_acc
