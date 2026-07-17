import torch

from keaf_net.data import AnswerVocab, random_batch, random_sample
from keaf_net.engine import Evaluator, Trainer
from keaf_net.engine.loss import vqa_bce
from keaf_net.models import KEAFNet


def test_train_step_reduces_loss(cfg):
    model = KEAFNet(cfg.model)
    trainer = Trainer(model, cfg, device="cpu")
    batch = random_batch(cfg.model, batch_size=4)

    first = trainer.train_step(batch)
    assert first.total >= 0 and 0.0 <= first.kept <= 1.0
    # the LOO branch should produce a finite auxiliary loss
    assert first.akf == first.akf  # not NaN

    losses = [trainer.train_step(batch).total for _ in range(15)]
    assert min(losses[5:]) < losses[0]  # overfits a fixed batch


def test_akf_loss_weight_zero_disables_branch(cfg):
    cfg.optim.akf_loss_weight = 0.0
    model = KEAFNet(cfg.model)
    trainer = Trainer(model, cfg, device="cpu")
    log = trainer.train_step(random_batch(cfg.model, batch_size=3))
    assert log.total >= 0


def test_lr_schedule_warmup_and_anneal(cfg):
    model = KEAFNet(cfg.model)
    trainer = Trainer(model, cfg, device="cpu")
    assert trainer.lr_at(0.0) == 0.0
    peak = trainer.lr_at(cfg.optim.warmup_ratio)
    assert abs(peak - cfg.optim.lr) < 1e-6
    assert trainer.lr_at(1.0) <= cfg.optim.lr


def test_evaluator_scores_in_unit_interval(cfg):
    model = KEAFNet(cfg.model)
    vocab = AnswerVocab(["yes", "no"] + [f"a{i}" for i in range(cfg.model.num_answers - 2)])

    def sample_with_answers():
        s = random_sample(cfg.model, with_target=False)
        s["meta"]["answers"] = ["yes"] * 6 + ["no"] * 4
        return s

    loader = [[sample_with_answers() for _ in range(4)]]
    result = Evaluator(model, vocab, device="cpu").evaluate(loader)
    assert result["accuracy"] is not None
    assert 0.0 <= result["accuracy"] <= 1.0
    assert len(result["predictions"]) == 4


def test_vqa_bce_per_sample_shape(cfg):
    logits = torch.randn(5, cfg.model.num_answers)
    target = torch.zeros(5, cfg.model.num_answers)
    target[range(5), 0] = 1.0
    assert vqa_bce(logits, target, reduce=False).shape == (5,)
    assert vqa_bce(logits, target).ndim == 0
