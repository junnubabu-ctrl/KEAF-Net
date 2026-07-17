import torch

from keaf_net.data import AnswerVocab
from keaf_net.utils.metrics import vqa_soft_accuracy, wilcoxon_pvalue
from keaf_net.utils.text import normalize_answer


def test_normalize_answer_rules():
    assert normalize_answer("The Dog.") == "dog"
    assert normalize_answer("TWO") == "2"
    assert normalize_answer("  a Cat  ") == "cat"


def test_vocab_build_and_soft_target():
    annotations = [["yes"] * 10, ["no"] * 6 + ["yes"] * 4, ["maybe"]]
    vocab = AnswerVocab.build(annotations, num_answers=3)
    assert "yes" in vocab.stoi and "no" in vocab.stoi

    target = vocab.soft_target(["yes", "yes", "yes", "no"])
    assert target[vocab.stoi["yes"]] == 1.0          # 3 votes -> min(1, 3/3)
    assert abs(target[vocab.stoi["no"]] - 1 / 3) < 1e-6


def test_vocab_roundtrip(tmp_path):
    vocab = AnswerVocab(["yes", "no", "dog"])
    path = tmp_path / "vocab.json"
    vocab.save(str(path))
    assert AnswerVocab.load(str(path)).itos == vocab.itos


def test_vqa_soft_accuracy_caps_at_one():
    answers = ["dog"] * 4 + ["cat"] * 6
    assert vqa_soft_accuracy("dog", answers) == 1.0   # 4 votes -> min(1, 4/3)
    assert abs(vqa_soft_accuracy("bird", ["bird"] + ["x"] * 9) - 1 / 3) < 1e-6


def test_wilcoxon_identical_vectors():
    assert wilcoxon_pvalue([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == 1.0
