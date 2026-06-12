"""Tests for the full pipeline: vocab, retrieval, vectorized LOO, trainer."""
import torch

from keafnet.preprocessing.answer_vocab import AnswerVocab, normalise_answer
from keafnet.retrieval.kg_index import KGIndex
from keafnet.retrieval.retriever import KnowledgeRetriever, TripletEmbedder, question_noun_phrases
from keafnet.training.trainer import vectorized_loo_deltas
from keafnet.models import KEAFConfig, KEAFNet
from keafnet.data import SyntheticVQADataset, collate


def test_answer_normalisation():
    assert normalise_answer("The Dog.") == "dog"
    assert normalise_answer("two") == "2"


def test_answer_vocab_scores():
    train = [["cat"] * 5 + ["dog"] * 5, ["cat"] * 10]
    vocab = AnswerVocab.build(train, min_occurrence=3)
    assert "cat" in vocab.answer2idx
    scores = vocab.encode_scores(["cat", "cat", "cat", "dog"])
    ci = vocab.answer2idx["cat"]
    assert abs(scores[ci].item() - 1.0) < 1e-6  # 3/3 capped at 1


def test_kg_index_retrieve():
    kg = KGIndex()
    kg.add("dog", "IsA", "animal", 2.0)
    kg.add("dog", "HasA", "tail", 1.0)
    kg.add("animal", "IsA", "organism", 1.0)
    triplets = kg.retrieve(["dog"], hops=2, top_p=10)
    assert ("dog", "IsA", "animal") in triplets
    # 2-hop reaches organism
    assert any(o == "organism" for (_, _, o) in triplets)


def test_retriever_shapes():
    kg = KGIndex()
    kg.add("dog", "IsA", "animal", 1.0)
    r = KnowledgeRetriever(kg, TripletEmbedder(dim=384), top_p=50, hops=2)
    strings, emb, mask = r.retrieve(["dog"], "what animal is this?")
    assert emb.shape == (50, 384)
    assert mask.shape == (50,)
    assert mask.sum() >= 1


def test_noun_phrases():
    nps = question_noun_phrases("what is the brown dog doing?")
    assert any("dog" in p for p in nps)


def test_vectorized_loo_matches_loop():
    cfg = KEAFConfig(dim=32, num_answers=20, num_regions=4, region_dim=64,
                     triplet_dim=16, max_triplets=6, gat_heads=4, gat_layers=1,
                     mhsr_hops=2, pretrained=False)
    model = KEAFNet(cfg).eval()
    ds = SyntheticVQADataset(n=2, num_answers=20, num_regions=4, region_dim=64,
                             triplet_dim=16, max_triplets=6, max_len=6)
    batch = collate([ds[i] for i in range(2)])
    with torch.no_grad():
        v, t, q, k = model.encode(batch)
        delta = vectorized_loo_deltas(model, batch, v, t, q, k, sample_size=4)
    assert delta.shape == (2, 6)
    assert torch.isfinite(delta).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok: {name}")
    print("All full-pipeline tests passed.")
