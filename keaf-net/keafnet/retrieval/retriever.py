"""
Knowledge retrieval for a single VQA example.

Given an image's detected object labels and a question, this module:
  1. extracts seed entities (visual labels + question noun phrases),
  2. retrieves ranked triplets from the KGIndex (ConceptNet + CSKG),
  3. embeds each triplet string with Sentence-BERT.

spaCy and sentence-transformers are optional; lightweight fallbacks keep the
module importable and testable without them.

Reference: KEAF-Net, Section 3.3.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .kg_index import KGIndex

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    _HAS_SPACY = True
except Exception:  # pragma: no cover
    _NLP = None
    _HAS_SPACY = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:  # pragma: no cover
    _HAS_ST = False

_STOPWORDS = {
    "what", "which", "who", "where", "when", "why", "how", "is", "are", "the",
    "a", "an", "of", "in", "on", "to", "and", "this", "that", "there", "do",
    "does", "did", "can", "you", "see", "picture", "image", "photo",
}


def question_noun_phrases(question: str) -> list[str]:
    """Extract noun-phrase seeds from the question."""
    if _HAS_SPACY:
        doc = _NLP(question)
        phrases = [chunk.text.lower().strip() for chunk in doc.noun_chunks]
        return [p for p in phrases if p and p not in _STOPWORDS]
    # Fallback: content words minus stopwords.
    toks = [t.lower().strip("?.,!") for t in question.split()]
    return [t for t in toks if t and t not in _STOPWORDS and len(t) > 2]


class TripletEmbedder:
    """Wraps Sentence-BERT all-MiniLM-L6-v2 (384-d). Falls back to a
    deterministic hashing embedding when the library is unavailable."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384) -> None:
        self.dim = dim
        if _HAS_ST:
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        else:  # pragma: no cover
            self.model = None

    def encode(self, sentences: Sequence[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.model is not None:
            return self.model.encode(list(sentences), convert_to_numpy=True,
                                     normalize_embeddings=True).astype(np.float32)
        # Deterministic hashing fallback (keeps shapes correct for tests).
        out = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            rng = np.random.default_rng(abs(hash(s)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-8)
        return out


class KnowledgeRetriever:
    """End-to-end retrieval: entities -> triplets -> embeddings."""

    def __init__(self, kg: KGIndex, embedder: TripletEmbedder | None = None,
                 top_p: int = 50, hops: int = 2) -> None:
        self.kg = kg
        self.embedder = embedder or TripletEmbedder()
        self.top_p = top_p
        self.hops = hops

    def retrieve(self, visual_labels: Sequence[str], question: str):
        """Return (triplet_strings, triplet_embeddings, mask).

        triplet_embeddings: (top_p, dim) padded; mask: (top_p,) 1/0 valid.
        """
        seeds = [l.lower() for l in visual_labels] + question_noun_phrases(question)
        seeds = list(dict.fromkeys(seeds))  # dedupe, keep order

        triplets = self.kg.retrieve(seeds, hops=self.hops, top_p=self.top_p)
        strings = [f"{s} {r} {o}" for (s, r, o) in triplets]
        emb = self.embedder.encode(strings)

        # Pad / truncate to top_p.
        out = np.zeros((self.top_p, self.embedder.dim), dtype=np.float32)
        mask = np.zeros((self.top_p,), dtype=np.float32)
        m = min(len(strings), self.top_p)
        if m > 0:
            out[:m] = emb[:m]
            mask[:m] = 1.0
        return strings[:self.top_p], out, mask
