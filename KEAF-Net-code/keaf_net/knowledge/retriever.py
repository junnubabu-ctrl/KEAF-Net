"""External knowledge retrieval (Section 3.3).

Given the visual entities (top Faster R-CNN labels) and the question noun
phrases, we expand one- and two-hop neighbours over a merged ConceptNet 5.5 +
CSKG index, rank by edge weight, de-duplicate ``(subject, relation, object)``
and cap the candidate set at ``P`` triplets. The retriever is recall-oriented on
purpose; the AKF prunes the noise afterwards.

The index is built once from the knowledge-graph dumps (``KnowledgeIndex.from_csv``)
and cached. ``nltk`` (WordNet) is imported lazily for the entity linker.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

_HOP_DECAY = 0.5  # two-hop neighbours are down-weighted relative to one-hop


def normalize_entity(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class Triplet:
    subject: str
    relation: str
    object: str
    weight: float = 1.0

    def linearise(self) -> str:
        """Surface form fed to Sentence-BERT, Eq. (7)."""
        return f"{self.subject.replace('_', ' ')} {self.relation} {self.object.replace('_', ' ')}"


class KnowledgeIndex:
    """Adjacency over a merged knowledge graph keyed by entity."""

    def __init__(self) -> None:
        self._adj: dict[str, list[Triplet]] = defaultdict(list)

    def add(self, triplet: Triplet) -> None:
        self._adj[triplet.subject].append(triplet)
        self._adj[triplet.object].append(triplet)

    def neighbours(self, entity: str) -> list[Triplet]:
        return self._adj.get(normalize_entity(entity), [])

    @classmethod
    def from_csv(cls, path: str, weight_threshold: float = 0.0) -> "KnowledgeIndex":
        """Load assertions from a ``subject,relation,object,weight`` CSV."""
        index = cls()
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 3:
                    continue
                weight = float(row[3]) if len(row) > 3 and row[3] else 1.0
                if weight < weight_threshold:
                    continue
                index.add(Triplet(normalize_entity(row[0]), row[1].strip(),
                                  normalize_entity(row[2]), weight))
        return index


class KnowledgeRetriever:
    def __init__(self, index: KnowledgeIndex, max_triplets: int = 50, max_hops: int = 2):
        self.index = index
        self.max_triplets = max_triplets
        self.max_hops = max_hops

    def retrieve(self, seed_entities: Iterable[str]) -> list[Triplet]:
        """Rank one/two-hop neighbours by (decayed) edge weight and cap at P."""
        scored: dict[Triplet, float] = {}
        frontier = {normalize_entity(e) for e in seed_entities}
        seen_entities: set[str] = set()
        for hop in range(self.max_hops):
            decay = _HOP_DECAY ** hop
            next_frontier: set[str] = set()
            for entity in frontier - seen_entities:
                seen_entities.add(entity)
                for triplet in self.index.neighbours(entity):
                    score = triplet.weight * decay
                    if score > scored.get(triplet, 0.0):
                        scored[triplet] = score
                    next_frontier.update((triplet.subject, triplet.object))
            frontier = next_frontier
        ranked = sorted(scored, key=lambda t: scored[t], reverse=True)
        return ranked[: self.max_triplets]


class EntityLinker:
    """Two-stage linker: exact match, then WordNet-lemma match (Section 3.5)."""

    def __init__(self, use_wordnet: bool = True):
        self._lemmatize = _wordnet_lemmatizer() if use_wordnet else (lambda w: w)

    def _key(self, text: str) -> str:
        return self._lemmatize(normalize_entity(text).replace("_", " "))

    def link(self, mentions: list[str], triplets: list[Triplet]) -> list[tuple[int, int]]:
        """Index pairs ``(mention_idx, triplet_idx)`` for matched entities."""
        entity_keys = [(self._key(t.subject), self._key(t.object)) for t in triplets]
        pairs: list[tuple[int, int]] = []
        for mi, mention in enumerate(mentions):
            key = self._key(mention)
            if not key:
                continue
            for ti, (subj, obj) in enumerate(entity_keys):
                if key == subj or key == obj:
                    pairs.append((mi, ti))
        return pairs


def _wordnet_lemmatizer():
    from nltk.stem import WordNetLemmatizer  # lazy

    lemmatizer = WordNetLemmatizer()
    return lambda word: " ".join(lemmatizer.lemmatize(w) for w in word.split())
