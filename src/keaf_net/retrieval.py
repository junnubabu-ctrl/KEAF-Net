"""Knowledge retrieval interfaces for ConceptNet + CSKG.

No third-party knowledge dump is redistributed here. Convert legally obtained
sources to the small JSONL index format documented in README.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass(frozen=True)
class Triplet:
    subject: str
    relation: str
    object: str
    source: str

    @property
    def text(self):
        return f"{self.subject} {self.relation} {self.object}"

class LocalKnowledgeIndex:
    def __init__(self, path):
        self.triplets=[]
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip(): self.triplets.append(Triplet(**json.loads(line)))

    def retrieve_lexical(self, entities, top_k=50):
        terms={e.lower().strip() for e in entities if e.strip()}
        scored=[]
        for t in self.triplets:
            words=set(t.text.lower().replace('_',' ').split())
            score=len(terms & words)
            if score: scored.append((score,t))
        scored.sort(key=lambda x:x[0],reverse=True)
        return [t for _,t in scored[:top_k]]


def encode_triplets(triplets, encoder):
    """Encode triplets with a SentenceTransformer-compatible `.encode` API."""
    texts=[t.text for t in triplets]
    return encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
