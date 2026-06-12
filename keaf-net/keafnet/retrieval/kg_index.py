"""
Knowledge-graph index over ConceptNet 5.5 (+ CSKG).

Loads a knowledge graph from the ConceptNet CSV assertions dump (and optionally
CSKG edges), builds an in-memory adjacency keyed by surface term, and exposes a
1-hop / 2-hop neighbour query that returns ranked `(subject, relation, object)`
triplets for a set of seed entities.

This is the retrieval backend used during feature preprocessing
(see `keafnet/retrieval/retriever.py`).

Reference: KEAF-Net, Section 3.3.
"""
from __future__ import annotations

import csv
import gzip
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

_TERM_RE = re.compile(r"^/c/en/([^/]+)")


def _surface(uri: str) -> str | None:
    """Map a ConceptNet URI like /c/en/dog/n to the surface form 'dog'."""
    m = _TERM_RE.match(uri)
    if not m:
        return None
    return m.group(1).replace("_", " ")


def _rel(uri: str) -> str:
    """Map /r/IsA -> 'IsA'."""
    return uri.rsplit("/", 1)[-1]


@dataclass
class KGIndex:
    """In-memory knowledge-graph adjacency.

    edges[term] -> list of (relation, object_term, weight)
    """
    edges: dict[str, list[tuple[str, str, float]]] = field(default_factory=lambda: defaultdict(list))

    def add(self, subj: str, rel: str, obj: str, weight: float = 1.0) -> None:
        self.edges[subj].append((rel, obj, weight))

    # -------------------------------------------------------------- loaders

    @classmethod
    def from_conceptnet_csv(cls, path: str, max_edges: int | None = None,
                            english_only: bool = True) -> "KGIndex":
        """Load ConceptNet from the official `conceptnet-assertions-5.x.csv(.gz)`.

        Each line: uri \\t relation \\t start \\t end \\t json_metadata
        """
        idx = cls()
        opener = gzip.open if path.endswith(".gz") else open
        n = 0
        with opener(path, "rt", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 5:
                    continue
                _, rel_uri, start_uri, end_uri, meta = row[:5]
                s = _surface(start_uri)
                o = _surface(end_uri)
                if s is None or o is None:
                    if english_only:
                        continue
                weight = 1.0
                m = re.search(r'"weight":\s*([0-9.]+)', meta)
                if m:
                    weight = float(m.group(1))
                idx.add(s, _rel(rel_uri), o, weight)
                n += 1
                if max_edges and n >= max_edges:
                    break
        return idx

    @classmethod
    def from_cskg_tsv(cls, path: str, base: "KGIndex | None" = None) -> "KGIndex":
        """Merge CSKG edges (node1, relation, node2, ... , weight) into an index."""
        idx = base or cls()
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                s = (row.get("node1;label") or row.get("node1") or "").lower()
                o = (row.get("node2;label") or row.get("node2") or "").lower()
                rel = (row.get("relation;label") or row.get("relation") or "related").split("/")[-1]
                if not s or not o:
                    continue
                w = float(row.get("weight", 1.0) or 1.0)
                idx.add(s, rel, o, w)
        return idx

    # -------------------------------------------------------------- queries

    def neighbours(self, term: str) -> list[tuple[str, str, float]]:
        return self.edges.get(term.lower(), [])

    def retrieve(self, seeds: list[str], hops: int = 2, top_p: int = 50
                 ) -> list[tuple[str, str, str]]:
        """Return up to `top_p` ranked, deduplicated triplets for seed entities.

        Gathers 1-hop and (if hops>=2) 2-hop neighbours, ranks by edge weight,
        deduplicates on (subject, relation, object).
        """
        scored: dict[tuple[str, str, str], float] = {}
        frontier = [(s.lower(), 1.0) for s in seeds]
        seen_terms = set(s.lower() for s in seeds)

        for hop in range(hops):
            next_frontier = []
            for term, decay in frontier:
                for rel, obj, w in self.neighbours(term):
                    triplet = (term, rel, obj)
                    score = w * decay
                    if triplet not in scored or scored[triplet] < score:
                        scored[triplet] = score
                    if hop + 1 < hops and obj not in seen_terms:
                        seen_terms.add(obj)
                        next_frontier.append((obj, decay * 0.5))
            frontier = next_frontier
            if not frontier:
                break

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return [t for t, _ in ranked[:top_p]]
