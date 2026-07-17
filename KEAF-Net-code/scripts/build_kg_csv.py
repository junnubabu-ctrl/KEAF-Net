"""Merge ConceptNet 5.5 and CSKG into the retriever's edge-list format.

Produces a plain ``subject,relation,object,weight`` CSV readable by
:meth:`keaf_net.knowledge.KnowledgeIndex.from_csv` (Section 3.3 of the paper).
Only English concepts are kept, URI prefixes are stripped, and duplicate
(subject, relation, object) triples keep their highest weight.

    python scripts/build_kg_csv.py \
        --conceptnet data/downloads/conceptnet-assertions-5.5.5.csv.gz \
        --cskg data/downloads/cskg.tsv.gz \
        --out data/knowledge/conceptnet_cskg.csv

Either source may be omitted to build a single-source index (the knowledge
ablation in Table 9). Streaming + a seen-set keeps memory around a few GB for
the full merge.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import gzip
import json
import os


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def _norm(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def _concept_surface(uri: str) -> str | None:
    """'/c/en/green_curry/n' -> 'green_curry'; non-English concepts -> None."""
    parts = uri.split("/")
    if len(parts) < 4 or parts[1] != "c" or parts[2] != "en":
        return None
    return parts[3]


def iter_conceptnet(path: str):
    with _open(path) as fh:
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            _, rel_uri, start, end, meta = cols[:5]
            subj, obj = _concept_surface(start), _concept_surface(end)
            if not subj or not obj:
                continue
            relation = rel_uri.rsplit("/", 1)[-1]
            try:
                weight = float(json.loads(meta).get("weight", 1.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                weight = 1.0
            yield subj, relation, obj, weight


def iter_cskg(path: str):
    with _open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}

        def col(row, *names):
            for n in names:
                i = idx.get(n)
                if i is not None and i < len(row) and row[i]:
                    return row[i]
            return ""

        for line in fh:
            row = line.rstrip("\n").split("\t")
            subj = _norm(col(row, "node1;label", "node1"))
            obj = _norm(col(row, "node2;label", "node2"))
            relation = col(row, "relation;label", "relation").strip() or "RelatedTo"
            if not subj or not obj:
                continue
            yield subj, relation.replace(" ", ""), obj, 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged knowledge edge list")
    parser.add_argument("--conceptnet", help="conceptnet-assertions-*.csv[.gz]")
    parser.add_argument("--cskg", help="cskg.tsv[.gz]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-weight", type=float, default=0.0)
    args = parser.parse_args()
    if not args.conceptnet and not args.cskg:
        parser.error("provide --conceptnet and/or --cskg")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best: dict[tuple[str, str, str], float] = {}
    sources = []
    if args.conceptnet:
        sources.append(("ConceptNet", iter_conceptnet(args.conceptnet)))
    if args.cskg:
        sources.append(("CSKG", iter_cskg(args.cskg)))

    for name, edges in sources:
        n = 0
        for subj, relation, obj, weight in edges:
            if weight < args.min_weight or subj == obj:
                continue
            key = (subj, relation, obj)
            if weight > best.get(key, 0.0):
                best[key] = weight
            n += 1
        print(f"{name}: {n} edges read")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for (subj, relation, obj), weight in best.items():
            writer.writerow([subj, relation, obj, f"{weight:g}"])
    print(f"wrote {len(best)} unique triples -> {args.out}")


if __name__ == "__main__":
    main()
