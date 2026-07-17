"""Retrieve, embed and cache external knowledge per question (Section 3.3).

For each question we collect seed entities from the detector labels and the
question noun phrases, expand one/two-hop neighbours over the merged
ConceptNet + CSKG index, cap at ``P`` triplets, embed them with Sentence-BERT
and pre-compute the visual/textual entity-link index pairs. The result is the
per-question cache read by :class:`keaf_net.data.VQADataset`.

    python scripts/retrieve_knowledge.py --config configs/okvqa.yaml --split train \
        --kg-csv data/knowledge/conceptnet_cskg.csv --labels data/okvqa/region_labels.json

Requires the ``extract`` extra (Sentence-BERT, spaCy, NLTK).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import os

import torch
from tqdm import tqdm

from keaf_net.config import load_config
from keaf_net.knowledge import EntityLinker, KnowledgeIndex, KnowledgeRetriever


def _noun_phrases(nlp, question: str) -> list[str]:
    return [chunk.text for chunk in nlp(question).noun_chunks]


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve and cache knowledge triplets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--kg-csv", required=True, help="merged ConceptNet+CSKG edge list")
    parser.add_argument("--labels", required=True, help="json mapping image_id -> detector labels")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import spacy
    from sentence_transformers import SentenceTransformer

    cfg = load_config(args.config)
    out_dir = os.path.join(cfg.data.knowledge_cache, args.split)
    os.makedirs(out_dir, exist_ok=True)

    index = KnowledgeIndex.from_csv(args.kg_csv)
    retriever = KnowledgeRetriever(index, max_triplets=cfg.model.max_triplets)
    linker = EntityLinker(use_wordnet=True)
    nlp = spacy.load("en_core_web_sm")
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=args.device)

    with open(args.labels, "r", encoding="utf-8") as fh:
        region_labels = json.load(fh)
    with open(os.path.join(cfg.data.data_root, cfg.data.dataset, f"{args.split}.json")) as fh:
        records = json.load(fh)

    for rec in tqdm(records, desc=f"retrieve {args.split}"):
        labels = region_labels.get(str(rec["image_id"]), [])[:10]    # top-10 detector labels
        phrases = _noun_phrases(nlp, rec["question"])
        triplets = retriever.retrieve(labels + phrases)

        embeddings = sbert.encode([t.linearise() for t in triplets], convert_to_tensor=True).cpu()
        vk = linker.link(labels, triplets)                            # visual-knowledge links
        tk = linker.link(phrases, triplets)                           # textual-knowledge links
        torch.save({
            "triplets": embeddings,
            "vk_pairs": torch.tensor(vk, dtype=torch.long).reshape(-1, 2),
            "tk_pairs": torch.tensor(tk, dtype=torch.long).reshape(-1, 2),
            "strings": [t.linearise() for t in triplets],
        }, os.path.join(out_dir, f"{rec['qid']}.pt"))

    print(f"cached knowledge for {len(records)} questions -> {out_dir}")


if __name__ == "__main__":
    main()
