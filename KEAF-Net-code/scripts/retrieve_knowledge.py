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
from keaf_net.data.grounding import (phrase_pairs_to_tokens, question_sha256,
                                     region_pairs_to_visual)


def _noun_phrases(nlp, question: str) -> list[tuple[str, int, int]]:
    return [(chunk.text, chunk.start_char, chunk.end_char) for chunk in nlp(question).noun_chunks]


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
        feature_dir = os.path.join(cfg.data.feature_root, args.split)
        grid = torch.load(os.path.join(feature_dir, "grid", f"{rec['image_id']}.pt"))
        region = torch.load(os.path.join(feature_dir, "region", f"{rec['image_id']}.pt"))
        question = torch.load(os.path.join(feature_dir, "question", f"{rec['qid']}.pt"))
        if question.get("question_sha256") != question_sha256(rec["question"]) or "offset_mapping" not in question:
            raise ValueError("question cache requires matching text and BERT offsets; rebuild features")
        if question["offset_mapping"].shape != (question["tokens"].size(0), 2):
            raise ValueError("cached BERT offsets must align with the token feature rows")
        raw_labels = region_labels.get(str(rec["image_id"]), [])
        # A list follows detector row order. An explicit object allows selected
        # labels to retain their original region row IDs.
        if isinstance(raw_labels, dict):
            labels = raw_labels["labels"][:10]
            region_ids = raw_labels["region_indices"][:10]
        else:
            labels = raw_labels[:10]
            region_ids = list(range(len(labels)))
        if len(labels) != len(region_ids):
            raise ValueError("detector labels and region_indices must have equal lengths")
        chunks = _noun_phrases(nlp, rec["question"])
        phrases = [text for text, _, _ in chunks]
        triplets = retriever.retrieve(labels + phrases)

        embeddings = (sbert.encode([t.linearise() for t in triplets], convert_to_tensor=True).cpu()
                      if triplets else torch.empty(0, cfg.model.knowledge_feat_dim))
        vk = region_pairs_to_visual(
            [(region_ids[label], knowledge) for label, knowledge in linker.link(labels, triplets)],
            grid["feat"].size(0), region["feat"].size(0),
        )
        tk = phrase_pairs_to_tokens(linker.link(phrases, triplets),
                                    [(start, end) for _, start, end in chunks], question["offset_mapping"])
        torch.save({
            "triplets": embeddings,
            "vk_pairs": vk, "tk_pairs": tk,
            "vk_index_space": "visual_grid_then_region", "tk_index_space": "bert_token",
            "alignment": {"version": 2, "num_grid": grid["feat"].size(0),
                          "num_regions": region["feat"].size(0), "num_tokens": question["tokens"].size(0),
                          "question_sha256": question_sha256(rec["question"]),
                          "tokenizer_name": question.get("tokenizer_name")},
            "strings": [t.linearise() for t in triplets],
        }, os.path.join(out_dir, f"{rec['qid']}.pt"))

    print(f"cached knowledge for {len(records)} questions -> {out_dir}")


if __name__ == "__main__":
    main()
