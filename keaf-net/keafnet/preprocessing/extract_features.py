"""
Offline feature extraction: turn parsed records into per-example `.pt` files.

For every record this script:
  1. loads the image, runs the ViT grid + Faster R-CNN region extractors,
  2. retrieves knowledge triplets (ConceptNet+CSKG) from visual labels + question,
     and embeds them with Sentence-BERT,
  3. tokenises the question with the BERT tokenizer,
  4. encodes the answer soft scores against the answer vocabulary,
  5. saves a dict of tensors to `<features_dir>/<id>.pt`.

The output matches exactly what `keafnet.data.KnowledgeVQADataset` expects.

Usage:
    python -m keafnet.preprocessing.extract_features \
        --records data/okvqa/val.json \
        --images  data/coco/val2014 \
        --kg      data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
        --cskg    data/cskg/cskg.tsv \
        --vocab   data/okvqa/answer_vocab.json \
        --out     data/okvqa/features
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .answer_vocab import AnswerVocab
from .visual_features import GridExtractor, RegionExtractor
from ..retrieval.kg_index import KGIndex
from ..retrieval.retriever import KnowledgeRetriever, TripletEmbedder

try:
    from transformers import BertTokenizerFast
    _HAS_HF = True
except Exception:  # pragma: no cover
    _HAS_HF = False

try:
    from PIL import Image
    import torchvision.transforms as T
    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    return tf(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--images", required=True, help="image folder")
    ap.add_argument("--kg", required=True, help="ConceptNet assertions csv(.gz)")
    ap.add_argument("--cskg", default=None, help="optional CSKG tsv")
    ap.add_argument("--vocab", required=True, help="answer vocab json")
    ap.add_argument("--out", required=True, help="output features dir")
    ap.add_argument("--max-triplets", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--kg-max-edges", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="process first N")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.records) as f:
        records = json.load(f)
    if args.limit:
        records = records[: args.limit]

    print("[extract] loading knowledge graph ...")
    kg = KGIndex.from_conceptnet_csv(args.kg, max_edges=args.kg_max_edges)
    if args.cskg:
        kg = KGIndex.from_cskg_tsv(args.cskg, base=kg)
    retriever = KnowledgeRetriever(kg, TripletEmbedder(),
                                   top_p=args.max_triplets, hops=2)

    region_ex = RegionExtractor(num_regions=36)
    grid_ex = GridExtractor()
    vocab = AnswerVocab.load(args.vocab)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased") if _HAS_HF else None

    for i, rec in enumerate(records):
        img_path = os.path.join(args.images, rec["image_file"])
        image = load_image(img_path)

        region_feats, boxes, labels = region_ex(image)
        grid_feats = grid_ex(image)
        _, triplet_feats, triplet_mask = retriever.retrieve(labels, rec["question"])

        if tok is not None:
            enc = tok(rec["question"], padding="max_length", truncation=True,
                      max_length=args.max_len, return_tensors="pt")
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]
        else:  # pragma: no cover
            input_ids = torch.zeros(args.max_len, dtype=torch.long)
            attention_mask = torch.ones(args.max_len, dtype=torch.long)

        answer_scores = vocab.encode_scores(rec["answers"])

        torch.save({
            "region_feats": torch.from_numpy(region_feats),
            "boxes": torch.from_numpy(boxes),
            "grid_feats": torch.from_numpy(grid_feats),
            "triplet_feats": torch.from_numpy(triplet_feats),
            "triplet_mask": torch.from_numpy(triplet_mask),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_scores": answer_scores,
        }, os.path.join(args.out, f"{rec['id']}.pt"))

        if (i + 1) % 100 == 0:
            print(f"[extract] {i+1}/{len(records)}")

    print(f"[extract] done -> {args.out}")


if __name__ == "__main__":
    main()
