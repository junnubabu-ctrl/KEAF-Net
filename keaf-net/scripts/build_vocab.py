"""
Build the answer vocabulary from training annotations.

Usage:
    python -m scripts.build_vocab \
        --records data/okvqa/train.json \
        --out     data/okvqa/answer_vocab.json \
        --min-occurrence 8
"""
from __future__ import annotations

import argparse
import json

from keafnet.preprocessing.answer_vocab import AnswerVocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="parsed train records json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-occurrence", type=int, default=8)
    args = ap.parse_args()

    with open(args.records) as f:
        records = json.load(f)
    vocab = AnswerVocab.build((r["answers"] for r in records),
                              min_occurrence=args.min_occurrence)
    vocab.save(args.out)
    print(f"[vocab] {len(vocab)} answers -> {args.out}")


if __name__ == "__main__":
    main()
