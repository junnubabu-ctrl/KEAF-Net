"""Build the answer vocabulary from training annotations (Section 5.4).

    python scripts/build_vocab.py --annotations data/okvqa/train.json \
        --num-answers 3129 --out data/answer_vocab.json

``annotations`` is a JSON list of records with an ``answers`` field (the list of
human answers per question).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

from keaf_net.data import AnswerVocab


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KEAF-Net answer vocabulary")
    parser.add_argument("--annotations", nargs="+", required=True)
    parser.add_argument("--num-answers", type=int, default=3129)
    parser.add_argument("--min-count", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    answer_lists = []
    for path in args.annotations:
        with open(path, "r", encoding="utf-8") as fh:
            answer_lists.extend(rec.get("answers", []) for rec in json.load(fh))

    vocab = AnswerVocab.build(answer_lists, num_answers=args.num_answers, min_count=args.min_count)
    vocab.save(args.out)
    print(f"built vocabulary of {len(vocab)} answers from {len(answer_lists)} questions -> {args.out}")


if __name__ == "__main__":
    main()
