"""Convert official dataset annotations into the loader's record format.

Every dataset is normalised to ``data/<dataset>/<split>.json``, a list of::

    {"qid": ..., "image_id": ..., "image_file": ..., "question": ..., "answers": [...]}

which is what :class:`keaf_net.data.VQADataset`, ``scripts/build_vocab.py`` and
``scripts/extract_features.py`` consume.

Examples (after ``bash scripts/download_data.sh okvqa aokvqa``)::

    python scripts/prepare_annotations.py okvqa \
        --questions data/okvqa/raw/OpenEnded_mscoco_train2014_questions.json \
        --annotations data/okvqa/raw/mscoco_train2014_annotations.json \
        --coco-split train2014 --out data/okvqa/train.json

    python scripts/prepare_annotations.py okvqa \
        --questions data/okvqa/raw/OpenEnded_mscoco_val2014_questions.json \
        --annotations data/okvqa/raw/mscoco_val2014_annotations.json \
        --coco-split val2014 --out data/okvqa/val.json

    python scripts/prepare_annotations.py aokvqa \
        --input data/aokvqa/raw/aokvqa_v1p0_train.json \
        --coco-split train2017 --out data/aokvqa/train.json

VQA v2 uses the same MSCOCO layout as OK-VQA::

    python scripts/prepare_annotations.py vqav2 \
        --questions data/vqav2/raw/v2_OpenEnded_mscoco_train2014_questions.json \
        --annotations data/vqav2/raw/v2_mscoco_train2014_annotations.json \
        --coco-split train2014 --out data/vqav2/train.json
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import os


def _dump(records: list[dict], out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False)
    print(f"wrote {len(records)} records -> {out}")


def _coco_file(split: str, image_id: int) -> str:
    # 2014 releases prefix the split name; 2017 releases are bare 12-digit ids
    if split.endswith("2014"):
        return f"COCO_{split}_{image_id:012d}.jpg"
    return f"{image_id:012d}.jpg"


def convert_mscoco(questions_path: str, annotations_path: str, coco_split: str, out: str) -> None:
    """OK-VQA and VQA v2 share the official MSCOCO question/annotation format."""
    with open(questions_path, encoding="utf-8") as fh:
        questions = {q["question_id"]: q for q in json.load(fh)["questions"]}
    with open(annotations_path, encoding="utf-8") as fh:
        annotations = json.load(fh)["annotations"]

    records = []
    for ann in annotations:
        q = questions[ann["question_id"]]
        records.append({
            "qid": ann["question_id"],
            "image_id": ann["image_id"],
            "image_file": _coco_file(coco_split, ann["image_id"]),
            "question": q["question"],
            "answers": [a["answer"] for a in ann["answers"]],
        })
    _dump(records, out)


def convert_aokvqa(input_path: str, coco_split: str, out: str) -> None:
    """A-OKVQA v1.0 records carry direct answers (empty on the hidden test split)."""
    with open(input_path, encoding="utf-8") as fh:
        entries = json.load(fh)

    records = []
    for e in entries:
        records.append({
            "qid": e["question_id"],
            "image_id": e["image_id"],
            "image_file": _coco_file(coco_split, e["image_id"]),
            "question": e["question"],
            "answers": e.get("direct_answers") or [],
        })
    _dump(records, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise official VQA annotations")
    sub = parser.add_subparsers(dest="dataset", required=True)

    for name in ("okvqa", "vqav2"):
        p = sub.add_parser(name)
        p.add_argument("--questions", required=True)
        p.add_argument("--annotations", required=True)
        p.add_argument("--coco-split", required=True, help="e.g. train2014 / val2014")
        p.add_argument("--out", required=True)

    p = sub.add_parser("aokvqa")
    p.add_argument("--input", required=True, help="aokvqa_v1p0_<split>.json")
    p.add_argument("--coco-split", required=True, help="e.g. train2017 / val2017")
    p.add_argument("--out", required=True)

    args = parser.parse_args()
    if args.dataset in ("okvqa", "vqav2"):
        convert_mscoco(args.questions, args.annotations, args.coco_split, args.out)
    else:
        convert_aokvqa(args.input, args.coco_split, args.out)


if __name__ == "__main__":
    main()
