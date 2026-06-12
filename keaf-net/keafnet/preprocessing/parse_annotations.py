"""
Parsers for OK-VQA and A-OKVQA annotation files into a common record format.

Common record:
    {
        "id": str,                # unique example id
        "image_id": int,
        "image_file": str,        # COCO filename, e.g. COCO_val2014_000000000123.jpg
        "question": str,
        "answers": list[str],     # 10 annotator answers (OK-VQA) or DA answers
    }

Reference: KEAF-Net, Section 5 (Datasets).
"""
from __future__ import annotations

import json
import os


def _coco_filename(image_id: int, split: str) -> str:
    # OK-VQA images come from COCO train2014 / val2014.
    subset = "val2014" if "val" in split else "train2014"
    return f"COCO_{subset}_{image_id:012d}.jpg"


def parse_okvqa(questions_file: str, annotations_file: str, split: str = "val"
                ) -> list[dict]:
    """Parse OK-VQA v1.1 question + annotation JSON files."""
    with open(questions_file) as f:
        questions = {q["question_id"]: q for q in json.load(f)["questions"]}
    with open(annotations_file) as f:
        anns = json.load(f)["annotations"]

    records = []
    for a in anns:
        qid = a["question_id"]
        q = questions.get(qid)
        if q is None:
            continue
        records.append({
            "id": f"okvqa_{qid}",
            "image_id": a["image_id"],
            "image_file": _coco_filename(a["image_id"], split),
            "question": q["question"],
            "answers": [ans["answer"] for ans in a["answers"]],
        })
    return records


def parse_aokvqa(annotations_file: str, split: str = "val") -> list[dict]:
    """Parse A-OKVQA JSON (direct-answer setting).

    A-OKVQA stores a list of dicts each with `question_id`, `image_id`,
    `question`, `direct_answers` (10 strings), and `choices`/`correct_choice_idx`.
    """
    with open(annotations_file) as f:
        data = json.load(f)
    records = []
    for ex in data:
        records.append({
            "id": f"aokvqa_{ex['question_id']}",
            "image_id": ex["image_id"],
            "image_file": _coco_filename(ex["image_id"], split),
            "question": ex["question"],
            "answers": ex.get("direct_answers", []),
        })
    return records


def write_records(records: list[dict], out_file: str) -> None:
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(records, f)
    print(f"[parse] wrote {len(records)} records -> {out_file}")
