#!/usr/bin/env bash
# End-to-end KEAF-Net pipeline: download -> parse -> vocab -> features -> train -> eval
# Edit the paths at the top, then run:  bash scripts/run_pipeline.sh
set -euo pipefail

# --------------------------------------------------------------------------- #
# 0. Configuration (edit these)
# --------------------------------------------------------------------------- #
DATA=${DATA:-data}
OKVQA=$DATA/okvqa
COCO=$DATA/coco
CONCEPTNET=$DATA/conceptnet/conceptnet-assertions-5.7.0.csv.gz
CSKG=$DATA/cskg/cskg.tsv                       # optional; comment out --cskg if absent
SPLIT=${SPLIT:-val}

mkdir -p "$OKVQA" "$COCO"

# --------------------------------------------------------------------------- #
# 1. Parse OK-VQA annotations into the common record format
#    (assumes you have downloaded the official OK-VQA json files)
# --------------------------------------------------------------------------- #
python - <<PY
from keafnet.preprocessing.parse_annotations import parse_okvqa, write_records
recs = parse_okvqa("$OKVQA/OpenEnded_mscoco_${SPLIT}2014_questions.json",
                   "$OKVQA/mscoco_${SPLIT}2014_annotations.json", split="$SPLIT")
write_records(recs, "$OKVQA/${SPLIT}.json")
PY

# --------------------------------------------------------------------------- #
# 2. Build the answer vocabulary (from the TRAIN split)
# --------------------------------------------------------------------------- #
if [ ! -f "$OKVQA/answer_vocab.json" ]; then
  python -m scripts.build_vocab --records "$OKVQA/train.json" \
      --out "$OKVQA/answer_vocab.json" --min-occurrence 8
fi

# --------------------------------------------------------------------------- #
# 3. Extract features (ViT + Faster R-CNN + knowledge retrieval + tokenise)
# --------------------------------------------------------------------------- #
python -m keafnet.preprocessing.extract_features \
    --records "$OKVQA/${SPLIT}.json" \
    --images  "$COCO/${SPLIT}2014" \
    --kg      "$CONCEPTNET" \
    --cskg    "$CSKG" \
    --vocab   "$OKVQA/answer_vocab.json" \
    --out     "$OKVQA/features"

# --------------------------------------------------------------------------- #
# 4. Train
# --------------------------------------------------------------------------- #
python -m scripts.train_full --config configs/keafnet_okvqa.yaml

# --------------------------------------------------------------------------- #
# 5. Evaluate the best checkpoint
# --------------------------------------------------------------------------- #
python -m scripts.evaluate --config configs/keafnet_okvqa.yaml \
    --ckpt checkpoints/okvqa/best.pt

echo "Pipeline complete."
