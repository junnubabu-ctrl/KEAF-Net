#!/usr/bin/env bash
# Download the benchmark datasets and knowledge graphs from their official
# sources into the layout expected by the KEAF-Net pipeline (see DATA.md).
#
# Usage:
#   bash scripts/download_data.sh okvqa aokvqa coco-val   # pick what you need
#   bash scripts/download_data.sh --all                   # everything (~60 GB)
#
# Downloads are resumable (wget -c). Each dataset remains under the licence of
# its original authors; this script only automates the official links.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
DL="$DATA_ROOT/downloads"
mkdir -p "$DL"

fetch() {  # fetch <url> [output-name]
    local url="$1" out="${2:-$(basename "$1")}"
    echo ">> $out"
    wget -c -q --show-progress -O "$DL/$out" "$url"
}

unzip_to() {  # unzip_to <zip> <dest>
    mkdir -p "$2"
    unzip -oq "$DL/$1" -d "$2"
}

okvqa() {
    echo "== OK-VQA annotations (~50 MB; images come from COCO train2014/val2014) =="
    local base="https://okvqa.allenai.org/static/data"
    fetch "$base/OpenEnded_mscoco_train2014_questions.json.zip"
    fetch "$base/mscoco_train2014_annotations.json.zip"
    fetch "$base/OpenEnded_mscoco_val2014_questions.json.zip"
    fetch "$base/mscoco_val2014_annotations.json.zip"
    for z in OpenEnded_mscoco_train2014_questions mscoco_train2014_annotations \
             OpenEnded_mscoco_val2014_questions mscoco_val2014_annotations; do
        unzip_to "$z.json.zip" "$DATA_ROOT/okvqa/raw"
    done
}

aokvqa() {
    echo "== A-OKVQA annotations (~50 MB; images come from COCO 2017) =="
    fetch "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"
    mkdir -p "$DATA_ROOT/aokvqa/raw"
    tar -xzf "$DL/aokvqa_v1p0.tar.gz" -C "$DATA_ROOT/aokvqa/raw"
}

vqav2() {
    echo "== VQA v2.0 annotations (~250 MB; images come from COCO 2014) =="
    local base="https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa"
    fetch "$base/v2_Questions_Train_mscoco.zip"
    fetch "$base/v2_Questions_Val_mscoco.zip"
    fetch "$base/v2_Annotations_Train_mscoco.zip"
    fetch "$base/v2_Annotations_Val_mscoco.zip"
    for z in v2_Questions_Train_mscoco v2_Questions_Val_mscoco \
             v2_Annotations_Train_mscoco v2_Annotations_Val_mscoco; do
        unzip_to "$z.zip" "$DATA_ROOT/vqav2/raw"
    done
}

coco_train() {
    echo "== COCO train2014 images (~13 GB) =="
    fetch "http://images.cocodataset.org/zips/train2014.zip"
    unzip_to "train2014.zip" "$DATA_ROOT/coco"
}

coco_val() {
    echo "== COCO val2014 images (~6 GB; needed for the OK-VQA evaluation split) =="
    fetch "http://images.cocodataset.org/zips/val2014.zip"
    unzip_to "val2014.zip" "$DATA_ROOT/coco"
}

coco2017() {
    echo "== COCO 2017 images for A-OKVQA (~19 GB train + 1 GB val) =="
    fetch "http://images.cocodataset.org/zips/train2017.zip"
    fetch "http://images.cocodataset.org/zips/val2017.zip"
    unzip_to "train2017.zip" "$DATA_ROOT/coco"
    unzip_to "val2017.zip" "$DATA_ROOT/coco"
}

gqa() {
    echo "== GQA (~1.5 GB questions + ~21 GB images) =="
    fetch "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
    fetch "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" "gqa_images.zip"
    unzip_to "questions1.2.zip" "$DATA_ROOT/gqa/raw"
    unzip_to "gqa_images.zip" "$DATA_ROOT/gqa"
}

textvqa() {
    echo "== TextVQA (~7 GB with images) =="
    fetch "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json"
    fetch "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"
    fetch "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
    mkdir -p "$DATA_ROOT/textvqa/raw"
    cp "$DL"/TextVQA_0.5.1_{train,val}.json "$DATA_ROOT/textvqa/raw/"
    unzip_to "train_val_images.zip" "$DATA_ROOT/textvqa"
}

knowledge() {
    echo "== ConceptNet 5.5 assertions (~350 MB) and CSKG (~1 GB) =="
    fetch "https://s3.amazonaws.com/conceptnet/downloads/2017/edges/conceptnet-assertions-5.5.5.csv.gz"
    # CSKG (Ilievski et al.): hosted on Zenodo; if the link moves, see
    # https://github.com/usc-isi-i2/cskg for the current download location.
    fetch "https://zenodo.org/records/4331372/files/cskg.tsv.gz" || \
        echo "!! CSKG download failed - fetch cskg.tsv.gz manually (see DATA.md)"
    mkdir -p "$DATA_ROOT/knowledge"
    echo "Now build the merged index:"
    echo "  python scripts/build_kg_csv.py \\"
    echo "      --conceptnet $DL/conceptnet-assertions-5.5.5.csv.gz \\"
    echo "      --cskg $DL/cskg.tsv.gz \\"
    echo "      --out $DATA_ROOT/knowledge/conceptnet_cskg.csv"
}

if [ $# -eq 0 ]; then
    sed -n '2,10p' "$0"
    echo "targets: okvqa aokvqa vqav2 coco-train coco-val coco2017 gqa textvqa knowledge --all"
    exit 0
fi

for target in "$@"; do
    case "$target" in
        okvqa) okvqa ;;
        aokvqa) aokvqa ;;
        vqav2) vqav2 ;;
        coco-train) coco_train ;;
        coco-val) coco_val ;;
        coco2017) coco2017 ;;
        gqa) gqa ;;
        textvqa) textvqa ;;
        knowledge) knowledge ;;
        --all) okvqa; aokvqa; vqav2; coco_train; coco_val; coco2017; gqa; textvqa; knowledge ;;
        *) echo "unknown target: $target"; exit 1 ;;
    esac
done
echo "Done. Next: python scripts/prepare_annotations.py (see DATA.md)."
