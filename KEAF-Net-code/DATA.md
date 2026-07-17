# Datasets and knowledge sources

All benchmarks and knowledge graphs are released by their original authors under
their own licences and are therefore **downloaded from the official sources**
rather than redistributed with this repository. `scripts/download_data.sh`
automates every link below; sizes are approximate.

| Resource | Official source | Size | Used for |
| --- | --- | --- | --- |
| OK-VQA v1.1 | https://okvqa.allenai.org | 50 MB + COCO 2014 images | Main benchmark (5,046-question eval split) |
| A-OKVQA v1.0 | https://allenai.org/project/a-okvqa | 50 MB + COCO 2017 images | Direct-answer test split |
| VQA v2.0 | https://visualqa.org/download.html | 250 MB + COCO 2014 images | General VQA |
| COCO images | https://cocodataset.org | 13 GB (train2014) + 6 GB (val2014) + 20 GB (2017) | Images for the three datasets above |
| GQA | https://cs.stanford.edu/people/dorarad/gqa | 1.5 GB + 21 GB images | Compositional reasoning |
| TextVQA 0.5.1 | https://textvqa.org | 7 GB | OCR-dependent questions (no OCR branch in this release) |
| ConceptNet 5.5 | https://conceptnet.io (S3 assertions dump) | 350 MB | Knowledge retrieval |
| CSKG | https://github.com/usc-isi-i2/cskg (Zenodo) | 1 GB | Knowledge retrieval |

## Preparation pipeline

```bash
# 1. download what you need (resumable)
bash scripts/download_data.sh okvqa coco-train coco-val knowledge

# 2. normalise the official annotation formats to the loader's schema
python scripts/prepare_annotations.py okvqa \
    --questions data/okvqa/raw/OpenEnded_mscoco_train2014_questions.json \
    --annotations data/okvqa/raw/mscoco_train2014_annotations.json \
    --coco-split train2014 --out data/okvqa/train.json
python scripts/prepare_annotations.py okvqa \
    --questions data/okvqa/raw/OpenEnded_mscoco_val2014_questions.json \
    --annotations data/okvqa/raw/mscoco_val2014_annotations.json \
    --coco-split val2014 --out data/okvqa/val.json

# 3. build the merged knowledge index (subject,relation,object,weight CSV)
python scripts/build_kg_csv.py \
    --conceptnet data/downloads/conceptnet-assertions-5.5.5.csv.gz \
    --cskg data/downloads/cskg.tsv.gz \
    --out data/knowledge/conceptnet_cskg.csv

# 4. answer vocabulary, features and per-question knowledge cache
python scripts/build_vocab.py --annotations data/okvqa/train.json \
    --num-answers 3129 --out data/okvqa/answer_vocab.json
python scripts/extract_features.py --config configs/okvqa.yaml \
    --split train --images data/coco/train2014
python scripts/retrieve_knowledge.py --config configs/okvqa.yaml --split train \
    --kg-csv data/knowledge/conceptnet_cskg.csv \
    --labels data/okvqa/region_labels.json
```

The loader's record schema produced by step 2 is::

    {"qid", "image_id", "image_file", "question", "answers"}

## Region features and detector labels

The Faster R-CNN region features (36 proposals per image, 2048-d) follow the
standard bottom-up-attention format and are pre-extracted offline as in the
paper; place the per-image dumps under `features/<split>/region/<image_id>.pt`
as `{feat: [36, 2048], boxes: [36, 4]}`. `data/<dataset>/region_labels.json`
maps each `image_id` to the detector's top class labels and is produced by the
same detector pass; the retrieval script uses the top-10 labels above the 0.3
confidence threshold (Section 3.3).

## Licence notes

COCO images are under the Flickr terms of use; VQA/OK-VQA/A-OKVQA annotations
are CC-BY; GQA is CC-BY 4.0; TextVQA is CC-BY 4.0; ConceptNet is CC-BY-SA 4.0;
CSKG inherits the licences of its source graphs. Check each project page before
any use beyond research.
