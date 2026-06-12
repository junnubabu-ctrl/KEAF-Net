# Data Preparation

KEAF-Net consumes **pre-extracted features** so training is fast and the heavy
detectors/encoders run only once, offline. This document describes the expected
layout for OK-VQA and A-OKVQA.

## 1. Download the benchmarks

- **OK-VQA** (v1.1): questions + annotations on COCO `val2014` images. The
  5,046-question validation split is used as the standard test set.
- **A-OKVQA**: train / val / test with direct-answer (DA) and multiple-choice
  settings. KEAF-Net uses the direct-answer setting.

## 2. Extract features (offline)

| Feature | Tool | Output shape | Notes |
|---------|------|--------------|-------|
| Region features | Faster R-CNN (Visual Genome, 1600 classes) | `(M=36, 2048)` | bottom-up attention features |
| Region boxes | same detector | `(M=36, 4)` | `(x1,y1,x2,y2)` normalised |
| Grid features (optional) | ViT-B/16 | `(G, 768)` | if precomputed; else pass raw images |
| Triplet embeddings | Sentence-BERT `all-MiniLM-L6-v2` | `(P=50, 384)` | one per retrieved triplet |
| Question tokens | BERT-base tokenizer | `(L,)` ids + mask | |
| Answer soft scores | VQA scoring | `(num_answers,)` | `min(1, count/3)` per answer |

## 3. Knowledge retrieval

For each example, gather candidate triplets:

1. Visual entities = top-10 Faster R-CNN labels with confidence > 0.3.
2. Textual entities = noun phrases from spaCy chunking of the question.
3. Query a merged **ConceptNet 5.5 + CSKG** index; keep 1-hop and 2-hop
   neighbours, rank by edge weight, deduplicate on `(subject, relation, object)`,
   truncate to the top `P = 50`.
4. Embed each triplet string `"s r o"` with Sentence-BERT.

## 4. On-disk layout

```
data/okvqa/
├── train.json          # [{"id": "...", "answers": ["...", ...]}, ...]
├── val.json
└── features/
    ├── <id>.pt         # dict of tensors (see keys below)
    └── ...
```

Each `<id>.pt` is a dict with keys:
`region_feats, boxes, triplet_feats, triplet_mask, input_ids,
attention_mask, answer_scores`.

The loader `keafnet.data.KnowledgeVQADataset` reads exactly this layout.

## 5. Sanity check without real data

`keafnet.data.SyntheticVQADataset` emits the same keys with random tensors so
you can validate the pipeline before investing in feature extraction:

```bash
python -m scripts.train --synthetic --epochs 1 --cpu
```
