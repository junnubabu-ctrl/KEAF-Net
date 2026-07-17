# KEAF-Net

[![tests](https://github.com/junnubabu-ctrl/KEAF-Net/actions/workflows/ci.yml/badge.svg)](https://github.com/junnubabu-ctrl/KEAF-Net/actions/workflows/ci.yml)

**Knowledge-Enhanced Adaptive Fusion Network for Visual Question Answering with Multi-Hop Graph Reasoning**

Noorbhasha Junnu Babu, S. P. Rajamohana — Department of Computer Science, Pondicherry University (Karaikal Campus)

This repository is the reference implementation of KEAF-Net, a compact
knowledge-enhanced VQA model that treats knowledge selection, multimodal fusion
and reasoning depth as one connected problem. Retrieved knowledge triplets are
uneven in quality, so KEAF-Net filters them before they ever reach the reasoner,
fuses the survivors with visual regions and question tokens in a single typed
graph, and answers through a short chain of attention hops.

The model is built from three modules:

- **AKF — Adaptive Knowledge Filter** (`keaf_net/models/akf.py`). Scores each
  retrieved triplet against the image-question context and keeps the ones above
  a learned threshold. It is trained with sampled **leave-one-out** prediction-loss
  differences rather than answer-overlap labels, so the filter never learns an
  answer-string shortcut.
- **HGAF — Heterogeneous Graph Adaptive Fusion** (`keaf_net/models/hgaf.py`). A
  type-aware multi-head GAT over visual / textual / knowledge nodes with five
  relation categories and a gated fusion read-out.
- **MHSR — Multi-Hop Semantic Reasoning** (`keaf_net/models/mhsr.py`). Refines
  the question state over `T = 3` GRU attention hops on the fused graph.

```
A_hat = Classifier( MHSR( HGAF( V, T, K_hat ) ) )
```

## Results

Accuracy (%), mean ± std over three seeds `{42, 123, 2024}`.

| Dataset            | Split                     | KEAF-Net      |
| ------------------ | ------------------------- | ------------- |
| OK-VQA             | 5,046-question eval       | **68.7 ± 0.2** |
| A-OKVQA            | direct-answer test        | **62.4 ± 0.3** |
| VQA v2.0           | val                       | 72.86 ± 0.2   |
| GQA                | test-dev                  | 65.8 ± 0.3    |
| TextVQA            | val (no OCR pathway)      | 48.2 ± 0.4    |

In the configuration reported in the paper (Table 5), the full system has ~252M
inference parameters (~184M trainable), of which the ViT-B/16 and BERT-base
backbones account for the majority; the AKF/HGAF/MHSR reasoning head on top of
the cached features is the part trained here. Ablations (Section 6) show all
three modules contribute, with the largest single gain from HGAF and a positive
interaction between AKF and MHSR.

## Installation

```bash
git clone https://github.com/junnubabu-ctrl/KEAF-Net.git
cd KEAF-Net
pip install -e .                 # core (model + training + evaluation)
pip install -e ".[extract]"      # + offline feature extraction / retrieval
```

Tested with Python 3.10–3.11 and PyTorch ≥ 2.0. Training used 2× NVIDIA RTX 3090.

## Data preparation

All datasets download from their official sources with the bundled script —
see **[DATA.md](DATA.md)** for sources, sizes, licences and the full pipeline:

```bash
bash scripts/download_data.sh okvqa coco-train coco-val knowledge
python scripts/prepare_annotations.py okvqa ...   # official files -> loader schema
python scripts/build_kg_csv.py ...                # merged ConceptNet+CSKG index
```

Features are pre-extracted once so that training focuses on filtering, fusion and
reasoning (Section 3). The pipeline expects the following layout:

```
data/<dataset>/{train,val,test}.json        # [{qid, image_id, image_file, question, answers}]
features/<split>/grid/<image_id>.pt         # ViT-B/16 patches + patch boxes
features/<split>/region/<image_id>.pt       # Faster R-CNN (VG) region features + boxes
features/<split>/question/<qid>.pt          # BERT token states + [CLS]
knowledge/cache/<split>/<qid>.pt            # Sentence-BERT triplet embeddings + link pairs
```

1. **Answer vocabulary** (3,129 most frequent normalised answers):
   ```bash
   python scripts/build_vocab.py --annotations data/okvqa/train.json \
       --num-answers 3129 --out data/okvqa/answer_vocab.json
   ```
2. **Grid + question features** (ViT-B/16, BERT-base). Region features come from
   a standard bottom-up-attention dump placed under `features/<split>/region/`:
   ```bash
   python scripts/extract_features.py --config configs/okvqa.yaml \
       --split train --images data/coco/train2014
   ```
3. **Knowledge retrieval** over merged ConceptNet 5.5 + CSKG (top-50 triplets,
   one/two-hop, Sentence-BERT embeddings, entity linking):
   ```bash
   python scripts/retrieve_knowledge.py --config configs/okvqa.yaml --split train \
       --kg-csv data/knowledge/conceptnet_cskg.csv --labels data/okvqa/region_labels.json
   ```

The benchmark datasets and knowledge graphs are released by their original
authors under their own licences and are **not** redistributed here.

## Training and evaluation

```bash
# train the three reported seeds
for s in 42 123 2024; do
  python scripts/train.py --config configs/okvqa.yaml --seed $s
done

# evaluate a checkpoint and dump predictions
python scripts/evaluate.py --config configs/okvqa.yaml \
    --checkpoint outputs/okvqa/keaf_net_okvqa_seed42_best.pt --split val
```

Key hyperparameters (Table 4 of the paper) live in `configs/*.yaml`; the
defaults in `keaf_net/config.py` reproduce the reported configuration. Useful
ablation knobs: `model.num_hops` (T, Table 10), `optim.akf_loss_weight` (w_AKF),
`model.max_triplets` (P) and `model.akf_loss_weight = 0` to disable the filter.

## Repository layout

```
keaf_net/
  models/        akf.py  hgaf.py  mhsr.py  keaf_net.py  encoders.py  graph.py
  knowledge/     retriever.py        # ConceptNet+CSKG retrieval, entity linking
  data/          dataset.py  vocab.py  synthetic.py
  engine/        trainer.py  evaluator.py  loss.py
  utils/         scatter.py  metrics.py  text.py  seed.py
  config.py  structures.py
configs/         okvqa.yaml  aokvqa.yaml  vqav2.yaml
scripts/         download_data  prepare_annotations  build_kg_csv  build_vocab
                 extract_features  retrieve_knowledge  train  evaluate
tests/           unit tests (run with `pytest`)
```

## Tests

```bash
pytest
```

The tests build a synthetic batch with the same shapes as the real pipeline and
exercise the graph construction, the three modules, the full forward/backward
pass, the LOO training step and the evaluator — no datasets or pretrained
backbones required.

## Implementation notes

A few design choices worth flagging for anyone reading the code against the paper:

- **Cached features.** The detector (Faster R-CNN) region features and the
  Sentence-BERT triplet embeddings are pre-extracted offline, exactly as in
  Section 3.2. For efficiency this release also caches the ViT-B/16 grid and
  BERT-base question features; the encoders in `keaf_net/models/encoders.py`
  follow the paper's freezing scheme (ViT layers 1–6 frozen, BERT embeddings
  frozen) and are the ones used to produce those caches.
- **Visual nodes.** Following Algorithm 1 (`Vv := V`), the graph's visual nodes
  are `V = [V_grid; V_region]`. Grid patches tile the image, so each one carries
  a bounding box and shares the IoU rule with the region proposals.
- **Five relation categories.** HGAF realises them through the node-type-pair
  attention vector `a_{type(i),type(j)}` in Eq. (14) together with type-specific
  projections, so a visual–knowledge edge and a visual–visual edge are
  transformed differently (Proposition 2). Message passing is implemented with
  small scatter primitives (`keaf_net/utils/scatter.py`) to avoid a
  `torch_geometric` dependency.
- **Filter gate.** The keep/discard decision uses a straight-through estimator
  (Eq. 10); the leave-one-out targets (Eq. 11–13) are obtained by re-running only
  the reasoning path with one triplet removed, under `no_grad`, which keeps the
  overhead to `|S|` passes per batch.
- **Parameter counts.** Exact per-module parameter counts depend on projection
  shapes that the paper does not fully pin down, so they differ slightly from
  Table 5; the architecture and equations are reproduced as described.

## Citation

```bibtex
@article{junnubabu2026keafnet,
  title   = {KEAF-Net: Knowledge-Enhanced Adaptive Fusion Network for Visual
             Question Answering with Multi-Hop Graph Reasoning},
  author  = {Noorbhasha Junnu Babu and Rajamohana, S. P.},
  year    = {2026},
  note    = {Department of Computer Science, Pondicherry University.
             Manuscript under review}
}
```

## License

Released under the MIT License (see `LICENSE`). Datasets and knowledge graphs
remain under their respective original licences.

## Acknowledgements

We thank Pondicherry University and Madanapalle Institute of Technology and
Science for the computational infrastructure used in this work.
