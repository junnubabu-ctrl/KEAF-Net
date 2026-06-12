# KEAF-Net: Knowledge-Enhanced Adaptive Fusion Network for Visual Question Answering

Official PyTorch implementation of **KEAF-Net**, a knowledge-intensive VQA
architecture with three novel modules: an **Adaptive Knowledge Filter (AKF)**, a
**Heterogeneous Graph Adaptive Fusion (HGAF)** module, and a **Multi-Hop
Semantic Reasoning (MHSR)** module. KEAF-Net reaches the highest accuracy among
methods with sub-500M trainable parameters that do not invoke large language
models at inference.

> **Paper:** *KEAF-Net: Knowledge-Enhanced Adaptive Fusion Network for Visual
> Question Answering with Multi-Hop Graph Reasoning.*
> N. Junnu Babu and S. P. Rajamohana, Pondicherry University.

```
        Image I ──► ViT-B/16 ─┐
                  └ Faster R-CNN ─┐   V
        Question Q ─► BERT-base ──┼──► AKF ──► HGAF ──► MHSR ──► Classifier ──► Â
        ConceptNet+CSKG ─► triplets ┘  (filter) (graph)  (3 hops)
```

---

## Highlights

- **Answer-independent knowledge filtering.** The AKF is supervised by
  leave-one-out (LOO) prediction-loss differences, not answer-overlap labels,
  so it learns *which knowledge helps reasoning* without label leakage.
- **One unified heterogeneous graph.** HGAF places visual, textual, and
  knowledge nodes in a single typed graph with **five edge types** and
  type-aware multi-head attention (provably more expressive than 1-WL).
- **Multi-hop reasoning.** MHSR chains evidence over **T = 3** GRU-based hops.
- **Efficient.** 252M total / 184M trainable parameters; no LLM at inference.

## Results

Main results (accuracy %, mean ± std over 3 runs). See the paper for the full tables.

| Method | Params | OK-VQA | A-OKVQA (DA) |
|--------|:------:|:------:|:------------:|
| KRISP | 155M | 38.9 | — |
| KAT | 350M | 54.4 | — |
| REVIVE | 380M | 58.0 | — |
| Prophet* | 175B | 61.1 | 55.7 |
| PromptCap* | 175B | 60.4 | 59.6 |
| RK-VQA | ~3B | 64.1 | — |
| **KEAF-Net (ours)** | **252M** | **68.7 ± 0.2** | **62.4 ± 0.3** |

<sub>*Uses GPT-3 (175B) at inference. KEAF-Net is the strongest sub-500M, LLM-free model.</sub>

Module ablation (OK-VQA):

| Configuration | OK-VQA |
|---------------|:------:|
| Full KEAF-Net | **68.7** |
| w/o AKF | 65.5 (−3.2) |
| w/o HGAF | 64.9 (−3.8) |
| w/o MHSR | 66.8 (−1.9) |
| w/o AKF + MHSR | 62.8 (−5.9, super-additive) |

> The joint AKF+MHSR removal (−5.9) exceeds the sum of individual drops (−5.1),
> evidence of genuine inter-module synergy.

---

## Installation

```bash
git clone https://github.com/<your-username>/keaf-net.git
cd keaf-net
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # installs the `keafnet` package
```

Python ≥ 3.9 and PyTorch ≥ 2.0 are required. `transformers` and `timm` are
optional; without them the encoders fall back to lightweight learnable stubs so
the pipeline still runs (useful for testing).

## Quick start (no data required)

The repository ships with a synthetic dataset that produces shape-correct
tensors, so you can exercise the entire training / eval / inference loop without
downloading any features:

```bash
# end-to-end smoke test of the full model (random data)
python -m scripts.train    --synthetic --epochs 1 --cpu
python -m scripts.evaluate --synthetic --cpu
python -m scripts.infer    --synthetic --cpu      # shows AKF gates + MHSR hops
python tests/test_pipeline.py                     # unit tests
```

## Training on OK-VQA / A-OKVQA

1. Prepare features following [`docs/DATA.md`](docs/DATA.md) (region features,
   triplet embeddings, tokenised questions, soft answer scores).
2. Edit the paths in `configs/keafnet_okvqa.yaml`.
3. Train and evaluate:

```bash
python -m scripts.train    --config configs/keafnet_okvqa.yaml
python -m scripts.evaluate --config configs/keafnet_okvqa.yaml --ckpt checkpoints/okvqa/best.pt
```

The default hyperparameters mirror the paper: AdamW (lr 1e-4, wd 0.01), cosine
schedule, batch size 64, 20 epochs, LOO sample size |S| = 10, λ = 0.3.

## Full end-to-end pipeline (real data)

Everything needed to go from raw OK-VQA downloads to a trained model is included:

```bash
# 1. Parse annotations -> common record format
python -c "from keafnet.preprocessing.parse_annotations import *; \
  write_records(parse_okvqa('q.json','a.json','val'), 'data/okvqa/val.json')"

# 2. Build the answer vocabulary (from train split)
python -m scripts.build_vocab --records data/okvqa/train.json \
    --out data/okvqa/answer_vocab.json --min-occurrence 8

# 3. Extract features: ViT + Faster R-CNN + ConceptNet/CSKG retrieval + Sentence-BERT
python -m keafnet.preprocessing.extract_features \
    --records data/okvqa/val.json --images data/coco/val2014 \
    --kg data/conceptnet/conceptnet-assertions-5.7.0.csv.gz \
    --cskg data/cskg/cskg.tsv --vocab data/okvqa/answer_vocab.json \
    --out data/okvqa/features

# 4. Train (production engine: AMP, DDP, resume, TensorBoard, vectorized LOO)
python -m scripts.train_full --config configs/keafnet_okvqa.yaml
# multi-GPU:
torchrun --nproc_per_node=2 -m scripts.train_full --config configs/keafnet_okvqa.yaml

# 5. Evaluate
python -m scripts.evaluate --config configs/keafnet_okvqa.yaml --ckpt checkpoints/okvqa/best.pt
```

Or run the whole thing with the orchestrator: `bash scripts/run_pipeline.sh`.

## Repository layout

```
keaf-net/
├── keafnet/
│   ├── models/               # AKF, HGAF, MHSR, encoders, graph builder, full model
│   ├── retrieval/            # ConceptNet+CSKG index, entity extraction, triplet embedding
│   │   ├── kg_index.py       #   KG loader (ConceptNet CSV / CSKG TSV) + 1/2-hop retrieval
│   │   └── retriever.py      #   noun-phrase seeds, Sentence-BERT triplet embedder
│   ├── preprocessing/        # offline feature extraction
│   │   ├── parse_annotations.py  # OK-VQA / A-OKVQA -> common records
│   │   ├── answer_vocab.py       # VQA answer vocab + soft-score encoding
│   │   ├── visual_features.py    # Faster R-CNN regions + ViT grid features
│   │   └── extract_features.py   # end-to-end: image+question -> cached .pt
│   ├── training/trainer.py   # production engine: AMP, DDP, resume, vectorized LOO
│   ├── data/dataset.py       # real + synthetic datasets, collate
│   └── utils/metrics.py      # VQA soft accuracy
├── scripts/                  # train, train_full, evaluate, infer, build_vocab, run_pipeline.sh
├── configs/                  # YAML configs for OK-VQA / A-OKVQA
├── notebooks/demo.ipynb      # interactive walkthrough
├── tests/                    # test_pipeline.py + test_full_pipeline.py
└── docs/                     # DATA.md, MODEL.md
```

## How the modules map to the paper

| Paper section | Module | File |
|---------------|--------|------|
| 3.2 Dual-stream extraction | `VisualEncoder`, `TextEncoder` | `encoders.py` |
| 3.4 Adaptive Knowledge Filter | `AdaptiveKnowledgeFilter` | `akf.py` |
| 3.5 Heterogeneous Graph Adaptive Fusion | `HGAF`, `build_hetero_graph` | `hgaf.py`, `graph_builder.py` |
| 3.6 Multi-Hop Semantic Reasoning | `MHSR` | `mhsr.py` |
| 3.7 Training objective | `KEAFNet.compute_loss` | `keafnet.py` |

## Citation

```bibtex
@article{junnubabu2026keafnet,
  title   = {KEAF-Net: Knowledge-Enhanced Adaptive Fusion Network for Visual
             Question Answering with Multi-Hop Graph Reasoning},
  author  = {Junnu Babu, Noorbhasha and Rajamohana, S. P.},
  year    = {2026}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

## Notes & honest scope

This is a faithful, runnable **reference implementation** built from the
architecture described in the paper. The published accuracy numbers
(68.7% / 62.4%) come from the authors' full experiments with real pre-extracted
features and the complete knowledge-retrieval pipeline; reproducing them
requires preparing the datasets per `docs/DATA.md` and training on GPU. The
synthetic mode verifies correctness of shapes and the training loop, not task
accuracy.
