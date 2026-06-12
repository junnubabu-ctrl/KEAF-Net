# Model Architecture

This document maps each KEAF-Net component to its implementation and explains the
key design decisions.

## Overview

```
A_hat = Classifier( MHSR( HGAF( V, T, AKF(K) ) ) )
```

- `V` : visual features (ViT-B/16 grid + Faster R-CNN regions)
- `T` : question token states (BERT-base)
- `K` : candidate knowledge triplets (Sentence-BERT embeddings)
- `K_hat = AKF(K)` : filtered, low-noise triplets

## 1. Adaptive Knowledge Filter — `models/akf.py`

**Goal:** discard retrieved triplets that do not help reasoning, without ever
looking at the answer string (no label leakage).

- Builds a joint context `h_IQ = LayerNorm(W_q q_cls + W_v mean(V))`.
- Scores each triplet with a three-way interaction
  `alpha_j = sigmoid(MLP([k_j ; h_IQ ; k_j ⊙ h_IQ]))`.
- Applies a **learned threshold** `theta` via a **straight-through estimator**
  (`StraightThroughThreshold`), so the hard keep/discard decision still
  propagates gradients.
- **Supervision (no leakage):** leave-one-out loss differences
  `Delta_j = L(K_hat \ {k_j}) − L(K_hat)` are turned into soft targets
  `t_j = sigmoid(Delta_j / tau)` (stop-gradient). The filter loss is
  `L_AKF = mean (alpha_j − t_j)^2`. A positive `Delta_j` means the triplet was
  useful, so its target relevance is high.

## 2. Heterogeneous Graph Adaptive Fusion — `models/hgaf.py`

**Goal:** fuse the three modalities in one typed graph.

- `build_hetero_graph` (in `graph_builder.py`) creates nodes `[V; T; K]` and a
  `(B, R=5, N, N)` adjacency with five edge types: V-V, T-T, V-K, T-K, V-T.
- `TypeAwareGATLayer` gives **each edge type its own projection matrix**
  `W_{tau}`, then runs multi-head attention
  `e_ij = LeakyReLU(a^T [W_{tau_i} h_i ‖ W_{tau_j} h_j])`. Type-specific weights
  let the layer separate neighbour multisets a type-agnostic GAT would merge
  (the source of the >1-WL expressiveness claim in the paper).
- `GatedFusion` combines the modality streams with learned gates
  `h_fused = Σ_m g_m ⊙ h_m`.
- Defaults: **8 heads, 2 layers**.

## 3. Multi-Hop Semantic Reasoning — `models/mhsr.py`

**Goal:** chain evidence across several reasoning steps.

- Starts from `q0 = q_cls`.
- For `T = 3` hops: attention `beta_i = softmax(q^T W h_i)`, context
  `c = Σ_i beta_i h_i`, update `q ← GRU(c, q)`.
- Returns the refined query, the final context, and the per-hop attention maps
  (useful for interpretability — see `scripts/infer.py`).

## 4. Classifier & losses — `models/keafnet.py`

- `Classifier` maps `[q_final ; ctx]` to answer logits.
- `vqa_loss` is soft-target BCE over the answer vocabulary (standard VQA loss).
- `compute_loss` returns `L = L_VQA + lambda · L_AKF` (default `lambda = 0.3`).
- `loo_deltas` provides a clear (un-vectorised) reference for the LOO supervision;
  production code would batch the masked forward passes.

## Parameter budget

| Component | Params |
|-----------|:------:|
| ViT-B/16 | 86M |
| BERT-base | 110M |
| AKF | ~4M |
| HGAF | ~38M |
| MHSR | ~10M |
| Classifier | ~4M |
| **Total (inference)** | **252M** (184M trainable) |

Faster R-CNN (~60M) and Sentence-BERT (~22–33M) run **offline**; only their
cached features are loaded at train/inference time.

## Notes on faithfulness

- The five edge types and the type-aware attention follow the paper exactly.
- The entity-linking edges (V-K, T-K) are approximated here by embedding cosine
  similarity to keep the module self-contained; swap in your real entity linker
  by editing `graph_builder.build_hetero_graph`.
- The encoders degrade to learnable stubs when `transformers`/`timm` are absent,
  preserving shapes so the pipeline always runs.
