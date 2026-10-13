# Implementation status — KEAF-Net (PhD Objective 1)

Objective: a knowledge-enhanced VQA framework that treats knowledge selection,
multimodal fusion and reasoning depth as one connected problem — realised as
the three proposed modules (AKF, HGAF, MHSR) around dual-stream encoders and
ConceptNet+CSKG retrieval, trained end-to-end with retrieval-consistency
supervision. This document maps each element of the paper to the code that
implements it and records how each part was verified.

## Paper-to-code traceability

| Paper element | Where | Code | Status |
| --- | --- | --- | --- |
| Overall pipeline `A = Classifier(MHSR(HGAF(V,T,K)))` | Eq. (1) | `keaf_net/models/keaf_net.py` | Implemented, tested |
| ViT-B/16 grid features (layers 1–6 frozen) | Eq. (2), §3.2.1 | `keaf_net/models/encoders.py::VisualEncoder` | Implemented |
| Faster R-CNN 36-region features (pre-extracted) | Eq. (3) | loader contract in `keaf_net/data/dataset.py`; format in `DATA.md` | Implemented |
| Concatenated visual stream `V=[V_grid;V_region]` | Eq. (4) | `keaf_net/models/keaf_net.py::assemble_graph` | Implemented, tested |
| BERT-base question encoding, frozen embeddings | Eq. (5), §3.2.2 | `keaf_net/models/encoders.py::TextEncoder` | Implemented |
| Entity-seeded 1/2-hop retrieval, top-50 cap | Eq. (6), §3.3 | `keaf_net/knowledge/retriever.py::KnowledgeRetriever` | Implemented, tested |
| Sentence-BERT triplet embedding | Eq. (7) | `keaf_net/models/encoders.py::TripletEncoder` | Implemented |
| Image-question context `h_IQ` | Eq. (8) | `keaf_net/models/akf.py::context_vector` | Implemented, tested |
| Three-way triplet scoring | Eq. (9) | `keaf_net/models/akf.py::forward` | Implemented, tested |
| Learned keep threshold `c_th`, STE gating | Eq. (10) | `keaf_net/models/akf.py` (`threshold_logit`, STE gate) | Implemented, tested |
| Leave-one-out loss differences `d_j` | Eq. (11) | `keaf_net/engine/trainer.py::_akf_loss` | Implemented, tested |
| Soft targets via temperature sigmoid, stop-grad | Eq. (12) | `keaf_net/models/akf.py::loss` | Implemented, tested |
| Filter regression loss `L_AKF` | Eq. (13) | `keaf_net/models/akf.py::loss` | Implemented, tested |
| Sampled `|S|=10` LOO subset per batch | §3.4 | `keaf_net/engine/trainer.py::_akf_loss` | Implemented, tested |
| Five typed edge constructions (a)–(e) | §3.5, Alg. 1 | `keaf_net/models/graph.py::build_sample_graph` | Implemented, tested |
| Exact + WordNet-lemma entity linking | §3.5 | `keaf_net/knowledge/retriever.py::EntityLinker` | Implemented, tested |
| Type-aware multi-head GAT (8 heads, 2 layers) | Eq. (14)–(16) | `keaf_net/models/hgaf.py::TypedGATLayer` | Implemented, tested |
| Gated three-stream fusion | Eq. (17)–(18) | `keaf_net/models/hgaf.py::HGAF` | Implemented, tested |
| GRU-based reasoning hops, `T=3` | Eq. (19)–(21) | `keaf_net/models/mhsr.py` | Implemented, tested |
| Answer classifier over `[q^(T); c^(T)]` | Eq. (22) | `keaf_net/models/mhsr.py` | Implemented, tested |
| Joint objective `L_VQA + 0.3·L_AKF` | Eq. (23) | `keaf_net/engine/trainer.py::compute_losses` | Implemented, tested |
| Training procedure (AdamW, cosine LR, early stop) | Alg. 2, Table 4 | `keaf_net/engine/trainer.py`, `scripts/train.py` | Implemented |
| AKF inference gating (no LOO at test time) | Alg. 3 | `keaf_net/models/akf.py::forward` (same path) | Implemented, tested |
| VQA soft accuracy, answer normalisation | Eq. (25), §5.4 | `keaf_net/utils/metrics.py`, `keaf_net/utils/text.py` | Implemented, tested |
| 3,129-answer vocabulary with soft targets | §5.4 | `keaf_net/data/vocab.py`, `scripts/build_vocab.py` | Implemented, tested |
| Wilcoxon significance test | §5.4 | `keaf_net/utils/metrics.py::wilcoxon_pvalue` | Implemented, tested |
| Three-seed protocol {42, 123, 2024} | Table 4 | `keaf_net/utils/seed.py`, `scripts/train.py --seed` | Implemented |
| Table 4 hyperparameters | Table 4 | `keaf_net/config.py`, `configs/*.yaml` | Implemented |
| Dataset acquisition and conversion | §5.1, Table 3 | `scripts/download_data.sh`, `scripts/prepare_annotations.py`, `DATA.md` | Implemented, tested |
| ConceptNet+CSKG merged index | §3.3 | `scripts/build_kg_csv.py` | Implemented, tested |

Ablation switches used in Section 6 are plain configuration changes:
`optim.akf_loss_weight=0` (AKF supervision off), `model.num_hops∈{1..5}`
(Table 10 depth study), `model.max_triplets` (P sweep), and single-source
knowledge indexes built by passing only one input to `build_kg_csv.py`
(Table 9).

## Corrected-revision verification record

The legacy verification claims previously written here did not establish the
threshold gradient or authentic benchmark execution. The current retained
record is `verification/DC3_SOFTWARE_VERIFICATION.json` and its pytest XML.
These are software tests and a synthetic CPU forward/backward/optimizer probe.
They are not real-data benchmark results. See `DC3_REPAIR.md` for changes,
cache migration and exact verification commands.

Full benchmark accuracies, multi-seed means, ablations and significance remain
**not yet independently reproduced**. Corrected scoring, grounding and gate
training require regenerated caches, new training and official evaluation.
Historical checks cannot be reassigned to the repaired implementation.
