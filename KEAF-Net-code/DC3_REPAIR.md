# Corrected implementation revision

This local revision starts at `junnubabu-ctrl/KEAF-Net` commit
`c8a07e47cbf9084a8e2e52d65347e34fab022104`. It repairs software defects identified
in the DC3 source audit. It is not the historical trained system and does not
authenticate any published/manuscript benchmark accuracy. Historical metrics
are **not yet independently reproduced**. New cache generation, training and
evaluation are required before attributing a result to these changes.

## Verified behavior

- VQA/OK-VQA uses leave-one-annotator-out agreement, with official conditional
  answer processing. The punctuation condition and contraction mapping are
  corrected. A-OKVQA DA remains exact-string `min(1,matches/3)` and excludes
  `difficult_direct_answer=True` from its evaluated population. Annotation
  conversion preserves that field. Missing labels do not produce an accuracy.
- AKF keeps the same hard inference comparison. A threshold-dependent sigmoid
  STE supplies score and threshold gradients during training. Its gate
  temperature is an explicit configurable engineering choice, not a recovered
  historical setting. Gradient and optimizer-update tests establish that the
  repaired path can train the threshold, not that it improves benchmark answers.
- Leave-one-out utility measurements use a deterministic reasoning baseline
  and removal passes, with dropout disabled for both. Training state is restored
  afterward. VQA loss still uses the normal training path.
- Detector links use original detector row IDs plus the grid-node offset.
  Character spans map to every overlapping BERT WordPiece token using the
  cached tokenizer offsets. Special and truncated-away tokens are excluded.
- Cache schema version 2 declares node index spaces and checks counts and the
  question hash. Legacy phrase-index caches are rejected because a reliable
  conversion requires regenerating the token/span mapping.
- Detector boxes require an explicit coordinate frame and image size; boxes
  are transformed into the resized grid pixel frame. Original images are not
  overwritten. The dump's label order and coordinate metadata must be supplied
  from its source rather than inferred from coordinate magnitudes.
- The 384→768 knowledge projection and the HGAF fused-node collection passed
  to MHSR are explicit code interfaces. Their historical trained settings remain
  unresolved. Non-224 ViT extraction requires explicit positional interpolation
  and a newly documented experiment. 224/16 and 448/16 give 196/784 patches.

## Commands

Run from `KEAF-Net-code/`, using Python ≥3.10 and the listed dependencies:

```bash
python -m pip install -e '.[dev]'
OMP_NUM_THREADS=1 python -m pytest -q --junitxml=verification/tests.xml
python scripts/verify_dc3_repairs.py --junit verification/tests.xml
python scripts/train.py --help
python scripts/evaluate.py --help
```

The retained JSON and XML are synthetic software verification. They contain no
dataset benchmark prediction, published-score reproduction or real-data model
demonstration. CPU verification was available; no GPU, full benchmark feature
cache, original trained checkpoint or historical evaluation archive was used.

Before real-data extraction, install the `extract` extra and torchvision/Pillow,
obtain the named encoder weights, acquire the official image/question splits and
licensed knowledge dumps, and supply detector feature/label metadata. For each
grid file store `box_coordinate_frame='processed_pixels'`, `image_size=[W,H]`,
and `source_image_size=[native_W,native_H]`. Grid files declare `spatial_transform='resize_only'`. Detector files declare
`original_pixels`, `processed_pixels` or `normalized_xyxy`; processed/normalized
frames also require `spatial_transform='resize_only'`. Crops and letterboxes
are rejected because image sizes cannot recover their transforms. Selected detector
labels may be represented as `{labels:[...], region_indices:[...]}`; a bare list
must follow the exact detector feature-row order.

Full GQA/TextVQA adapters, OCR, corrected-revision multi-seed benchmark runs,
ablations, robustness, significance and cost analysis remain outstanding. An
auxiliary loss weight of zero disables AKF supervision; it does not remove AKF
gating or constitute a full no-AKF architectural ablation.

## Primary evaluator sources

- VQA: `GT-Vision-Lab/VQA/PythonEvaluationTools/vqaEvaluation/vqaEval.py`,
  fetched blob `e4ff7887d53195f12856ab1e9087e69abe2e75c8`.
- A-OKVQA: `allenai/aokvqa/evaluation/eval_predictions.py`, fetched blob
  `7edb3bec424a20a747193ef605e77591fe7fb8fd`.
