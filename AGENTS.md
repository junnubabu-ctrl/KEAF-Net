# KEAF-Net Agent Work Contract

## Goal
Complete a researcher-quality, reproducible implementation of KEAF-Net for Visual Question Answering without fabricating experimental results or pretending unverified implementation details are exact.

## Research fidelity
Implement only details supported by the manuscript/source materials or explicitly documented assumptions. When a paper detail is ambiguous, record the assumption in `docs/IMPLEMENTATION_ASSUMPTIONS.md` and keep the code configurable.

## Required modules
- dataset interfaces and preprocessing
- visual encoder interfaces
- question/text encoder interfaces
- knowledge retrieval/enrichment interfaces
- adaptive knowledge fusion
- heterogeneous graph construction/reasoning
- multi-hop reasoning
- VQA prediction head and losses
- train/validate/test/evaluate/infer entry points
- deterministic seed/reproducibility utilities
- configuration files
- unit tests and a synthetic end-to-end smoke test
- researcher-facing documentation

## Safety and publication rules
Never commit API keys, tokens, passwords, `.env` files, private keys, confidential data, copyrighted/restricted datasets, or model weights without confirmed redistribution rights. Use placeholders and official acquisition instructions instead.

## Validation
Before declaring the implementation complete:
1. run unit tests;
2. run a synthetic smoke test that does not require restricted datasets;
3. verify import/install instructions;
4. scan tracked files for obvious secrets;
5. ensure README commands correspond to real scripts;
6. document any results that were not independently reproduced as `not yet independently reproduced`.

## Definition of done
The repository is understandable to an independent researcher, contains no fabricated claims, and has exact commands for environment setup, data preparation, training, evaluation and inference. Keep all experimental claims traceable to evidence or clearly marked as targets/reference values.
