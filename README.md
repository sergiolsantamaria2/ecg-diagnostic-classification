# deep-ecg

Deep learning for 12-lead ECG diagnostic classification on **PTB-XL**, built in
phases. Phase 1 is a reproducible supervised baseline for multi-label
classification of the 5 diagnostic superclasses; the codebase is structured so
that self-supervised pretraining (Phase 2) and cross-source generalization
(Phase 3) plug in without rewrites.

## Roadmap

| Phase | Goal | Status |
|------:|------|:------:|
| 1 | Supervised baseline on PTB-XL — 5 superclasses, two architectures, full evaluation | In progress |
| 2 | Self-supervised pretraining (masked-signal / contrastive) + data-efficiency study (1 / 10 / 100 % labels) | Designed |
| 3 | Cross-source pretraining (e.g. MIMIC-IV-ECG) + transfer to unseen databases (PhysioNet/CinC 2021) | Designed |

The encoder is kept strictly separable from the task head, and the data layer is
built around an `ECGSource` adapter, so the same encoder and pipeline serve the
supervised, self-supervised, and cross-source regimes.

## Task and data

- **Dataset:** PTB-XL (PhysioNet) — ~21.8k 10-second 12-lead ECGs, WFDB format,
  100 Hz and 500 Hz. Phase 1 uses 100 Hz.
- **Labels:** SCP codes (`ptbxl_database.csv`) mapped to the 5 diagnostic
  superclasses NORM, MI, STTC, CD, HYP via `scp_statements.csv`.
- **Task:** multi-label classification (an ECG may carry several conditions).
- **Splits:** official stratified folds (`strat_fold`). Train = folds 1–8,
  validation = fold 9, test = fold 10.

## Models

Two architectures under comparable conditions:

1. **ResNet1D** — residual 1D CNN (xresnet1d style).
2. **CNN + Transformer** — convolutional feature extractor followed by a
   Transformer encoder over the temporal feature sequence.

## Evaluation

Primary metric **macro-AUROC**, reported with per-class AUROC and F1, on the
held-out test fold (fold 10). Results are compared against the PTB-XL benchmark
of Strodthoff et al. as reference.

### Results

_Reported on fold 10. Reference column from Strodthoff et al._

| Model | macro-AUROC | NORM | MI | STTC | CD | HYP | Strodthoff (ref.) |
|-------|:-----------:|:----:|:--:|:----:|:--:|:---:|:-----------------:|
| ResNet1D | — | — | — | — | — | — | — |
| CNN + Transformer | — | — | — | — | — | — | — |

## Project structure

```
configs/        Hydra config tree (data / model / task / trainer / experiment)
src/deep_ecg/
  data/         ECGSource adapter, datasets, transforms, SCP→superclass labels
  models/       encoders, heads, EncoderHeadModel (encoder ⟂ head)
  training/     custom trainer, losses, schedulers
  evaluation/   metrics, error analysis
scripts/        dataset download and utilities
```

## Setup

```bash
uv sync                       # create the environment from pyproject + uv.lock
uv run python scripts/download_ptbxl.py
uv run python train.py experiment=sanity     # end-to-end smoke run
```

Experiments are config-driven (Hydra), tracked in Weights & Biases, and seeded
for reproducibility.
