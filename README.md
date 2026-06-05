# deep-ecg

Deep learning for 12-lead ECG diagnostic classification on **PTB-XL**, built in
phases. Phase 1 is a reproducible supervised baseline for multi-label
classification of the 5 diagnostic superclasses; the codebase is structured so
that self-supervised pretraining (Phase 2) and cross-source generalization
(Phase 3) plug in without rewrites.

## Roadmap

| Phase | Goal | Status |
|------:|------|:------:|
| 1 | Supervised baseline on PTB-XL — 5 superclasses, two architectures, full evaluation | Done |
| 2+ | Open, results-driven directions (e.g. self-supervised pretraining, label-efficiency, cross-source transfer) | Planned |

The project is built to grow in phases. The encoder is kept strictly separable
from the task head, and the data layer is built around an `ECGSource` adapter,
so the same encoder and pipeline extend to self-supervised and cross-source
regimes without rewrites. Directions beyond Phase 1 are chosen from what the
results suggest rather than fixed in advance.

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

AUROC on the held-out test fold (fold 10), best checkpoint by validation
macro-AUROC. Both models train for 150 epochs with AdamW, a cosine schedule and
augmentation (temporal shift, scaling, lead masking, Gaussian noise).

| Model | Params | macro-AUROC | NORM | MI | STTC | CD | HYP |
|-------|:------:|:-----------:|:----:|:--:|:----:|:--:|:---:|
| ResNet1D (ResNet18-scale) | 6.3M | **0.923** | 0.946 | 0.930 | 0.931 | 0.917 | 0.892 |
| CNN + Transformer | 2.9M | 0.920 | 0.941 | 0.925 | 0.932 | 0.918 | 0.885 |

The PTB-XL benchmark of Strodthoff et al. reports a best macro-AUROC of ≈0.93 on
the diagnostic superclass task; both baselines reach that level.

### Effect of augmentation

| Model | no augmentation | augmentation | best epoch (no aug → aug) |
|-------|:---------------:|:------------:|:-------------------------:|
| ResNet1D | 0.922 | **0.923** | 10 → 22 |
| CNN + Transformer | 0.914 | **0.920** | 5 → 22 |

Without augmentation both models overfit early (training loss →0, validation
AUROC declining) and peak within the first ~10 epochs. Augmentation delays
overfitting — the best epoch roughly doubles — and lifts both models to the
benchmark level.

### Findings

- The residual CNN edges the CNN+Transformer (0.923 vs 0.920). On ~17k labelled
  records the Transformer's weaker inductive bias offers little advantage at
  this scale, though the gap is small.
- Augmentation helps the Transformer most (+0.006 vs +0.001 test macro-AUROC),
  narrowing the gap from 0.009 to 0.003 — consistent with the more data-hungry
  model gaining more from a data multiplier.
- HYP is the hardest superclass for both (AUROC ≈0.89, F1 ≈0.59), consistent
  with its low prevalence (12%) and amplitude-based definition, which per-lead
  standardization partially flattens.
- The early-overfitting regime — both models saturate ~17k labels quickly —
  motivates studying label efficiency and representation learning beyond the
  supervised baseline.

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
uv sync                                          # environment from pyproject + uv.lock
uv run python scripts/download_ptbxl.py          # download + extract PTB-XL

uv run python train.py +experiment=sanity                  # end-to-end smoke run
uv run python train.py +experiment=baseline_resnet1d       # ResNet1D baseline
uv run python train.py +experiment=baseline_cnn_transformer  # CNN+Transformer baseline

uv run python scripts/evaluate.py <run_dir>      # test metrics + per-class report
```

Experiments are config-driven (Hydra), tracked in Weights & Biases, and seeded
for reproducibility. Any component swaps with a single override, e.g.
`train.py model=cnn_transformer trainer.epochs=30 data.augmentations.gaussian_noise=0.1`.
