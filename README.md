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

Per-class AUROC on the held-out test fold (fold 10). The headline model is an
ensemble of both architectures across two augmentation regimes; it matches the
PTB-XL benchmark.

| Model | macro-AUROC | NORM | MI | STTC | CD | HYP |
|-------|:-----------:|:----:|:--:|:----:|:--:|:---:|
| ResNet1D (ResNet18-scale, 6.3M) | 0.923 | 0.946 | 0.930 | 0.931 | 0.917 | 0.892 |
| CNN + Transformer (2.9M, crop + TTA) | 0.923 | 0.941 | 0.925 | 0.932 | 0.918 | 0.885 |
| **Ensemble** | **0.928** | 0.949 | 0.935 | 0.939 | 0.926 | 0.893 |

Strodthoff et al. report a best macro-AUROC of ≈0.93 on the diagnostic
superclass task; the ensemble reaches that level.

### Path to the benchmark

Single models plateau around 0.92; the gap closes through augmentation,
test-time crop augmentation and ensembling.

| Step | ResNet1D | CNN + Transformer |
|------|:--------:|:-----------------:|
| no augmentation (50 ep) | 0.922 | 0.914 |
| + augmentation (150 ep) | 0.923 | 0.920 |
| + random crop & TTA (200 ep) | 0.922 | 0.923 |
| **ensemble of the four runs** | **0.928** | |

### Findings

- Augmentation delays overfitting: without it both models peak within ~10
  epochs and then memorize (training loss →0, validation AUROC declining); with
  it the best epoch roughly doubles (to ~22).
- It helps the CNN+Transformer most — under strong augmentation with random
  cropping the Transformer overtakes the ResNet — consistent with the more
  data-hungry, lower-inductive-bias model gaining more from a data multiplier.
- Training on random 5 s crops requires matching test-time crop averaging (TTA);
  evaluating the cropped model on the full 10 s signal underperforms.
- HYP is the hardest superclass (AUROC ≈0.89, F1 ≈0.57): lowest prevalence
  (12%) and amplitude-based, which per-lead standardization partially flattens.
- Single models sit ~0.005 below the benchmark and only reach it in ensemble;
  this saturation of ~17k labels motivates studying label efficiency and
  representation learning beyond the supervised baseline.

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
uv run python train.py +experiment=resnet1d_aug            # ResNet1D, augmentation
uv run python train.py +experiment=cnn_transformer_crop    # Transformer, crop augmentation

uv run python scripts/evaluate.py <run_dir>                # test metrics + per-class report
uv run python scripts/evaluate.py <run_dir> --tta-crop-len 500   # with test-time crop averaging
uv run python scripts/ensemble.py <run_dir> <run_dir> ...  # ensemble (auto-TTA for cropped runs)
```

Experiments are config-driven (Hydra), tracked in Weights & Biases, and seeded
for reproducibility. Any component swaps with a single override, e.g.
`train.py model=cnn_transformer trainer.epochs=30 data.augmentations.gaussian_noise=0.1`.
