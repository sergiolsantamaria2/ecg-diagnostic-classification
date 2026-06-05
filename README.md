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

AUROC on the held-out test fold (fold 10), best checkpoint by validation
macro-AUROC. Both models train for 50 epochs with AdamW and a cosine schedule,
no augmentation.

| Model | Params | macro-AUROC | NORM | MI | STTC | CD | HYP |
|-------|:------:|:-----------:|:----:|:--:|:----:|:--:|:---:|
| ResNet1D (ResNet18-scale) | 6.3M | **0.922** | 0.938 | 0.921 | 0.935 | 0.917 | 0.900 |
| CNN + Transformer | 2.9M | 0.914 | 0.939 | 0.907 | 0.924 | 0.901 | 0.897 |

The PTB-XL benchmark of Strodthoff et al. reports a best macro-AUROC of ≈0.93 on
the diagnostic superclass task; the ResNet1D baseline lands within ≈0.01 of it
despite using a ResNet18-scale network without augmentation.

### Findings

- The residual CNN edges the CNN+Transformer (0.922 vs 0.914). On ~17k labelled
  records the Transformer's weaker inductive bias offers no advantage at this
  scale.
- HYP is the hardest superclass for both models (AUROC ≈0.90, F1 ≈0.60),
  consistent with its low prevalence (12%) and amplitude-based definition, which
  per-lead standardization partially flattens.
- Both models reach their best validation macro-AUROC very early — epoch 10
  (ResNet1D) and epoch 5 (CNN+Transformer) — and then overfit (training loss
  →0, validation AUROC declines). Selection by validation macro-AUROC recovers
  the peak. This early overfitting motivates the augmentation ablations and,
  more fundamentally, the self-supervised pretraining and data-efficiency study
  planned for Phase 2.

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
