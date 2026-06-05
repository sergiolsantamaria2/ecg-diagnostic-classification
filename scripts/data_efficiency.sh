#!/usr/bin/env bash
# Data-efficiency sweep: macro-AUROC versus label fraction for three regimes
# (supervised from scratch, linear probe, fine-tuning) sharing one SSL-pretrained
# encoder. Small fractions run more epochs (fewer steps per epoch) and repeat
# over three subsampling seeds; the full set runs once.
# Usage: scripts/data_efficiency.sh <pretrained_best.pt> [model]
set -euo pipefail

CKPT="${1:?path to pretrained encoder checkpoint required}"
MODEL="${2:-resnet1d}"

declare -A EPOCHS=([0.01]=300 [0.1]=150 [1.0]=50)
SEEDS_SMALL=(0 1 2)

run() {  # regime fraction seed [extra hydra overrides...]
  local regime="$1" frac="$2" seed="$3"
  shift 3
  .venv/bin/python train.py +experiment="$regime" model="$MODEL" \
    data.label_fraction="$frac" data.subsample_seed="$seed" \
    trainer.epochs="${EPOCHS[$frac]}" "$@"
}

for frac in 0.01 0.1; do
  for seed in "${SEEDS_SMALL[@]}"; do
    run scratch "$frac" "$seed"
    run linear_probe "$frac" "$seed" model.pretrained_ckpt="$CKPT"
    run finetune "$frac" "$seed" model.pretrained_ckpt="$CKPT"
  done
done

run scratch 1.0 0
run linear_probe 1.0 0 model.pretrained_ckpt="$CKPT"
run finetune 1.0 0 model.pretrained_ckpt="$CKPT"
