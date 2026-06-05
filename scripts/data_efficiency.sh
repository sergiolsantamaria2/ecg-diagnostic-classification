#!/usr/bin/env bash
# Data-efficiency sweep: macro-AUROC versus label fraction for three regimes
# (supervised from scratch, linear probe, fine-tuning) sharing one SSL-pretrained
# encoder. Small fractions run more epochs (fewer steps per epoch) and repeat
# over three subsampling seeds; the full set runs once.
#
# Each run retries a few times (W&B init can time out transiently) and records a
# done-marker on success, so re-running the script resumes where it left off.
# Usage: scripts/data_efficiency.sh <pretrained_best.pt> [model]
set -uo pipefail

CKPT="${1:?path to pretrained encoder checkpoint required}"
MODEL="${2:-resnet1d}"

declare -A EPOCHS=([0.01]=300 [0.1]=150 [1.0]=50)
SEEDS_SMALL=(0 1 2)
STATE_DIR="outputs/sweep_state"
mkdir -p "$STATE_DIR"

run() {  # regime fraction seed [extra hydra overrides...]
  local regime="$1" frac="$2" seed="$3"
  shift 3
  local marker="$STATE_DIR/${regime}_f${frac}_s${seed}.done"
  if [[ -f "$marker" ]]; then
    echo ">> skip $regime f=$frac s=$seed (already done)"
    return
  fi
  for attempt in 1 2 3; do
    if .venv/bin/python train.py +experiment="$regime" model="$MODEL" \
      data.label_fraction="$frac" data.subsample_seed="$seed" \
      trainer.epochs="${EPOCHS[$frac]}" "$@"; then
      touch "$marker"
      return
    fi
    echo ">> $regime f=$frac s=$seed failed (attempt $attempt/3), retrying in 20s..."
    sleep 20
  done
  echo ">> GAVE UP: $regime f=$frac s=$seed"
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
