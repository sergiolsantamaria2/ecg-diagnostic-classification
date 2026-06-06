#!/usr/bin/env bash
# Data-efficiency sweep: macro-AUROC versus label fraction for three regimes
# (supervised from scratch, linear probe, fine-tuning) sharing one pretrained
# encoder. Small fractions run more epochs (fewer steps per epoch) and repeat
# over three subsampling seeds; the full set runs once.
#
# An optional method label tags the SSL runs (e.g. masked, contrastive) and is
# included in their W&B run names, so several pretext methods share one figure.
# The from-scratch baseline is method-independent and only runs when no method is
# given. Each run retries a few times (W&B init can time out transiently) and
# records a done-marker on success, so re-running the script resumes.
# Usage: scripts/data_efficiency.sh <pretrained_best.pt> [model] [method]
set -uo pipefail

CKPT="${1:?path to pretrained encoder checkpoint required}"
MODEL="${2:-resnet1d}"
METHOD="${3:-}"

declare -A EPOCHS=([0.01]=300 [0.1]=150 [1.0]=50)
SEEDS_SMALL=(0 1 2)
STATE_DIR="outputs/sweep_state"
mkdir -p "$STATE_DIR"

tag_suffix=""
name_suffix=""
if [[ -n "$METHOD" ]]; then
  tag_suffix=",$METHOD"
  name_suffix="${METHOD}_"
fi

run() {  # regime fraction seed [extra hydra overrides...]
  local regime="$1" frac="$2" seed="$3"
  shift 3
  local key="${regime}_${name_suffix}f${frac}_s${seed}"
  local marker="$STATE_DIR/${key}.done"
  if [[ -f "$marker" ]]; then
    echo ">> skip $key (already done)"
    return
  fi
  for attempt in 1 2 3; do
    if .venv/bin/python train.py +experiment="$regime" model="$MODEL" \
      data.label_fraction="$frac" data.subsample_seed="$seed" \
      trainer.epochs="${EPOCHS[$frac]}" \
      "wandb.name=${regime}_${name_suffix}${MODEL}_f${frac}_s${seed}" \
      "wandb.tags=[data_efficiency,${regime}${tag_suffix}]" \
      "$@"; then
      touch "$marker"
      return
    fi
    echo ">> $key failed (attempt $attempt/3), retrying in 20s..."
    sleep 20
  done
  echo ">> GAVE UP: $key"
}

if [[ -z "$METHOD" ]]; then  # from-scratch baseline is shared across methods
  for frac in 0.01 0.1; do
    for seed in "${SEEDS_SMALL[@]}"; do
      run scratch "$frac" "$seed"
    done
  done
  run scratch 1.0 0
fi

for frac in 0.01 0.1; do
  for seed in "${SEEDS_SMALL[@]}"; do
    run linear_probe "$frac" "$seed" model.pretrained_ckpt="$CKPT"
    run finetune "$frac" "$seed" model.pretrained_ckpt="$CKPT"
  done
done
run linear_probe 1.0 0 model.pretrained_ckpt="$CKPT"
run finetune 1.0 0 model.pretrained_ckpt="$CKPT"
