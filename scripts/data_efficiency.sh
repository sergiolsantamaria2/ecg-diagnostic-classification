#!/usr/bin/env bash
# Data-efficiency sweep for one regime family: macro-AUROC versus label fraction.
# Small fractions run more epochs (fewer steps per epoch) and repeat over three
# subsampling seeds; the full set runs once. Runs are tagged with the method and
# model so several methods and architectures share the W&B group and the figure.
#
# "scratch" needs no checkpoint and runs the from-scratch baseline; any other
# method runs linear probe and fine-tuning from the given pretrained encoder.
# Each run retries a few times (W&B init can time out) and records a done-marker,
# so re-running resumes.
# Usage: scripts/data_efficiency.sh <method> <model> [pretrained_best.pt]
#   e.g. scripts/data_efficiency.sh scratch     cnn_transformer
#        scripts/data_efficiency.sh contrastive cnn_transformer outputs/<run>/checkpoints/best_pretext.pt
set -uo pipefail

METHOD="${1:?method required (scratch|masked|contrastive|...)}"
MODEL="${2:?model required (resnet1d|cnn_transformer)}"
CKPT="${3:-}"

declare -A EPOCHS=([0.01]=300 [0.1]=150 [1.0]=50)
SEEDS_SMALL=(0 1 2)
STATE_DIR="outputs/sweep_state"
mkdir -p "$STATE_DIR"

run() {  # regime fraction seed [extra hydra overrides...]
  local regime="$1" frac="$2" seed="$3"
  shift 3
  local key="${METHOD}_${MODEL}_${regime}_f${frac}_s${seed}"
  local marker="$STATE_DIR/${key}.done"
  if [[ -f "$marker" ]]; then
    echo ">> skip $key (already done)"
    return
  fi
  for attempt in 1 2 3; do
    if .venv/bin/python train.py +experiment="$regime" model="$MODEL" \
      data.label_fraction="$frac" data.subsample_seed="$seed" \
      trainer.epochs="${EPOCHS[$frac]}" \
      "wandb.name=${regime}_${METHOD}_${MODEL}_f${frac}_s${seed}" \
      "wandb.tags=[data_efficiency,${regime},${METHOD},${MODEL}]" \
      "$@"; then
      touch "$marker"
      return
    fi
    echo ">> $key failed (attempt $attempt/3), retrying in 20s..."
    sleep 20
  done
  echo ">> GAVE UP: $key"
}

sweep_regime() {  # regime [extra hydra overrides...]
  local regime="$1"
  shift
  for frac in 0.01 0.1; do
    for seed in "${SEEDS_SMALL[@]}"; do
      run "$regime" "$frac" "$seed" "$@"
    done
  done
  run "$regime" 1.0 0 "$@"
}

if [[ "$METHOD" == "scratch" ]]; then
  sweep_regime scratch
else
  [[ -n "$CKPT" ]] || { echo "checkpoint required for method '$METHOD'"; exit 1; }
  sweep_regime linear_probe model.pretrained_ckpt="$CKPT"
  sweep_regime finetune model.pretrained_ckpt="$CKPT"
fi
