# deep-ecg

[![CI](https://github.com/sergiolsantamaria2/deep-ecg/actions/workflows/ci.yml/badge.svg)](https://github.com/sergiolsantamaria2/deep-ecg/actions/workflows/ci.yml)

Multi-label diagnostic classification of 12-lead ECGs on **PTB-XL** (5 diagnostic
superclasses). A residual 1D CNN and a CNN+Transformer are trained, regularized
with signal augmentation, and ensembled to the level of the PTB-XL benchmark. Two
self-supervised objectives — masked signal reconstruction and a contrastive
(CLOCS-style) objective — then pretrain the encoder on the unlabeled signals,
improving label efficiency in the low-data regime. The same pretraining,
performed on five *other* ECG databases and transferred to PTB-XL, retains the
gain across a change of acquisition source.

![Results](figures/results.png)

## Results

macro-AUROC on the official test fold (fold 10), best checkpoint by validation
macro-AUROC.

| Model | macro-AUROC | NORM | MI | STTC | CD | HYP |
|-------|:-----------:|:----:|:--:|:----:|:--:|:---:|
| ResNet1D (6.3M) | 0.923 | 0.946 | 0.930 | 0.931 | 0.917 | 0.892 |
| CNN+Transformer (2.9M, crop + TTA) | 0.923 | 0.941 | 0.925 | 0.932 | 0.918 | 0.885 |
| **Ensemble** | **0.928** | 0.949 | 0.935 | 0.939 | 0.926 | 0.893 |

Strodthoff et al. report a best macro-AUROC of ≈0.93 on this task; the ensemble
matches it (0.928, bootstrap 95% CI 0.921–0.936).

## What the experiments show

- Without augmentation both models overfit within ~10 epochs. Augmentation
  delays this and helps the CNN+Transformer most — under strong augmentation with
  random cropping it overtakes the ResNet, consistent with the more data-hungry
  model gaining more from a data multiplier.
- Random-crop training requires matching test-time crop averaging (TTA);
  evaluating a cropped model on the full-length signal underperforms.
- Ensembling the two architectures closes the remaining gap to the benchmark.
- HYP is the hardest superclass (lowest prevalence at 12%, amplitude-based).

## Label efficiency with self-supervised pretraining

The encoder is pretrained on the unlabeled training signals with two
self-supervised objectives — masked reconstruction (random 0.5 s spans masked
across all leads and reconstructed under MSE) and a contrastive CLOCS-style
objective (two augmented crops of a record as a positive pair, NT-Xent loss) —
then assessed for both architectures under three regimes at 1%, 10% and 100% of
the labels: training from scratch, a linear probe on the frozen encoder, and
fine-tuning. Test macro-AUROC; the 1% and 10% regimes report the mean over three
subsampling seeds.

![ResNet1D label efficiency](figures/data_efficiency.png)

ResNet1D:

| Regime | Pretext | 1% | 10% | 100% |
|--------|---------|:--:|:---:|:----:|
| From scratch | — | 0.822 | 0.870 | 0.922 |
| Linear probe | Masked | 0.804 | 0.849 | 0.868 |
| Linear probe | Contrastive | 0.811 | 0.860 | 0.875 |
| Fine-tune | Masked | 0.831 | 0.884 | 0.919 |
| Fine-tune | Contrastive | 0.830 | 0.889 | **0.924** |

![CNN+Transformer label efficiency](figures/data_efficiency_cnn_transformer.png)

- Fine-tuning a self-supervised encoder improves label efficiency in the
  low-label regime for both architectures, and the gain is larger for the more
  data-hungry CNN+Transformer: at 1% labels it lifts test macro-AUROC by ~2
  points (0.805 from scratch to 0.827 fine-tuned) versus ~1 point for the
  ResNet1D.
- The contrastive objective is the stronger pretext on both: its frozen features
  are more linearly separable than the reconstruction ones (higher linear probe),
  and fine-tuning matches or exceeds masked reconstruction — masked keeps only a
  slight edge at 1% fine-tuning.
- Contrastive fine-tuning also improves on from-scratch at full labels for both
  architectures (ResNet1D 0.924 vs 0.922; CNN+Transformer 0.920 vs 0.915), where
  masked reconstruction gives little to no gain.
- A frozen encoder (linear probe) still trails end-to-end training for every
  pretext and architecture — the diagnostic task needs the encoder to adapt.

## Pretraining checkpoint selection

The pretext loss is an imperfect proxy for downstream quality, so pretraining is
monitored online: every 10 epochs a linear probe (frozen encoder, validation
fold) measures the encoder's diagnostic AUROC, which selects the checkpoint and
early-stops training.

![Pretraining diagnostic](figures/probe_diagnostic.png)

Masked reconstruction tracks the probe monotonically — loss and downstream
quality improve together. The contrastive probe instead peaks around epoch 70 and
then declines while the NT-Xent loss keeps falling, the objective over-optimizing
augmentation invariances past the point of useful representations.

Selecting the checkpoint by the probe rather than by the pretext loss did not
change the downstream results, however — within ~0.1–0.5 points on the test fold,
in either direction. The validation-probe divergence is small in absolute terms
and does not transfer to the test set, and fine-tuning re-adapts the encoder
enough to absorb it. The pretext loss is therefore an adequate selector here; the
online probe earns its place as a diagnostic and an early-stopping signal (it
halted the contrastive run at epoch 100 rather than 200).

## Cross-source generalization

The self-supervised recipe is source-agnostic by construction — it consumes raw
signals, not labels — so the encoder can be pretrained on entirely different ECG
databases and transferred to PTB-XL. To test whether the label-efficiency gain
survives a change of acquisition source, the encoder is pretrained (no labels) on
a source-balanced ~40k-record corpus pooled from the five non-PTB-XL
PhysioNet/CinC 2021 databases (CPSC, CPSC-Extra, Georgia, Chapman-Shaoxing,
Ningbo), each harmonized to the canonical 12-lead, 100 Hz, 10-second format, then
fine-tuned and probed on held-out PTB-XL exactly as before. PTB-XL is never seen
during pretraining.

![ResNet1D cross-source transfer](figures/cross_source.png)

ResNet1D, test macro-AUROC (in-domain pretraining on PTB-XL vs cross-source
pretraining on the five other databases):

| Regime | Pretext | In-domain | Cross-source |
|--------|---------|:---------:|:------------:|
| Linear probe | Masked | 0.804 / 0.849 / 0.868 | 0.820 / 0.860 / 0.876 |
| Linear probe | Contrastive | 0.811 / 0.860 / 0.875 | 0.816 / 0.865 / 0.879 |
| Fine-tune | Masked | 0.831 / 0.884 / 0.919 | 0.836 / 0.886 / 0.919 |
| Fine-tune | Contrastive | 0.830 / 0.889 / 0.924 | 0.828 / 0.887 / **0.925** |

Cells are 1% / 10% / 100% of the labels.

![CNN+Transformer cross-source transfer](figures/cross_source_cnn_transformer.png)

- Pretraining on foreign sources carries no transfer penalty: the cross-source
  curves match or exceed the in-domain ones for both architectures. A diverse
  multi-source corpus is at least as useful as PTB-XL's own signals.
- The effect is clearest under the linear probe, where cross-source features are
  consistently more linearly separable for the PTB-XL task than in-domain ones
  (masked reconstruction gains ~1.6 points at 1% labels for the ResNet1D) — a
  larger, more heterogeneous pretraining distribution yields more transferable
  representations.
- Fine-tuning is on par across sources, and the best full-label result in the
  study is cross-source contrastive fine-tuning (ResNet1D 0.925, CNN+Transformer
  0.922), edging both the in-domain and the from-scratch references.

## Task and data

PTB-XL (PhysioNet): ~21.8k 10-second 12-lead ECGs at 100 Hz. SCP codes are mapped
to the five diagnostic superclasses (NORM, MI, STTC, CD, HYP); the official
stratified folds are used (train 1–8, validation 9, test 10). Exploratory
analysis is in [`notebooks/eda.ipynb`](notebooks/eda.ipynb). PTB-XL is released
under CC-BY 4.0 and is not redistributed here.

Cross-source pretraining draws on the five non-PTB-XL databases of the
PhysioNet/Computing in Cardiology Challenge 2021 (CPSC and CPSC-Extra, Georgia,
Chapman-Shaoxing, Ningbo): heterogeneous in sampling rate and duration, they are
resampled, reordered and cropped to the canonical 12-lead, 100 Hz, 10-second
format by the `ECGSource` adapter and pooled into a source-balanced corpus
(capped at 10k records per database). Their SNOMED-CT labels are unused — the
corpus serves self-supervised pretraining only.

## Design

The encoder is kept separate from the task head and the data layer is built
around an `ECGSource` adapter, so the encoder and pipeline extend to
self-supervised pretraining and additional ECG databases without rewrites.
Experiments are config-driven (Hydra), tracked in Weights & Biases and seeded.

## Serving & deployment

The trained model is packaged into a self-describing serving bundle and exposed
behind a small FastAPI service. Inference runs on ONNX Runtime, with the
TorchScript model under PyTorch as a fallback. The default serving target is the
single ResNet1D model on the full-length signal — it needs no test-time
augmentation, so it exports to a static graph and is the cheapest to serve; the
export tooling also accepts several runs to serve the full ensemble.

Export a trained run to a bundle (ONNX + TorchScript + lead statistics, class
names, decision thresholds and input metadata). Export verifies that ONNX and
TorchScript match PyTorch within tolerance on random inputs:

```bash
uv run python scripts/export_model.py outputs/<run> --out artifacts/resnet1d
# ensemble (faithful to the benchmark): pass several runs and the tuned thresholds
uv run python scripts/export_model.py outputs/<run1> outputs/<run2> ... \
    --out artifacts/ensemble --thresholds ensemble.json
```

Serve it locally; `MODEL_DIR` selects the bundle:

```bash
MODEL_DIR=artifacts/resnet1d uv run uvicorn deep_ecg.serving.api:app --port 8000
```

Or in Docker — a lightweight ONNX-only image (no PyTorch or training stack):

```bash
docker build -t deep-ecg-serve .                 # bundle is copied from artifacts/
docker run -p 8000:8000 deep-ecg-serve
# serve another bundle without rebuilding:
docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" \
    -e MODEL_DIR=/app/artifacts/ensemble deep-ecg-serve
```

`POST /predict` takes a 12-lead signal as a `[12, L]` array (leads in canonical
order I, II, III, aVR, aVL, aVF, V1–V6) at any stated sampling rate; it is
resampled to 100 Hz and cropped or padded to the trained 10-second window with the
same preprocessing used in training, then returns the five superclass
probabilities and the thresholded multi-label decision. `GET /health` reports the
loaded model.

```bash
curl localhost:8000/health

python -c "import json, numpy as np; print(json.dumps(
    {'signal': np.random.randn(12, 1000).round(4).tolist(), 'sampling_rate': 100}))" \
  | curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @-
```

```json
{
  "probabilities": {"NORM": 0.91, "MI": 0.02, "STTC": 0.05, "CD": 0.03, "HYP": 0.01},
  "labels": {"NORM": true, "MI": false, "STTC": false, "CD": false, "HYP": false},
  "thresholds": {"NORM": 0.75, "MI": 0.15, "STTC": 0.2, "CD": 0.4, "HYP": 0.3},
  "model": "resnet1d-baseline",
  "backend": "onnx"
}
```

## References

- Wagner et al. *PTB-XL, a large publicly available electrocardiography dataset.*
  Scientific Data, 2020.
- Strodthoff et al. *Deep learning for ECG analysis: benchmarks and insights from
  PTB-XL.* IEEE JBHI, 2021.
- Reyna et al. *Will Two Do? Varying Dimensions in Electrocardiography: the
  PhysioNet/Computing in Cardiology Challenge 2021.* Computing in Cardiology, 2021.
