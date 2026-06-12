"""Request and response schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

N_LEADS = 12


class PredictRequest(BaseModel):
    """A single 12-lead ECG to classify.

    ``signal`` is a ``[12, L]`` array of physical-unit samples (mV), one row per
    lead in canonical order (I, II, III, aVR, aVL, aVF, V1–V6). ``sampling_rate``
    is the rate of the submitted signal; it is resampled to the model's rate and
    cropped/padded to the trained length server-side.
    """

    signal: list[list[float]] = Field(..., description="[12, L] samples, canonical lead order")
    sampling_rate: int = Field(100, gt=0, description="sampling rate of the submitted signal (Hz)")

    @field_validator("signal")
    @classmethod
    def _check_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != N_LEADS:
            raise ValueError(f"signal must have {N_LEADS} leads, got {len(v)}")
        lengths = {len(row) for row in v}
        if len(lengths) != 1:
            raise ValueError("all leads must have the same number of samples")
        if lengths == {0}:
            raise ValueError("signal has no samples")
        return v


class PredictResponse(BaseModel):
    """Per-class probabilities and the thresholded multi-label decision."""

    probabilities: dict[str, float]
    labels: dict[str, bool]
    thresholds: dict[str, float]
    model: str
    backend: str


class HealthResponse(BaseModel):
    """Liveness and the loaded model's identity."""

    status: str
    model: str
    backend: str
    classes: list[str]
    n_models: int
