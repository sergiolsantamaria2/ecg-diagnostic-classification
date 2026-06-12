"""FastAPI service exposing the ECG classifier.

The bundle directory is read from the ``MODEL_DIR`` environment variable (default
``artifacts/resnet1d``) and the runtime backend from ``BACKEND`` (``auto``, ``onnx``
or ``torchscript``). The predictor is built once at startup and reused.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from .predict import Predictor
from .schemas import HealthResponse, PredictRequest, PredictResponse

MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/resnet1d")
BACKEND = os.environ.get("BACKEND", "auto")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor.from_dir(MODEL_DIR, backend=BACKEND)
    yield


app = FastAPI(
    title="deep-ecg",
    summary="12-lead ECG diagnostic classification (PTB-XL superclasses)",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    predictor: Predictor = app.state.predictor
    meta = predictor.meta
    return HealthResponse(
        status="ok",
        model=meta.name,
        backend=predictor.backend,
        classes=list(meta.classes),
        n_models=len(meta.models),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predictor: Predictor = app.state.predictor
    signal = np.asarray(request.signal, dtype=np.float32)
    try:
        result = predictor.predict(signal, request.sampling_rate)
    except ValueError as exc:  # invalid signal content (e.g. non-finite samples)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(
        backend=predictor.backend,
        model=predictor.meta.name,
        **result,
    )
