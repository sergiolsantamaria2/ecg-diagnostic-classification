"""Smoke tests for the FastAPI endpoints against a tiny bundle."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from deep_ecg.data.labels import SUPERCLASSES


@pytest.fixture
def client(tiny_bundle, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tiny_bundle))
    monkeypatch.setenv("BACKEND", "auto")
    # Reimport so the module-level MODEL_DIR/BACKEND pick up the patched env.
    from deep_ecg.serving import api

    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["classes"] == list(SUPERCLASSES)


def test_predict_synthetic_signal(client):
    signal = np.random.randn(12, 1000).tolist()
    response = client.post("/predict", json={"signal": signal, "sampling_rate": 100})
    assert response.status_code == 200
    body = response.json()
    assert set(body["probabilities"]) == set(SUPERCLASSES)
    assert set(body["labels"]) == set(SUPERCLASSES)
    for p in body["probabilities"].values():
        assert 0.0 <= p <= 1.0
    assert all(isinstance(v, bool) for v in body["labels"].values())


def test_predict_rejects_wrong_lead_count(client):
    signal = np.random.randn(11, 1000).tolist()
    response = client.post("/predict", json={"signal": signal, "sampling_rate": 100})
    assert response.status_code == 422


def test_predict_accepts_other_sampling_rate(client):
    signal = np.random.randn(12, 5000).tolist()  # 500 Hz, 10 s
    response = client.post("/predict", json={"signal": signal, "sampling_rate": 500})
    assert response.status_code == 200
    assert len(response.json()["probabilities"]) == len(SUPERCLASSES)
