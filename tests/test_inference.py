"""Tests for the bits that break quietly.

The prediction tests need a trained checkpoint and skip without one, so the
suite is still useful on a machine that hasn't run training yet.
"""

from __future__ import annotations

import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src import config
from src.gpu_check import collect_gpu_info, select_device
from src.inference import BeanClassifier

MODEL_AVAILABLE = (config.MODEL_DIR / "config.json").exists()
needs_model = pytest.mark.skipif(
    not MODEL_AVAILABLE, reason="no trained model; run python -m src.train first"
)


def make_image(size=(300, 300), mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=(40, 120, 60) if mode == "RGB" else 128).save(
        buffer, format="JPEG"
    )
    return buffer.getvalue()


def test_device_selection_prefers_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device().type == "cuda"


def test_device_selection_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device().type == "cpu"


def test_gpu_info_reports_without_a_gpu(monkeypatch):
    # The report has to stay well-formed on a CPU-only machine, since that's
    # exactly when someone is reading it to find out what's wrong.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = collect_gpu_info()
    assert info["cuda_available"] is False
    assert info["devices"] == []
    assert info["torch_version"]


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_is_always_answerable(client):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["model_loaded"] is MODEL_AVAILABLE
    assert isinstance(body["cuda_available"], bool)


def test_predict_rejects_non_image(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"this is not a jpeg", "image/jpeg")}
    )
    assert response.status_code in {400, 503}


def test_predict_rejects_empty_upload(client):
    response = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code in {400, 503}


@needs_model
def test_predict_returns_expected_shape(client):
    response = client.post(
        "/predict", files={"file": ("leaf.jpg", make_image(), "image/jpeg")}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["predicted_class"] in body["probabilities"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert round(sum(body["probabilities"].values()), 2) == 1.0
    assert body["device"].startswith(("cuda", "cpu"))


@needs_model
def test_greyscale_image_does_not_crash():
    # Real uploads aren't always three-channel, and the processor won't
    # convert for us - hence the .convert("RGB") in BeanClassifier.predict.
    classifier = BeanClassifier()
    with Image.open(io.BytesIO(make_image(mode="L"))) as image:
        assert classifier.predict(image)["predicted_class"]


@needs_model
def test_cpu_fallback_produces_the_same_prediction():
    gpu_classifier = BeanClassifier()
    if gpu_classifier.device.type != "cuda":
        pytest.skip("no GPU to compare against")

    cpu_classifier = BeanClassifier(device=torch.device("cpu"))
    payload = make_image()
    with Image.open(io.BytesIO(payload)) as image:
        gpu_result = gpu_classifier.predict(image)
    with Image.open(io.BytesIO(payload)) as image:
        cpu_result = cpu_classifier.predict(image)

    assert gpu_result["predicted_class"] == cpu_result["predicted_class"]
    assert cpu_result["using_gpu"] is False
