"""FastAPI wrapper around the local classifier.

    uvicorn api.main:app --reload

The model loads once at startup rather than per request. If it fails to load
the app still starts, so /health can explain what's wrong instead of the
whole service refusing to boot.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.gpu_check import collect_gpu_info
from src.inference import BeanClassifier, ModelNotFoundError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

state: dict = {"classifier": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["classifier"] = BeanClassifier()
    except ModelNotFoundError as exc:
        state["load_error"] = str(exc)
        log.warning("Model unavailable: %s", exc)
    yield
    state.clear()


app = FastAPI(
    title="Beans leaf classifier",
    description="Local ViT inference with GPU/CPU fallback.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    gpu = collect_gpu_info()
    classifier = state["classifier"]

    return {
        "status": "ok" if classifier else "degraded",
        "model_loaded": classifier is not None,
        "model_error": state["load_error"],
        "classes": classifier.labels if classifier else None,
        "device": str(classifier.device) if classifier else gpu["selected_device"],
        "cuda_available": gpu["cuda_available"],
        "gpu_name": gpu["devices"][0]["name"] if gpu["devices"] else None,
        "torch_version": gpu["torch_version"],
        "cudnn_version": gpu["cudnn_version"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    classifier = state["classifier"]
    if classifier is None:
        raise HTTPException(status_code=503, detail=state["load_error"])

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 10 MB")

    # Trust Pillow over the client's content-type header - browsers and curl
    # both get it wrong often enough that it isn't worth checking.
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Not a readable image")

    result = classifier.predict(image)
    result["filename"] = file.filename
    return result
