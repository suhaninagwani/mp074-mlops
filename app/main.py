"""FastAPI service for MP074: POST /predict and GET /health.

Run locally:   uvicorn app.main:app --host 0.0.0.0 --port 8000
The model is loaded from MODEL_PATH (default models/model.joblib).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from mp074.cleaning import clean_features
from mp074.logging_setup import setup_json_logging
from mp074.schemas import PredictRequest, PredictResponse

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "champion")

setup_json_logging()
log = logging.getLogger("mp074.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    log.info("model loaded", extra={"event": "startup", "model_path": str(MODEL_PATH)})
    yield


app = FastAPI(title="MP074 Parcel Delivery Days", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    return {"status": "ok", "model_loaded": getattr(request.app.state, "model", None) is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    started = time.perf_counter()
    features = clean_features(pd.DataFrame([body.model_dump()]))
    value = float(request.app.state.model.predict(features)[0])
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    # One JSON line per prediction. No personal details are logged.
    log.info(
        "prediction",
        extra={
            "event": "predict",
            "request_id": str(uuid.uuid4()),
            "service": body.service,
            "destination": body.destination,
            "predicted_days": round(value, 4),
            "model_version": MODEL_VERSION,
            "latency_ms": latency_ms,
        },
    )
    return PredictResponse(prediction=value)
