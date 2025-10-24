from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from .model_utils import load_model
from .schemas import PredictionRequest, PredictionResponse
from .settings import MODEL_DIR, MODEL_PATH

logger = logging.getLogger(__name__)

model: Optional[Pipeline] = None
metrics: dict = {}


def read_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    raw = metrics_path.read_text()
    return json.loads(raw)


def load_artifacts() -> None:
    global model, metrics
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        logger.warning("Model file not found at %s. Run the training pipeline first.", MODEL_PATH)
        model = None
    metrics = read_metrics(MODEL_DIR / "metrics.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Ice-Cream Sales Regressor",
    version="0.1.0",
    description="Predict ice-cream sales from calendar and weather features.",
    lifespan=lifespan,
)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


def _prepare_features(payload: PredictionRequest) -> Tuple[pd.DataFrame, str, str]:
    resolved_day = payload.day_of_week
    resolved_month = payload.month
    frame = pd.DataFrame(
        [
            {
                "DayOfWeek": resolved_day,
                "Month": resolved_month,
                "Temperature": payload.temperature,
                "Rainfall": payload.rainfall,
            }
        ]
    )
    return frame, resolved_day, resolved_month


@app.post("/predict", response_model=PredictionResponse)
def predict_sales(payload: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    features, resolved_day, resolved_month = _prepare_features(payload)
    predicted = float(model.predict(features)[0])

    metrics_payload = {
        "train_r2": float(metrics.get("train_r2", 0.0)),
        "train_rmse": float(metrics.get("train_rmse", 0.0)),
        "test_r2": float(metrics.get("test_r2", 0.0)),
        "test_rmse": float(metrics.get("test_rmse", 0.0)),
    }

    return PredictionResponse(
        predicted_sales=predicted,
        temperature=payload.temperature,
        rainfall=payload.rainfall,
        day_of_week=resolved_day,
        month=resolved_month,
        model_version=MODEL_PATH.name,
        **metrics_payload,
    )
