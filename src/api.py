from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from sklearn.linear_model import LinearRegression

from .model_utils import load_model
from .schemas import PredictionRequest, PredictionResponse
from .settings import MODEL_DIR, MODEL_PATH

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ice-Cream Sales Regressor",
    version="0.1.0",
    description="Predict ice-cream sales from ambient temperature.",
)

model: Optional[LinearRegression] = None
metrics: dict = {}


def read_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    raw = metrics_path.read_text()
    return json.loads(raw)


@app.on_event("startup")
def load_artifacts() -> None:
    global model, metrics
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        logger.warning("Model file not found at %s. Run the training pipeline first.", MODEL_PATH)
        model = None
    metrics = read_metrics(MODEL_DIR / "metrics.json")


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_sales(payload: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    temperature_array = np.array([[payload.temperature]], dtype=float)
    predicted = model.predict(temperature_array)[0]
    return PredictionResponse(
        predicted_sales=float(predicted),
        temperature=payload.temperature,
        model_version=MODEL_PATH.name,
        r2=float(metrics.get("r2", 0.0)),
        rmse=float(metrics.get("rmse", 0.0)),
    )
