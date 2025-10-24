from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class RegressionArtifacts:
    model: LinearRegression
    r2: float
    rmse: float


def load_dataset(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the simple temperature -> sales dataset."""
    data = pd.read_csv(csv_path)
    if "temperature" not in data.columns or "sales" not in data.columns:
        raise ValueError("Dataset must contain 'temperature' and 'sales' columns")
    x = data[["temperature"]].astype(float).values
    y = data["sales"].astype(float).values
    return x, y


def train_regressor(x: np.ndarray, y: np.ndarray) -> RegressionArtifacts:
    """Fit a linear regression model and compute training metrics."""
    model = LinearRegression()
    model.fit(x, y)
    preds = model.predict(x)
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    return RegressionArtifacts(model=model, r2=r2, rmse=rmse)


def save_model(model: LinearRegression, model_path: Path) -> None:
    """Persist the trained model to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path) -> LinearRegression:
    """Load a persisted regression model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file missing at {model_path}. Run the training script first."
        )
    return joblib.load(model_path)
