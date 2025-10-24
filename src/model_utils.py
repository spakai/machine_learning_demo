from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class RegressionArtifacts:
    model: Pipeline
    train_r2: float
    train_rmse: float
    test_r2: float
    test_rmse: float


REQUIRED_COLUMNS = {
    "DayOfWeek",
    "Month",
    "Temperature",
    "Rainfall",
    "IceCreamsSold",
}


def load_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the multi-feature ice-cream sales dataset."""
    data = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    data = data.copy()
    data["DayOfWeek"] = data["DayOfWeek"].astype(str)
    data["Month"] = data["Month"].astype(str)
    data["Temperature"] = data["Temperature"].astype(float)
    data["Rainfall"] = data["Rainfall"].astype(float)
    data["IceCreamsSold"] = data["IceCreamsSold"].astype(float)

    features = data[[
        "DayOfWeek",
        "Month",
        "Temperature",
        "Rainfall",
    ]]
    target = data["IceCreamsSold"]
    return features, target


def _split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    split_index = int(len(x) * train_ratio)
    if split_index == 0 or split_index == len(x):
        raise ValueError("Dataset too small to split with the requested ratio")

    x_train = x.iloc[:split_index]
    x_test = x.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    return x_train, x_test, y_train, y_test


def train_regressor(
    x: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
) -> RegressionArtifacts:
    """Fit a regression model with preprocessing and compute train/test metrics."""

    x_train, x_test, y_train, y_test = _split_train_test(x, y, train_ratio=train_ratio)

    categorical_features = ["DayOfWeek", "Month"]
    numeric_features = [col for col in x.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    pipeline.fit(x_train, y_train)

    train_predictions = pipeline.predict(x_train)
    test_predictions = pipeline.predict(x_test)

    train_r2 = r2_score(y_train, train_predictions)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_predictions)))
    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_predictions)))

    return RegressionArtifacts(
        model=pipeline,
        train_r2=float(train_r2),
        train_rmse=train_rmse,
        test_r2=float(test_r2),
        test_rmse=test_rmse,
    )


def save_model(model: Pipeline, model_path: Path) -> None:
    """Persist the trained model to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path) -> Pipeline:
    """Load a persisted regression model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file missing at {model_path}. Run the training script first."
        )
    return joblib.load(model_path)
