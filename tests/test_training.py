from pathlib import Path

import httpx
import numpy as np
import pytest

from src import train
from src.api import app, load_artifacts
from src.model_utils import load_dataset, load_model, save_model, train_regressor
from src.settings import DATA_PATH, MODEL_PATH

HOLDOUT_TEMPERATURES = np.array([52, 67, 70, 73, 78, 83], dtype=float).reshape(-1, 1)
HOLDOUT_SALES = np.array([0, 14, 23, 22, 26, 36], dtype=float)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_training_pipeline_creates_persisted_model(tmp_path: Path) -> None:
    x, y = load_dataset(DATA_PATH)
    artifacts = train_regressor(x, y)
    assert artifacts.r2 > 0.9
    model_path = tmp_path / "test_model.joblib"
    save_model(artifacts.model, model_path)
    loaded_model = load_model(model_path)
    sample_temperature = np.array([[25.0]])
    expected = artifacts.model.predict(sample_temperature)[0]
    actual = loaded_model.predict(sample_temperature)[0]
    assert actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.anyio
async def test_fastapi_endpoint_returns_prediction() -> None:
    # Ensure the default model is present for the API to load.
    train.main([])
    load_artifacts()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/predict", json={"temperature": 30.0})

    assert response.status_code == 200
    payload = response.json()
    assert "predicted_sales" in payload
    assert payload["temperature"] == pytest.approx(30.0)


def test_model_generalizes_to_holdout_examples() -> None:
    # Train with default dataset and ensure predictions stay close on held-out points.
    train.main([])
    model = load_model(MODEL_PATH)
    predictions = model.predict(HOLDOUT_TEMPERATURES)
    mae = np.abs(predictions - HOLDOUT_SALES)
    # Expect every holdout prediction to be within ~5 units, and average error to be small.
    assert np.all(mae < 5.0)
    assert mae.mean() < 3.0
