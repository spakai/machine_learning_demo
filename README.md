# Ice-Cream Sales Regressor
- Predict daily ice-cream sales from a single feature: ambient temperature.
- Training pipeline powered by `scikit-learn`, packaged behind a FastAPI service.
- GitHub Actions CI trains, tests, builds the Docker image, and (optionally) calls GitHub Models for reporting.

## Project Layout
- `data/icecream_sales.csv` – sample dataset used for initial training.
- `src/train.py` – command-line training entrypoint.
- `src/api.py` – FastAPI app exposing `/predict`.
- `scripts/github_models_report.py` – optional GitHub Models helper that turns metrics into prose.
- `Dockerfile` – builds a ready-to-serve container that trains during the image build.
- `.github/workflows/ci.yml` – CI workflow (tests, Docker build, GitHub Models summary).

The datasets and regression setup follow the Microsoft Learn module on regression fundamentals, and the hold-out examples come from the same guide:
- Tutorial: https://learn.microsoft.com/en-us/training/modules/fundamentals-machine-learning/4-regression?pivots=text

## Quickstart
1. Create and activate a virtual environment, then install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Train the regression model (writes `models/` artifacts) using the updated sales data:
   ```bash
   python -m src.train
   ```
   Sample dataset (`data/icecream_sales.csv`):
   ```
   temperature,sales
   51,1
   65,14
   69,20
   72,23
   75,26
   81,30
   ```
3. Launch the FastAPI server:
   ```bash
   uvicorn src.api:app --reload
   ```
4. Query the API:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"temperature": 30}'
   ```

## Docker Workflow
- Build the image (training runs during the build step):
  ```bash
  docker build -t icecream-sales .
  ```
- Run the container:
  ```bash
  docker run -p 8000:8000 icecream-sales
  ```

## GitHub Actions CI
- Runs on every push/PR:
  - Installs dependencies, trains the model, and executes `pytest`.
  - Builds the Docker image to ensure containerization stays healthy.
  - Optionally invokes GitHub Models (see next section) to produce a summary.
- Configure repository secrets for private runners if needed (no secrets needed for the base pipeline).

## GitHub Models Integration (Optional)
- Generate a natural-language training report with the GitHub Models inference API:
  1. Create a fine-grained personal access token with **Models** scope.
  2. Store it as `GITHUB_MODELS_TOKEN` in your environment (locally) or repository secrets (for CI).
  3. Run the helper script after training:
     ```bash
     python scripts/github_models_report.py
     ```
  4. On CI the `github-models-report` job will automatically run when the secret is present.

## Testing
- With the virtual environment active, execute:
  ```bash
  pytest
  ```
- Tests cover the training utilities, API endpoint, and now validate on held-out examples.

Hold-out validation data (used by tests to sanity-check the model):
```
temperature,sales
52,0
67,14
70,23
73,22
78,26
83,36
```

## Metrics at a Glance
- Training prints `r2` (coefficient of determination) and `rmse` (root mean square error).  
  - `r2` ranges up to 1.0, with higher meaning the model explains more variance in the data.  
  - `rmse` stays in sales units; lower values mean smaller average prediction errors.
- Example run:  
  ```bash
  $ python -m src.train
  Model trained and saved to .../models/icecream_regressor.joblib
  {
    "r2": 0.990987388929917,
    "rmse": 0.8972884647243659
  }
  ```
  These numbers show the linear fit matches the new training data closely. The hold-out check keeps you aware of generalization performance.

## Workflow Summary
- Retrain: `python -m src.train`
- Evaluate locally: `pytest` (re-runs training, API smoke test, and hold-out validation)
- Serve predictions: `uvicorn src.api:app --reload`
- Docker path: `docker build …` then `docker run …`
- Optional GitHub Models summary: `python scripts/github_models_report.py` once you set `GITHUB_MODELS_TOKEN`.

## Dependencies
- `fastapi==0.110.0` – async web framework powering the prediction API.
- `uvicorn[standard]==0.27.1` – ASGI server (the `[standard]` extra adds uvloop/watchgod for speed and reloads).
- `pandas==2.2.1` – CSV/dataframe handling to load and preprocess training data.
- `scikit-learn==1.4.1.post1` – provides the linear regression model and evaluation utilities.
- `joblib==1.3.2` – serializes the trained scikit-learn model to disk.
- `numpy==1.26.4` – numerical array operations underpinning pandas and scikit-learn.
- `httpx==0.27.0` – async HTTP client used in tests and for the optional GitHub Models helper.
- `pytest==8.1.1` – test runner that validates the training pipeline and FastAPI endpoints.

## Next Steps
- Replace the sample CSV with your own data.
- Extend the FastAPI schema to handle confidence intervals or batch predictions.
- Wire the Docker image into your deployment platform of choice.
