# Ice-Cream Sales Regressor
- Predict daily ice-cream sales from four drivers: day of week, month, temperature, and rainfall.
- Training pipeline powered by `scikit-learn`, packaged behind a FastAPI service.
- GitHub Actions CI trains, tests, builds the Docker image, and (optionally) calls GitHub Models for reporting.

## Project Layout
- `data/icecream_sales.csv` – sample dataset used for initial training (DayOfWeek, Month, Temperature, Rainfall, IceCreamsSold).
- `src/train.py` – command-line training entrypoint.
- `src/api.py` – FastAPI app exposing `/predict`.
- `scripts/github_models_report.py` – optional GitHub Models helper that turns metrics into prose.
- `Dockerfile` – builds a ready-to-serve container that trains during the image build.
- `.github/workflows/ci.yml` – CI workflow (tests, Docker build, GitHub Models summary).

The dataset and regression setup are adapted from the Microsoft Learn AI fundamentals exercise:
- Exercise: https://microsoftlearning.github.io/mslearn-ai-fundamentals/Instructions/Exercises/01-machine-learning.html
- Background reading: https://learn.microsoft.com/en-us/training/modules/fundamentals-machine-learning/4-regression?pivots=text

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
   DayOfWeek,Month,Temperature,Rainfall,IceCreamsSold
   Tuesday,April,59.4,0.74,61
   Thursday,April,53.6,0.28,33
   Sunday,April,51.4,0.14,21
   Monday,April,50.8,0.06,23
   Tuesday,April,57.4,0.79,51
   Wednesday,April,59.9,0.25,73
   ```
3. Launch the FastAPI server:
   ```bash
   uvicorn src.api:app --reload
   ```
4. Query the API (provide the categorical context explicitly):
    ```bash
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
     -d '{
       "day_of_week": "Friday",
       "month": "July",
       "temperature": 84.0,
       "rainfall": 0.3
     }'
    ```
   Sample response:
   ```json
   {
     "predicted_sales": 187.06131603708448,
     "temperature": 84.0,
     "rainfall": 0.3,
     "day_of_week": "Friday",
     "month": "July",
     "model_version": "icecream_regressor.joblib",
     "train_r2": 0.9934421121757082,
     "train_rmse": 4.833032939425899,
     "test_r2": 0.9847831112058534,
     "test_rmse": 5.144147563133475
   }
   ```
   Response fields:
   - `predicted_sales` – point estimate for daily sales given the supplied context (≈187 cones).
   - `temperature`, `rainfall`, `day_of_week`, `month` – echo the inputs so you can confirm which observation was scored.
   - `model_version` – trained model artifact that produced the prediction (`icecream_regressor.joblib`).
   - `train_r2`/`train_rmse` – goodness-of-fit on the 70% training slice (higher R², lower RMSE are better).
   - `test_r2`/`test_rmse` – same metrics on the unseen 30% hold-out slice, showing generalization.

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
  - Installs dependencies, trains the model with a chronological 70/30 split, and executes `pytest`.
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

Hold-out validation uses the final 30% of `data/icecream_sales.csv` to ensure the model generalizes to unseen records.

## Metrics at a Glance
- Training prints `train_r2`, `train_rmse`, `test_r2`, and `test_rmse`.  
  - `*_r2` range up to 1.0, with higher scores indicating more variance explained.  
  - `*_rmse` stay in sales units; lower numbers mean smaller average errors.
- Example run:  
  ```bash
  $ python -m src.train
  Model trained and saved to .../models/icecream_regressor.joblib
  {
    "train_r2": 0.98,
    "train_rmse": 8.84,
    "test_r2": 0.95,
    "test_rmse": 12.31
  }
  ```
  These numbers show how the model performs on both the training slice and the unseen 30% hold-out partition.

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
