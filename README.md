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

## Quickstart
1. Create and activate a virtual environment, then install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Train the regression model (writes `models/` artifacts):
   ```bash
   python -m src.train
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
- Tests cover both the training utilities and the FastAPI endpoint (loading the on-disk model).

## Next Steps
- Replace the sample CSV with your own data.
- Extend the FastAPI schema to handle confidence intervals or batch predictions.
- Wire the Docker image into your deployment platform of choice.
