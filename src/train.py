from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_utils import load_dataset, save_model, train_regressor
from .settings import DATA_PATH, MODEL_DIR, MODEL_PATH


def write_metrics(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear regression model for ice-cream sales prediction."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Where to persist the trained model.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=MODEL_DIR / "metrics.json",
        help="Optional path to save metrics summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    x, y = load_dataset(args.data_path)
    artifacts = train_regressor(x, y)
    save_model(artifacts.model, args.model_path)
    metrics = {"r2": artifacts.r2, "rmse": artifacts.rmse}
    write_metrics(metrics, args.metrics_path)
    print(f"Model trained and saved to {args.model_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
