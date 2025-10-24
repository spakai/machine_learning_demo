from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "icecream_sales.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "icecream_regressor.joblib"
DEFAULT_TEST_TEMPERATURE = 25.0
