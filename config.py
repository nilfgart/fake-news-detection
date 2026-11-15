"""Глобальные настройки проекта: пути, параметры разбиения и обучения."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SQL_DIR = RESULTS_DIR / "sql"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "news.db"

FAKE_CSV = RAW_DIR / "Fake.csv"
TRUE_CSV = RAW_DIR / "True.csv"
PROCESSED_PARQUET = PROCESSED_DIR / "articles.parquet"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CHUNK_SIZE = 10_000
MIN_WORDS = 5


def ensure_dirs():
    """Создаёт служебные папки, если их нет."""
    for folder in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, SQL_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)
