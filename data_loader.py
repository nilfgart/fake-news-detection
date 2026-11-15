"""Загрузка исходных CSV по частям, очистка и сохранение в Parquet."""
import re

import pandas as pd

from config import CHUNK_SIZE, FAKE_CSV, PROCESSED_PARQUET, TRUE_CSV

# Префикс агентства вида "WASHINGTON (Reuters) - " присутствует почти только в настоящих новостях.
# Его удаление снижает утечку признака источника (data leakage).
REUTERS_PREFIX = re.compile(r"^[^()]{0,60}\(Reuters\)\s*-\s*")


def read_csv_in_chunks(path):
    """Читает CSV порциями (chunksize), все значения как строки, пустые ячейки сохраняются как ''."""
    reader = pd.read_csv(path, chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False)
    return pd.concat(list(reader), ignore_index=True)


def load_raw_data():
    """Читает Fake.csv и True.csv и добавляет метку класса: 0 = fake, 1 = real."""
    try:
        fake = read_csv_in_chunks(FAKE_CSV)
        real = read_csv_in_chunks(TRUE_CSV)
    except FileNotFoundError as err:
        raise FileNotFoundError("положите Fake.csv и True.csv в папку data/raw/") from err
    fake["label"] = 0
    real["label"] = 1
    return pd.concat([fake, real], ignore_index=True)


def clean_raw_frame(df):
    """Удаляет дубликаты и пустые записи, убирает префикс агентства, готовит поля."""
    df = df.drop_duplicates(subset=["title", "text"]).copy()
    df["title"] = df["title"].str.strip()
    df["text"] = df["text"].str.strip().str.replace(REUTERS_PREFIX, "", regex=True)
    df["full_text"] = (df["title"] + " " + df["text"]).str.strip()
    df = df[df["full_text"].str.len() > 0].copy()
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["subject"] = df["subject"].astype("category")
    df["label"] = df["label"].astype("int8")
    return df.reset_index(drop=True)


def save_processed(df):
    """Сохраняет подготовленный корпус в Parquet (компактный колоночный формат)."""
    PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PARQUET, index=False)


def load_processed():
    """Загружает подготовленный корпус из Parquet."""
    if not PROCESSED_PARQUET.exists():
        raise FileNotFoundError("нет подготовленных данных, сначала выполните пункт 1")
    return pd.read_parquet(PROCESSED_PARQUET)


def memory_usage_mb(df):
    """Оценивает объём DataFrame в памяти (МБ) для отчёта."""
    return round(df.memory_usage(deep=True).sum() / 1024 ** 2, 1)
