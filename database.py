"""SQLite: схема (ER-модель), загрузка данных, сохранение экспериментов и аналитические SQL-запросы."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

from config import DB_PATH

TABLES = ["predictions", "keywords", "user_queries", "experiments", "articles", "subjects"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    subject_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS articles (
    article_id     INTEGER PRIMARY KEY,
    title          TEXT,
    full_text      TEXT,
    clean_text     TEXT,
    label          INTEGER NOT NULL CHECK (label IN (0, 1)),
    subject_id     INTEGER REFERENCES subjects(subject_id),
    published_date TEXT,
    n_words        INTEGER
);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id    INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    accuracy         REAL,
    f1_macro         REAL,
    precision_macro  REAL,
    recall_macro     REAL,
    train_time_s     REAL,
    predict_time_s   REAL,
    peak_memory_mb   REAL,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   INTEGER PRIMARY KEY,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    article_id      INTEGER NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    predicted_label INTEGER NOT NULL,
    confidence      REAL,
    is_correct      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS keywords (
    keyword_id INTEGER PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES articles(article_id) ON DELETE CASCADE,
    term       TEXT NOT NULL,
    weight     REAL
);
CREATE TABLE IF NOT EXISTS user_queries (
    query_id        INTEGER PRIMARY KEY,
    query_text      TEXT NOT NULL,
    predicted_label INTEGER,
    confidence      REAL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_subject ON articles(subject_id);
CREATE INDEX IF NOT EXISTS idx_pred_experiment  ON predictions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_pred_article     ON predictions(article_id);
CREATE INDEX IF NOT EXISTS idx_keywords_term    ON keywords(term);
"""

ANALYTIC_QUERIES = {
    "q01_subject_label_stats": ("JOIN + GROUP BY: статистика по темам и классам", """
        SELECT s.name AS subject, a.label, COUNT(*) AS articles,
               ROUND(AVG(a.n_words), 1) AS avg_words
        FROM articles a
        JOIN subjects s ON s.subject_id = a.subject_id
        GROUP BY s.name, a.label
        ORDER BY s.name, a.label;"""),

    "q02_fake_heavy_subjects": ("GROUP BY + HAVING: темы, где фейков больше половины", """
        SELECT s.name AS subject, COUNT(*) AS total,
               SUM(a.label = 0) AS fake_count,
               ROUND(100.0 * SUM(a.label = 0) / COUNT(*), 1) AS fake_share_pct
        FROM articles a
        JOIN subjects s ON s.subject_id = a.subject_id
        GROUP BY s.name
        HAVING COUNT(*) >= 100 AND SUM(a.label = 0) * 1.0 / COUNT(*) > 0.5
        ORDER BY fake_share_pct DESC;"""),

    "q03_long_articles_subquery": ("Подзапрос: длинные статьи из тем, где преобладают реальные новости", """
        SELECT a.article_id, a.title, a.label, a.n_words
        FROM articles a
        WHERE a.subject_id IN (
            SELECT subject_id FROM articles
            GROUP BY subject_id
            HAVING AVG(label) > 0.5
        )
          AND a.n_words > (SELECT AVG(n_words) FROM articles)
        ORDER BY a.n_words DESC
        LIMIT 20;"""),

    "q04_monthly_trend_cte_window": ("CTE + оконная функция: динамика доли фейков по месяцам", """
        WITH monthly AS (
            SELECT substr(published_date, 1, 7) AS month,
                   COUNT(*) AS total,
                   SUM(label = 0) AS fake_count
            FROM articles
            WHERE published_date IS NOT NULL
            GROUP BY month
        )
        SELECT month, total, fake_count,
               ROUND(100.0 * fake_count / total, 1) AS fake_pct,
               ROUND(AVG(100.0 * fake_count / total)
                     OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 1) AS fake_pct_ma3
        FROM monthly
        WHERE total >= 20
        ORDER BY month;"""),

    "q05_subject_rank_window": ("CTE + RANK() OVER: ранг тем внутри каждого класса", """
        WITH counts AS (
            SELECT s.name AS subject, a.label, COUNT(*) AS n
            FROM articles a
            JOIN subjects s USING (subject_id)
            GROUP BY s.name, a.label
        )
        SELECT subject, label, n,
               RANK() OVER (PARTITION BY label ORDER BY n DESC) AS rank_in_class
        FROM counts
        ORDER BY label, rank_in_class;"""),

    "q06_model_leaderboard": ("Подзапрос + оконная функция: таблица методов по F1 (последний запуск)", """
        SELECT name, ROUND(f1_macro, 4) AS f1_macro, ROUND(accuracy, 4) AS accuracy,
               train_time_s, peak_memory_mb,
               RANK() OVER (ORDER BY f1_macro DESC) AS f1_rank,
               ROUND(f1_macro - MAX(f1_macro) OVER (), 4) AS gap_to_best
        FROM experiments
        WHERE experiment_id IN (SELECT MAX(experiment_id) FROM experiments GROUP BY name)
        ORDER BY f1_rank;"""),

    "q07_error_rate_by_subject": ("JOIN x4 + GROUP BY + HAVING: доля ошибок по темам для каждой модели", """
        SELECT s.name AS subject, e.name AS model,
               COUNT(*) AS predictions,
               SUM(1 - p.is_correct) AS errors,
               ROUND(100.0 * SUM(1 - p.is_correct) / COUNT(*), 2) AS error_pct
        FROM predictions p
        JOIN experiments e ON e.experiment_id = p.experiment_id
        JOIN articles a    ON a.article_id = p.article_id
        JOIN subjects s    ON s.subject_id = a.subject_id
        WHERE e.experiment_id IN (SELECT MAX(experiment_id) FROM experiments GROUP BY name)
        GROUP BY s.name, e.name
        HAVING COUNT(*) >= 50
        ORDER BY error_pct DESC
        LIMIT 30;"""),

    "q08_confusion_best_model": ("GROUP BY + подзапрос: матрица ошибок лучшей модели из БД", """
        SELECT e.name AS model,
               a.label AS true_label,
               p.predicted_label,
               COUNT(*) AS n
        FROM predictions p
        JOIN articles a    ON a.article_id = p.article_id
        JOIN experiments e ON e.experiment_id = p.experiment_id
        WHERE e.experiment_id = (SELECT experiment_id FROM experiments
                                 ORDER BY f1_macro DESC LIMIT 1)
        GROUP BY e.name, a.label, p.predicted_label
        ORDER BY a.label, p.predicted_label;"""),

    "q09_fake_vocabulary_cte": ("CTE + JOIN + HAVING: слова, характерные для фейков", """
        WITH term_stats AS (
            SELECT k.term,
                   SUM(a.label = 0) AS in_fake,
                   SUM(a.label = 1) AS in_real,
                   COUNT(*) AS total
            FROM keywords k
            JOIN articles a ON a.article_id = k.article_id
            GROUP BY k.term
            HAVING COUNT(*) >= 200
        )
        SELECT term, in_fake, in_real, total,
               ROUND(1.0 * in_fake / total, 3) AS fake_ratio
        FROM term_stats
        ORDER BY fake_ratio DESC, total DESC
        LIMIT 25;"""),

    "q10_not_in_test_not_exists": ("NOT EXISTS: статьи, не попавшие в тестовую выборку лучшего запуска", """
        SELECT a.label, COUNT(*) AS train_articles
        FROM articles a
        WHERE NOT EXISTS (
            SELECT 1 FROM predictions p
            WHERE p.article_id = a.article_id
              AND p.experiment_id = (SELECT MAX(experiment_id) FROM experiments)
        )
        GROUP BY a.label;"""),
}

SCHEMA_EXPERIMENTS_NOTE = "experiments: один запуск модели; predictions: прогноз по каждой статье тестовой выборки"


@contextmanager
def db_session():
    """Открывает соединение с SQLite с включёнными внешними ключами и закрывает его по выходу."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def reset_database():
    """Удаляет старые таблицы и создаёт схему заново."""
    with db_session() as conn:
        for table in TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.executescript(SCHEMA)


def load_articles(df):
    """Загружает справочник тем и статьи (связь subjects 1 — N articles)."""
    subjects = sorted(df["subject"].astype(str).unique())
    subject_ids = {name: i for i, name in enumerate(subjects, start=1)}
    dates = df["date_parsed"].dt.strftime("%Y-%m-%d").where(df["date_parsed"].notna(), None)
    article_rows = list(zip(
        range(1, len(df) + 1),
        df["title"],
        df["full_text"],
        df["clean_text"],
        df["label"].astype(int),
        df["subject"].astype(str).map(subject_ids).astype(int),
        dates,
        df["n_words"].astype(int),
    ))
    with db_session() as conn:
        conn.executemany("INSERT INTO subjects (subject_id, name) VALUES (?, ?)",
                         [(i, n) for n, i in subject_ids.items()])
        conn.executemany("INSERT INTO articles (article_id, title, full_text, clean_text, label, "
                         "subject_id, published_date, n_words) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         article_rows)


def load_keywords(rows):
    """Сохраняет ключевые слова: rows = (индекс статьи с 0, термин, вес)."""
    payload = [(i + 1, term, weight) for i, term, weight in rows]
    with db_session() as conn:
        conn.executemany("INSERT INTO keywords (article_id, term, weight) VALUES (?, ?, ?)", payload)


def save_experiment(metrics, article_ids, y_true, y_pred, proba_real):
    """Сохраняет метрики запуска и прогнозы по каждой статье тестовой выборки."""
    now = datetime.now().isoformat(timespec="seconds")
    confidence = np.maximum(proba_real, 1 - proba_real)
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO experiments (name, accuracy, f1_macro, precision_macro, recall_macro, "
            "train_time_s, predict_time_s, peak_memory_mb, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (metrics["experiment"], float(metrics["accuracy"]), float(metrics["f1_macro"]),
             float(metrics["precision_macro"]), float(metrics["recall_macro"]),
             metrics["train_time_s"], metrics["predict_time_s"], metrics["peak_memory_mb"], now),
        )
        exp_id = cur.lastrowid
        rows = [(exp_id, int(a), int(p), float(c), int(p == t))
                for a, p, c, t in zip(article_ids, y_pred, confidence, y_true)]
        conn.executemany(
            "INSERT INTO predictions (experiment_id, article_id, predicted_label, confidence, is_correct) "
            "VALUES (?, ?, ?, ?, ?)", rows)
    return exp_id


def save_user_query(text, label, confidence):
    """Сохраняет запрос пользователя и результат проверки."""
    with db_session() as conn:
        conn.execute(
            "INSERT INTO user_queries (query_text, predicted_label, confidence, created_at) VALUES (?, ?, ?, ?)",
            (text, int(label), float(confidence), datetime.now().isoformat(timespec="seconds")))


def run_analytics():
    """Выполняет все аналитические запросы; возвращает {ключ: (название, DataFrame)}."""
    if not DB_PATH.exists():
        raise FileNotFoundError("база данных не создана, сначала выполните пункт 1")
    with db_session() as conn:
        return {key: (title, pd.read_sql_query(sql, conn))
                for key, (title, sql) in ANALYTIC_QUERIES.items()}
