"""Точка входа: консольное меню проекта «Fake News Detection»."""
import json
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

import database as db
from config import (FIGURES_DIR, MIN_WORDS, MODELS_DIR, RANDOM_STATE, RESULTS_DIR, SQL_DIR,
                    TEST_SIZE, ensure_dirs)
from data_loader import clean_raw_frame, load_processed, load_raw_data, memory_usage_mb, save_processed
from eda import top_ngrams
from evaluation import (error_analysis, save_barh, save_confusion_matrix,
                        save_model_comparison, top_features)
from features import extract_keywords
from models import build_experiments, run_experiment
from preprocessing import (compare_normalization, ensure_nltk_resources, normalization_examples,
                           preprocess_series, preprocess_text)

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
META_PATH = MODELS_DIR / "best_model_meta.json"

MENU = """
========== Fake News Detection ==========
1 - Загрузить и подготовить данные (CSV -> Parquet -> SQLite)
2 - Обучить модели и сравнить методы (baseline, BoW, TF-IDF, Word2Vec)
3 - Анализ данных (частоты слов, стемминг и лемматизация)
4 - SQL-аналитика (SQLite -> pandas)
5 - Проверить новость (ввести текст, получить вердикт)
0 - Выход
"""


def step_prepare_data():
    """Пункт 1: читает CSV, чистит, лемматизирует, сохраняет Parquet и загружает SQLite."""
    print("Читаю Fake.csv и True.csv порциями...")
    df = clean_raw_frame(load_raw_data())
    print(f"Строк после очистки: {len(df)}")

    print("Предобработка текста (токенизация, стоп-слова, лемматизация)...")
    df["clean_text"] = preprocess_series(df["full_text"], method="lemma")
    df["n_words"] = df["clean_text"].str.split().str.len().fillna(0).astype("int32")
    df = df[df["n_words"] >= MIN_WORDS].reset_index(drop=True)
    save_processed(df)
    print(f"Parquet сохранён, объём в памяти: {memory_usage_mb(df)} МБ")

    print("Загрузка в SQLite и извлечение ключевых слов (TF-IDF)...")
    db.reset_database()
    db.load_articles(df)
    db.load_keywords(extract_keywords(df["clean_text"], top_k=5))
    print(f"Готово. База данных: {db.DB_PATH}")


def step_train_models():
    """Пункт 2: обучает baseline и методы, сохраняет метрики, графики, ошибки и лучшую модель."""
    df = load_processed()
    X, y = df["clean_text"], df["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)
    idx_te = X_te.index.to_numpy()
    df_test = df.loc[idx_te]

    experiments = build_experiments()
    results, outputs = [], {}
    for name, pipeline in experiments.items():
        print(f"Обучение: {name} ...")
        metrics, y_pred, proba = run_experiment(name, pipeline, X_tr, y_tr, X_te, y_te)
        results.append(metrics)
        outputs[name] = (y_pred, proba)
        db.save_experiment(metrics, idx_te + 1, y_te.to_numpy(), y_pred, proba)
        save_confusion_matrix(y_te, y_pred, name, FIGURES_DIR / f"confusion_{name}.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    save_model_comparison(results_df, FIGURES_DIR / "model_comparison.png")
    print(results_df.round(4).to_string(index=False))

    candidates = results_df[results_df["experiment"] != "baseline_majority"]
    best_name = candidates.sort_values("f1_macro", ascending=False).iloc[0]["experiment"]
    joblib.dump(experiments[best_name], BEST_MODEL_PATH)
    META_PATH.write_text(json.dumps({"best_model": best_name, "preprocessing": "lemma"},
                                    ensure_ascii=False, indent=2), encoding="utf-8")

    y_best, proba_best = outputs[best_name]
    errors = error_analysis(df_test, y_best, proba_best)
    for key, table in errors.items():
        table.to_csv(RESULTS_DIR / f"errors_{key}.csv", index=False)
    print("\nТипы ошибок лучшей модели:")
    print(errors["by_type"].to_string(index=False))

    fake_terms, real_terms = top_features(experiments["tfidf_logreg"], n=20)
    fake_terms.to_csv(RESULTS_DIR / "top_fake_terms_tfidf.csv", index=False)
    real_terms.to_csv(RESULTS_DIR / "top_real_terms_tfidf.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "best_model": best_name,
        "metrics": results,
    }
    (RESULTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nЛучшая модель: {best_name}, сохранена в {BEST_MODEL_PATH}")


def step_eda():
    """Пункт 3: частоты слов и биграмм по классам, сравнение нормализаций."""
    df = load_processed()
    print("Статистика по классам (число слов после очистки):")
    print(df.groupby("label")["n_words"].agg(["count", "mean", "median"]).round(1).to_string())
    print("\nТемы:")
    print(df["subject"].value_counts().to_string())

    for label, name in [(0, "fake"), (1, "real")]:
        texts = df.loc[df["label"] == label, "clean_text"]
        words = top_ngrams(texts, (1, 1), 20)
        bigrams = top_ngrams(texts, (2, 2), 20)
        words.to_csv(RESULTS_DIR / f"top_words_{name}.csv", index=False)
        save_barh(words, "term", "count", f"Топ-20 слов ({name})", FIGURES_DIR / f"top_words_{name}.png")
        save_barh(bigrams, "term", "count", f"Топ-20 биграмм ({name})",
                  FIGURES_DIR / f"top_bigrams_{name}.png")

    norm = compare_normalization(df["full_text"])
    norm.to_csv(RESULTS_DIR / "normalization_comparison.csv", index=False)
    print("\nСравнение нормализаций (выборка 2000 статей):")
    print(norm.to_string(index=False))
    print("\nПримеры стемминга и лемматизации:")
    print(normalization_examples().to_string(index=False))


def step_sql():
    """Пункт 4: выполняет SQL-запросы к SQLite и загружает результаты в pandas."""
    for key, (title, table) in db.run_analytics().items():
        table.to_csv(SQL_DIR / f"{key}.csv", index=False)
        print(f"\n== {title} ==")
        print(table.head(10).to_string(index=False))


def step_predict():
    """Пункт 5: пользователь вводит текст, модель выдаёт вердикт; запрос сохраняется в БД."""
    if not BEST_MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("модель не обучена, сначала выполните пункт 2")
    model = joblib.load(BEST_MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    print("Введите текст новости (пустая строка — назад):")
    text = input("> ").strip()
    if not text:
        return
    if len(text.split()) < MIN_WORDS:
        print("Текст очень короткий, точность может быть ниже.")

    clean = preprocess_text(text, method=meta["preprocessing"])
    proba_real = float(model.predict_proba([clean])[0, 1])
    label = int(proba_real >= 0.5)
    confidence = max(proba_real, 1 - proba_real)
    verdict = "Достоверная новость (real)" if label == 1 else "Фейк (fake)"
    print(f"\nВердикт: {verdict}")
    print(f"Уверенность: {confidence:.1%} (модель: {meta['best_model']})")
    db.save_user_query(text, label, confidence)


ACTIONS = {"1": step_prepare_data, "2": step_train_models, "3": step_eda,
           "4": step_sql, "5": step_predict}


def main():
    """Главный цикл меню: выбор пункта, обработка ошибок, выход."""
    ensure_dirs()
    ensure_nltk_resources()
    try:
        while True:
            print(MENU)
            choice = input("Выберите пункт: ").strip()
            if choice == "0":
                print("Работа завершена.")
                break
            action = ACTIONS.get(choice)
            if action is None:
                print("Такого пункта нет, попробуйте снова.")
                continue
            try:
                action()
            except FileNotFoundError as err:
                print(f"[Ошибка файла] {err}")
            except (sqlite3.Error, ValueError) as err:
                print(f"[Ошибка данных] {err}")
    except (KeyboardInterrupt, EOFError):
        print("\nВыход из программы.")


if __name__ == "__main__":
    main()
