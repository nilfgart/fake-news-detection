"""Baseline и методы классификации на разных представлениях текста; замер качества, времени и памяти."""
import time
import tracemalloc

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE
from features import Word2VecVectorizer, make_bow_vectorizer, make_tfidf_vectorizer


def build_experiments():
    """Словарь: название эксперимента -> sklearn Pipeline (от сырого очищенного текста до предсказания)."""
    return {
        "baseline_majority": Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "bow_logreg": Pipeline([
            ("vec", make_bow_vectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "bow_naive_bayes": Pipeline([
            ("vec", make_bow_vectorizer()),
            ("clf", MultinomialNB()),
        ]),
        "tfidf_logreg": Pipeline([
            ("vec", make_tfidf_vectorizer()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "w2v_logreg": Pipeline([
            ("vec", Word2VecVectorizer()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
    }


def run_experiment(name, pipeline, X_train, y_train, X_test, y_test):
    """Обучает конвейер, замеряет время и пиковую память (tracemalloc), считает метрики."""
    tracemalloc.start()
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    predict_time = time.perf_counter() - t1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    proba_real = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "experiment": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "train_time_s": round(train_time, 2),
        "predict_time_s": round(predict_time, 2),
        "peak_memory_mb": round(peak / 1024 ** 2, 1),
    }
    return metrics, y_pred, proba_real
