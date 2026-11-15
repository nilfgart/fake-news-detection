"""Метрики, матрица ошибок, графики и анализ ошибок модели."""
import matplotlib

matplotlib.use("Agg")  # графики сохраняются в файлы, без окна (подходит для консоли и Colab)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402

LABEL_NAMES = ["fake", "real"]


def save_confusion_matrix(y_true, y_pred, title, path):
    """Строит и сохраняет матрицу ошибок; возвращает массив cm."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm


def save_model_comparison(results_df, path):
    """Столбчатая диаграмма F1-macro для всех экспериментов."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(results_df["experiment"], results_df["f1_macro"], color="steelblue")
    ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1.05)
    ax.set_title("Сравнение методов на тестовой выборке")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_barh(df, label_col, value_col, title, path):
    """Горизонтальная столбчатая диаграмма (для частот слов и весов признаков)."""
    data = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(df) + 1)))
    ax.barh(data[label_col].astype(str), data[value_col], color="darkorange")
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def error_analysis(df_test, y_pred, proba_real, top_n=15):
    """Разбор ошибок: по типу, теме, длине текста и самые уверенные промахи модели."""
    work = df_test[["title", "subject", "label", "clean_text"]].copy()
    work["pred"] = np.asarray(y_pred)
    work["proba_real"] = np.asarray(proba_real)
    work["n_words"] = work["clean_text"].str.split().str.len()
    work["confidence"] = np.maximum(work["proba_real"], 1 - work["proba_real"])
    work["is_error"] = work["pred"] != work["label"]
    work["error_type"] = np.select(
        [(work["label"] == 0) & (work["pred"] == 1), (work["label"] == 1) & (work["pred"] == 0)],
        ["fake_as_real", "real_as_fake"],
        default="correct",
    )
    work["length_bin"] = pd.cut(work["n_words"], bins=[0, 100, 300, 600, np.inf],
                                labels=["<100", "100-300", "300-600", ">600"])

    by_type = (work["error_type"].value_counts().rename_axis("error_type")
               .reset_index(name="count"))
    by_subject = (work.groupby("subject", observed=True)["is_error"]
                  .agg(error_rate="mean", errors="sum", total="count")
                  .reset_index().sort_values("error_rate", ascending=False))
    by_length = (work.groupby("length_bin", observed=True)["is_error"]
                 .agg(error_rate="mean", total="count").reset_index())
    confident_errors = (work[work["is_error"]]
                        .sort_values("confidence", ascending=False)
                        .head(top_n)[["title", "label", "pred", "confidence", "subject"]])
    return {"by_type": by_type, "by_subject": by_subject,
            "by_length": by_length, "confident_errors": confident_errors}


def top_features(pipeline, n=20):
    """Слова, сильнее всего смещающие решение к fake и к real (для LogisticRegression)."""
    names = pipeline.named_steps["vec"].get_feature_names_out()
    coef = pipeline.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    fake_terms = pd.DataFrame({"term": names[order[:n]], "weight": coef[order[:n]]})
    real_terms = pd.DataFrame({"term": names[order[::-1][:n]], "weight": coef[order[::-1][:n]]})
    return fake_terms, real_terms
