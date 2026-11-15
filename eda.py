"""Частотный анализ: топ слов и n-грамм по классам."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def top_ngrams(texts, ngram_range=(1, 1), top_n=20):
    """Возвращает самые частотные n-граммы корпуса как DataFrame (term, count)."""
    vec = CountVectorizer(ngram_range=ngram_range, min_df=5, max_features=50_000)
    matrix = vec.fit_transform(texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    top = np.argsort(counts)[::-1][:top_n]
    names = vec.get_feature_names_out()
    return pd.DataFrame({"term": names[top], "count": counts[top]})
