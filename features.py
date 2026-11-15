"""Представления текста: Bag of Words, TF-IDF и усреднённые векторы Word2Vec (embeddings)."""
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config import RANDOM_STATE


def make_bow_vectorizer():
    """Bag of Words: частоты слов и биграмм."""
    return CountVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3, dtype=np.float32)


def make_tfidf_vectorizer():
    """TF-IDF с логарифмическим масштабом частот (sublinear_tf)."""
    return TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                           sublinear_tf=True, dtype=np.float32)


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Среднее арифметическое векторов слов Word2Vec; модель обучается только на обучающей выборке."""

    def __init__(self, vector_size=100, window=5, min_count=3, epochs=5, seed=RANDOM_STATE):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y=None):
        sentences = [str(doc).split() for doc in X]
        self.w2v_ = Word2Vec(sentences=sentences, vector_size=self.vector_size, window=self.window,
                             min_count=self.min_count, epochs=self.epochs, seed=self.seed, workers=4)
        return self

    def transform(self, X):
        wv = self.w2v_.wv
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, doc in enumerate(X):
            vectors = [wv[w] for w in str(doc).split() if w in wv]
            if vectors:
                out[i] = np.mean(vectors, axis=0)
        return out


def extract_keywords(texts, top_k=5):
    """Для каждой статьи возвращает top_k ключевых слов по TF-IDF: (индекс документа, термин, вес)."""
    vec = TfidfVectorizer(max_features=20_000, min_df=5)
    matrix = vec.fit_transform(texts).tocsr()
    terms = vec.get_feature_names_out()
    rows = []
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        weights = matrix.data[start:end]
        idx = matrix.indices[start:end]
        if weights.size == 0:
            continue
        for j in np.argsort(weights)[::-1][:top_k]:
            rows.append((i, terms[idx[j]], float(weights[j])))
    return rows
