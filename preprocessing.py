"""Предобработка текста: очистка, токенизация, стоп-слова, стемминг и лемматизация (NLTK)."""
import re
import time
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKENIZER = RegexpTokenizer(r"[a-z]+")
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = None


def ensure_nltk_resources():
    """Скачивает ресурсы NLTK (стоп-слова, WordNet), если их ещё нет на машине."""
    resources = {"stopwords": "corpora/stopwords", "wordnet": "corpora/wordnet", "omw-1.4": "corpora/omw-1.4"}
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def get_stopwords():
    """Возвращает английские стоп-слова (ленивая загрузка)."""
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = frozenset(stopwords.words("english"))
    return _STOPWORDS


@lru_cache(maxsize=200_000)
def stem_word(word):
    """Порter-стемминг одного слова (с кэшем, чтобы ускорить обработку корпуса)."""
    return STEMMER.stem(word)


@lru_cache(maxsize=200_000)
def lemmatize_word(word):
    """Лемматизация одного слова по WordNet (с кэшем)."""
    return LEMMATIZER.lemmatize(word)


def tokenize(text):
    """Приводит текст к нижнему регистру, удаляет ссылки и разбивает на слова."""
    return TOKENIZER.tokenize(URL_RE.sub(" ", text.lower()))


def remove_stopwords(tokens):
    """Убирает стоп-слова и слишком короткие токены."""
    stop = get_stopwords()
    return [t for t in tokens if t not in stop and len(t) > 2]


def normalize_tokens(tokens, method):
    """Применяет выбранную нормализацию: 'none', 'stem' или 'lemma'."""
    if method == "none":
        return tokens
    if method == "stem":
        return [stem_word(t) for t in tokens]
    if method == "lemma":
        return [lemmatize_word(t) for t in tokens]
    raise ValueError(f"неизвестный метод нормализации: {method}")


def preprocess_text(text, method="lemma"):
    """Полный конвейер для одного текста: токенизация, стоп-слова, нормализация."""
    tokens = remove_stopwords(tokenize(str(text)))
    return " ".join(normalize_tokens(tokens, method))


def preprocess_series(texts, method="lemma"):
    """Предобрабатывает последовательность текстов."""
    return [preprocess_text(t, method) for t in texts]


def compare_normalization(texts, sample_size=2000):
    """Сравнивает варианты: без нормализации, стемминг и лемматизация (время и размер словаря)."""
    sample = list(texts)[:sample_size]
    rows = []
    for method in ("none", "stem", "lemma"):
        start = time.perf_counter()
        processed = [preprocess_text(t, method) for t in sample]
        elapsed = time.perf_counter() - start
        vocab = {w for doc in processed for w in doc.split()}
        rows.append({"method": method, "vocab_size": len(vocab), "time_s": round(elapsed, 2)})
    return pd.DataFrame(rows)


def normalization_examples(words=("studies", "running", "better", "governments", "wolves")):
    """Показывает, как стемминг и лемматизация работают на типичных словах."""
    return pd.DataFrame({
        "word": list(words),
        "stem": [stem_word(w) for w in words],
        "lemma": [lemmatize_word(w) for w in words],
    })
