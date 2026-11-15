# Fake News Detection: классификация новостей на реальные и фейковые

Мини-проект по дисциплине «Programming for Data Analysis and Database».
Тема кластера: **Text Analytics & NLP**.

Задача: по тексту новости определить, является ли она достоверной (`real`) или фейком (`fake`).
Сравниваются три представления текста (**Bag of Words, TF-IDF, Word2Vec embeddings**), baseline и дополнительный наивный байесовский классификатор.

## Структура проекта

```
fake_news_project/
├── main.py            # точка входа, консольное меню (пункты 1–5)
├── config.py          # пути, random_state, размер теста
├── data_loader.py     # чтение CSV по частям, очистка, Parquet
├── preprocessing.py   # токенизация, стоп-слова, стемминг и лемматизация (NLTK)
├── features.py        # BoW, TF-IDF, Word2VecVectorizer (embeddings), ключевые слова
├── models.py          # baseline и классификаторы, замер времени и памяти
├── evaluation.py      # метрики, матрица ошибок, графики, анализ ошибок
├── eda.py             # частотный анализ слов и биграмм
├── database.py        # схема SQLite, загрузка, 10 аналитических SQL-запросов
├── requirements.txt   # библиотеки с версиями
├── docs/er_diagram.md # ER-диаграмма базы данных
├── data/
│   ├── raw/           # сюда кладутся Fake.csv и True.csv (не хранятся в git)
│   ├── processed/     # articles.parquet (создаётся пунктом 1)
│   └── news.db        # SQLite (создаётся пунктом 1)
├── models/            # best_model.joblib и метаданные (создаются пунктом 2)
└── results/           # таблицы метрик, графики, результаты SQL (создаются пунктами 2–4)
```

## Данные

Набор: **Fake and Real News Dataset** (Kaggle):
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Скачайте архив, распакуйте и положите файлы в `data/raw/`:
- `Fake.csv`: метка `fake` (0)
- `True.csv`: метка `real` (1)

Столбцы: `title`, `text`, `subject`, `date`.

## Установка и запуск

Требуется Python 3.10+.

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4   # выполняется и автоматически при первом запуске
python main.py
```

Запуск в Google Colab: загрузите папку проекта, положите CSV в `data/raw/`, затем
`!pip install -r requirements.txt` и `!python main.py` (меню работает через `input()`, поэтому
удобнее запускать ячейкой `%run main.py`).

Порядок работы в меню:
1. **Пункт 1** читает CSV порциями, чистит данные, лемматизирует текст, сохраняет Parquet и заполняет SQLite.
2. **Пункт 2** обучает baseline и четыре метода, сохраняет метрики, матрицы ошибок и лучшую модель.
3. **Пункт 3** строит частотный анализ слов и биграмм по классам и сравнивает стемминг с лемматизацией.
4. **Пункт 4** выполняет SQL-запросы к SQLite и выводит результаты как DataFrame.
5. **Пункт 5** принимает текст новости и выводит вердикт с уверенностью модели.

## Методы и метрики

| Эксперимент | Представление текста | Классификатор | Роль |
|---|---|---|---|
| `baseline_majority` | — | наиболее частый класс | baseline |
| `bow_logreg` | Bag of Words (1–2-граммы) | Logistic Regression | метод |
| `bow_naive_bayes` | Bag of Words | Multinomial Naive Bayes | метод |
| `tfidf_logreg` | TF-IDF (sublinear tf) | Logistic Regression | метод |
| `w2v_logreg` | Word2Vec (100-мерное среднее) | Logistic Regression | метод (embeddings) |

Метрики: **Accuracy**, **F1-macro** (основная), Precision-macro, Recall-macro.
Для каждого метода также измеряются **время обучения и предсказания** и **пиковая память** (`tracemalloc`).

Обучающая и тестовая выборки: 80/20, стратифицированное разбиение, `random_state=42`.
Word2Vec обучается только на обучающей части, чтобы не допустить утечки данных.

## Предобработка

1. Удаление дубликатов по `title` + `text` и пустых записей.
2. Удаление префикса агентства вида `WASHINGTON (Reuters) -`. Этот префикс почти всегда встречается только в настоящих новостях и без его удаления модель может выучить признак источника, а не содержание.
3. Приведение к нижнему регистру, удаление URL, токенизация (`RegexpTokenizer` из NLTK).
4. Удаление стоп-слов NLTK и слов короче 3 символов.
5. Нормализация: лемматизация WordNet (используется в моделях). Стемминг Porter сравнивается с лемматизацией и без нормализации в пункте 3.

## База данных и SQL

Схема описана в `docs/er_diagram.md`. Таблицы: `subjects`, `articles`, `keywords`, `experiments`, `predictions`, `user_queries`.

Аналитические запросы (пункт 4), каждый результат сохраняется в `results/sql/` и загружается в pandas:

| Ключ | Что используется |
|---|---|
| q01_subject_label_stats | JOIN, GROUP BY |
| q02_fake_heavy_subjects | GROUP BY, HAVING |
| q03_long_articles_subquery | подзапрос (IN, скалярный подзапрос) |
| q04_monthly_trend_cte_window | CTE, оконная функция (скользящее среднее) |
| q05_subject_rank_window | CTE, RANK() OVER (PARTITION BY) |
| q06_model_leaderboard | подзапрос, оконные функции |
| q07_error_rate_by_subject | JOIN ×4, GROUP BY, HAVING |
| q08_confusion_best_model | GROUP BY, подзапрос |
| q09_fake_vocabulary_cte | CTE, JOIN, GROUP BY, HAVING |
| q10_not_in_test_not_exists | NOT EXISTS |

## Анализ ошибок

После пункта 2 в `results/` сохраняются:
- `errors_by_type.csv`: fake→real и real→fake;
- `errors_by_subject.csv`: доля ошибок по темам;
- `errors_by_length.csv`: доля ошибок по длине текста;
- `errors_confident_errors.csv`: самые уверенные ошибки модели;
- `top_fake_terms_tfidf.csv` и `top_real_terms_tfidf.csv`: слова, сильнее всего влияющие на решение.

Матрицы ошибок всех методов лежат в `results/figures/confusion_*.png`.

## Ограничения

- Высокие метрики на этом наборе частично объясняются стилистическими различиями между источниками. Анализ ошибок и удаление префикса Reuters направлены на то, чтобы показать и ограничить этот эффект.
