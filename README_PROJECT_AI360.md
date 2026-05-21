# AI360 Product Search / Furniture Ranking

Проект строит поисковую систему по товарам: пользователь вводит текстовый запрос, система парсит его через LLM, находит похожие товары через BM25 + embeddings + FAISS и сортирует результаты обученным ранжировщиком.

Основная задача проекта — выдавать `top-20` релевантных товаров для каждого запроса.

---

## 1. Что делает проект

Пайплайн состоит из нескольких частей:

1. **Парсинг запроса через LLM**  
   Сырой запрос пользователя превращается в структуру:

   ```json
   {
     "domain": "Мебель",
     "characteristics": {
       "Тип товара": "диван",
       "Цвет": "розовый",
       "Материал": "-",
       "Особенности": "для кухни"
     }
   }
   ```

2. **Поиск кандидатов**  
   Для товаров строятся:
   - BM25-индекс для поиска по словам;
   - embedding-векторы через `BAAI/bge-m3`;
   - FAISS HNSW-индекс для быстрого поиска похожих векторов;
   - field embeddings по отдельным характеристикам: тип товара, цвет, материал, особенности и т.д.

3. **Ранжирование**  
   Для пары “запрос — товар” считаются признаки:

   ```text
   bm25_score
   general_embedding_score
   detail_embedding_score
   Тип товара_embedding_score
   Цвет_embedding_score
   Материал_embedding_score
   product_has_Наименование
   ...
   ```

   Затем XGBoost / обученные веса сортируют товары по итоговому `ranker_score`.

4. **Кеширование**  
   Чтобы не пересчитывать embeddings товаров каждый раз, проект заранее строит кеш:

   ```text
   data/search_cache/
   ```

   Также можно заранее построить обучающую матрицу:

   ```text
   data/training_cache/
   ```

---

## 2. Структура проекта

Примерная структура после реорганизации:

```text
project_ai360/
│
├── data/
│   ├── 1_таблица(1).csv
│   ├── 2_таблица(2).csv
│   ├── 3_таблица(1).csv
│   ├── 4_таблица(1).csv
│   ├── 5_таблица(1).csv
│   ├── golden_standard.json
│   ├── golden_queries_parsed.json
│   ├── golden_answers.json
│   ├── queries_for_search.json
│   ├── search_output_top_20.json
│   ├── domain_characteristics.json
│   └── search_cache/
│
├── src/
│   ├── core/
│   │   ├── search_core.py
│   │   └── search_cache.py
│   │
│   ├── parsing/
│   │   └── project.py
│   │
│   ├── ranking/
│   │   └── boost_ranker.py
│   │
│   └── scripts/
│       ├── prepare_golden.py
│       ├── build_search_cache.py
│       ├── build_training_cache.py
│       ├── test_boost_ranker.py
│       ├── run_search_top20.py
│       ├── run_one_query_loop.py
│       ├── check_search_output.py
│       └── visualize_vector_db.py
│
├── trained_boost_ranker/
│   └── Мебель/
│       ├── xgb_ranker.json
│       ├── feature_names.json
│       └── boost_feature_weights.json
│
├── eval_summary_boost.json
├── eval_rows_boost.json
├── requirements_search.txt
├── .env
└── README.md
```

---

## 3. Установка

### 3.1. Рекомендуемый вариант: запускать в WSL/Linux-папке

Лучше держать проект не на Windows-диске `/mnt/c/...`, а внутри Linux-директории:

```bash
~/project_ai360
```

Например:

```bash
cd ~
git clone <URL_ТВОЕГО_РЕПОЗИТОРИЯ> project_ai360
cd project_ai360
```

Если проект уже лежит на Windows-диске, можно скопировать его в Linux-папку:

```bash
rsync -av --progress \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".git" \
  /mnt/c/Users/Маргарита/PycharmProjects/project_ai360/ \
  ~/project_ai360/
```

Это важно, потому что `.venv` на `/mnt/c/...` может очень медленно грузить `torch`, `transformers`, `FlagEmbedding`.

---

### 3.2. Создание виртуального окружения

Из корня проекта:

```bash
cd ~/project_ai360

python -m venv .venv --upgrade-deps
source .venv/bin/activate
```

Проверь, что Python реально из `.venv`:

```bash
which python
python -m pip --version
```

Ожидаемый путь:

```text
/home/rita/project_ai360/.venv/bin/python
/home/rita/project_ai360/.venv/lib/python.../site-packages/pip
```

---

### 3.3. Установка зависимостей

Ставить зависимости нужно именно через:

```bash
python -m pip
```

Не через просто `pip`, потому что `pip` может указывать на глобальный `pyenv`.

Команды:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_search.txt
```

Если каких-то библиотек не хватает, можно доустановить полный набор:

```bash
python -m pip install numpy pandas xgboost rank-bm25 faiss-cpu FlagEmbedding transformers torch scikit-learn umap-learn matplotlib python-dotenv openai tqdm
```

Проверка:

```bash
python -c "import numpy; print('numpy ok')"
python -c "import faiss; print('faiss ok')"
python -c "import xgboost; print('xgboost ok')"
python -c "import FlagEmbedding; print('FlagEmbedding ok')"
```

---

## 4. Настройка `.env`

В корне проекта должен быть файл:

```text
.env
```

Пример для YandexGPT / OpenAI-compatible API:

```env
OPENAI_BASE_URL=https://llm.api.cloud.yandex.net/v1
OPENAI_API_KEY=YOUR_API_KEY
OPENAI_MODEL=gpt://YOUR_FOLDER_ID/yandexgpt-lite/latest
```

Или если используется другой OpenAI-compatible endpoint:

```env
OPENAI_BASE_URL=https://...
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

Важно: `.env` не нужно коммитить в публичный репозиторий, потому что там лежит секретный ключ.

В `.gitignore` лучше добавить:

```gitignore
.env
.venv/
__pycache__/
data/search_cache/
data/training_cache/
```

---

## 5. Полный запуск с нуля

Если проект запускается впервые, порядок такой:

```bash
python -m src.scripts.test_parser_only
python -m src.scripts.prepare_golden
python -m src.scripts.build_search_cache
python -m src.scripts.build_training_cache
python -m src.scripts.test_boost_ranker
python -m src.scripts.run_search_top20
python -m src.scripts.check_search_output
```

Ниже подробно, что делает каждая команда.

---

## 6. Проверка LLM-парсера

```bash
python -m src.scripts.test_parser_only
```

Команда проверяет, что:
- `.env` читается;
- LLM отвечает;
- запрос можно распарсить в JSON.

Если эта команда не работает, сначала надо чинить `.env` / API-ключ / модель.

---

## 7. Подготовка golden-запросов

```bash
python -m src.scripts.prepare_golden
```

Эта команда берёт:

```text
data/golden_standard.json
```

и создаёт:

```text
data/golden_queries_parsed.json
data/golden_answers.json
```

`golden_queries_parsed.json` содержит распарсенные запросы.

`golden_answers.json` содержит правильные товары:

```json
{
  "query_id": "g1",
  "relevant_ids": ["s1_123", "s1_456"]
}
```

Запускать заново нужно, если поменялся:
- `golden_standard.json`;
- логика парсинга в `src/parsing/project.py`;
- `domain_characteristics.json`.

---

## 8. Построение search cache

```bash
python -m src.scripts.build_search_cache
```

Это долгий шаг.

Он делает:

```text
CSV товары
↓
BM25
↓
BAAI/bge-m3 embeddings
↓
field embeddings
↓
FAISS HNSW index
↓
data/search_cache/
```

В процессе могут появляться строки:

```text
pre tokenize: 100%|...
Inference Embeddings: 100%|...
```

Это нормально. Модель переводит товары в embedding-векторы.

После успешного запуска появится папка:

```text
data/search_cache/
```

В ней лежат:
- товары;
- embeddings товаров;
- embeddings по характеристикам;
- FAISS-индексы;
- настройки доменов;
- feature names.

Эту команду не нужно запускать каждый раз.

Запускать заново нужно, если поменялись:
- CSV-файлы товаров;
- логика embeddings;
- `search_core.py`;
- `domain_characteristics.json`;
- набор признаков товаров.

---

## 9. Построение training cache

```bash
python -m src.scripts.build_training_cache
```

Это промежуточный кеш для обучения XGBoost.

Он делает:

```text
golden_queries_parsed.json
+ golden_answers.json
+ data/search_cache/
↓
X.npy
y.npy
qid.npy
feature_names.json
↓
data/training_cache/
```

Что такое `X`, `y`, `qid`:

```text
X   — матрица признаков для пар “запрос — товар”
y   — label: 1, если товар релевантен, 0 иначе
qid — номер группы запроса для XGBRanker
```

Этот шаг может быть долгим, потому что для 100 запросов строится примерно 100 000 пар “запрос — товар”.

Запускать заново нужно, если поменялись:
- `golden_queries_parsed.json`;
- `golden_answers.json`;
- `data/search_cache/`;
- логика признаков в `search_core.py`;
- логика построения train dataset в `boost_ranker.py`.

Если ты меняешь только параметры XGBoost (`n_estimators`, `learning_rate`, `max_depth`), этот шаг запускать не надо.

---

## 10. Обучение ранжировщика

```bash
python -m src.scripts.test_boost_ranker
```

Эта команда обучает XGBoost-ранжировщик.

После оптимизации она должна использовать готовый `training_cache`, а не пересчитывать embeddings товаров каждый раз.

На выходе создаётся:

```text
trained_boost_ranker/
```

Внутри:

```text
trained_boost_ranker/Мебель/xgb_ranker.json
trained_boost_ranker/Мебель/feature_names.json
trained_boost_ranker/Мебель/boost_feature_weights.json
```

Также сохраняются метрики:

```text
eval_summary_boost.json
eval_rows_boost.json
```

В терминале появится блок:

```text
Evaluation summary:
precision@10 ...
recall@10 ...
hit@10 ...
mrr@10 ...
ndcg@10 ...
```

---

## 11. Метрики качества

Пример:

```json
{
  "precision@10": 0.7,
  "recall@10": 0.07,
  "hit@10": 0.97,
  "mrr@10": 0.9,
  "ndcg@10": 0.7372,
  "error_hit@10": 0.03,
  "error_mrr@10": 0.1
}
```

Что означает:

- `precision@10 = 0.7` — в среднем 7 из 10 товаров в top-10 релевантные.
- `recall@10 = 0.07` — в top-10 найдено 7% всех релевантных товаров.
- `hit@10 = 0.97` — в 97% запросов в top-10 есть хотя бы один правильный товар.
- `mrr@10 = 0.9` — первый правильный товар почти всегда находится очень высоко.
- `ndcg@10 = 0.7372` — релевантные товары стоят достаточно близко к началу.
- `error_hit@10 = 1 - hit@10`.
- `error_mrr@10 = 1 - mrr@10`.

---

## 12. Запуск поиска по всем запросам

```bash
python -m src.scripts.run_search_top20
```

Команда берёт:

```text
data/queries_for_search.json
```

и создаёт:

```text
data/search_output_top_20.json
```

Формат ответа:

```json
[
  {
    "query_id": "q1",
    "query_text": "розовый стул",
    "top_20_product_ids": ["s1_123", "s1_456"]
  }
]
```

---

## 13. Проверка финального JSON

```bash
python -m src.scripts.check_search_output
```

Она проверяет:
- есть ли `query_id`;
- есть ли `query_text`;
- есть ли `top_20_product_ids`;
- не больше ли 20 товаров;
- существуют ли такие product IDs в базе.

Хороший результат:

```text
Format errors: 0
Unknown ids: 0
OK: output format is correct
```

---

## 14. Интерактивный поиск по одному запросу

```bash
python -m src.scripts.run_one_query_loop
```

После запуска:

```text
Loading cached search engine...
Loading trained weights...

Ready. Вводи запросы. Для выхода напиши: exit

query>
```

Пример:

```text
query> хочу купить новый розовый красивый диван большой на кухню
```

Система выведет:

```text
Prepared query:
...

Top 20 product ids:
[...]

Detailed results:
1. id=s1_450 score=0.861453
   Кресло Турин розовый
```

Чтобы выйти:

```text
query> exit
```

Этот режим удобен для ручной проверки качества поиска.

---

## 15. Ручное изменение коэффициентов

Финальные коэффициенты лежат в:

```text
trained_boost_ranker/Мебель/boost_feature_weights.json
```

Там можно менять веса:

```text
bm25_score
general_embedding_score
detail_embedding_score
Тип товара_embedding_score
Цвет_embedding_score
Материал_embedding_score
product_has_Наименование
...
```

Например, если нужно сделать тип товара важнее цвета:

```json
{
  "Тип товара_embedding_score": 0.28,
  "Цвет_embedding_score": 0.04
}
```

После ручного изменения коэффициентов **не надо переобучать модель**.

Достаточно перезапустить:

```bash
python -m src.scripts.run_one_query_loop
```

или для всех запросов:

```bash
python -m src.scripts.run_search_top20
python -m src.scripts.check_search_output
```

---

## 16. Где находятся коэффициенты BM25 и embeddings

Есть два уровня весов.

### Первый этап: отбор кандидатов

Файл:

```text
src/core/search_core.py
```

Метод:

```python
get_candidate_indexes_for_domain(...)
```

Там есть формула:

```python
first_stage_scores[local_index] = (
    0.5 * bm25_scores[local_index]
    + 0.5 * hnsw_scores[local_index]
)
```

Это первичный отбор кандидатов:

```text
50% BM25
50% general embedding / FAISS
```

### Финальный этап: итоговый ranker_score

Файл:

```text
trained_boost_ranker/Мебель/boost_feature_weights.json
```

А используется это в:

```text
src/ranking/boost_ranker.py
```

Формула:

```python
score += feature_value * weight
```

То есть:

```text
ranker_score =
bm25_score * weight_bm25
+ general_embedding_score * weight_embedding
+ Тип товара_embedding_score * weight_type
+ Цвет_embedding_score * weight_color
+ ...
```

---

## 17. Визуализация embedding-базы

```bash
python -m src.scripts.visualize_vector_db
```

Команда строит картинку embedding-пространства товаров.

Обычно на графике:

```text
одна точка = один товар
похожие товары находятся рядом
```

Если используется кеш, скрипт должен брать embeddings из:

```text
data/search_cache/
```

а не считать их заново.

---

## 18. Когда какие команды запускать

### Первый запуск проекта

```bash
python -m src.scripts.test_parser_only
python -m src.scripts.prepare_golden
python -m src.scripts.build_search_cache
python -m src.scripts.build_training_cache
python -m src.scripts.test_boost_ranker
python -m src.scripts.run_search_top20
python -m src.scripts.check_search_output
```

### Если поменялись только тестовые запросы

```bash
python -m src.scripts.run_search_top20
python -m src.scripts.check_search_output
```

### Если поменялись CSV-файлы товаров

```bash
python -m src.scripts.build_search_cache
python -m src.scripts.build_training_cache
python -m src.scripts.test_boost_ranker
python -m src.scripts.run_search_top20
python -m src.scripts.check_search_output
```

### Если поменялся `golden_standard.json`

```bash
python -m src.scripts.prepare_golden
python -m src.scripts.build_training_cache
python -m src.scripts.test_boost_ranker
```

### Если поменялись параметры XGBoost

```bash
python -m src.scripts.test_boost_ranker
```

### Если поменялись только ручные коэффициенты

```bash
python -m src.scripts.run_one_query_loop
```

или:

```bash
python -m src.scripts.run_search_top20
python -m src.scripts.check_search_output
```

---

## 19. Частые ошибки

### `ModuleNotFoundError: No module named 'faiss'`

Установить:

```bash
python -m pip install faiss-cpu
```

### `ModuleNotFoundError: No module named 'numpy'`

Установить:

```bash
python -m pip install numpy
```

### `ModuleNotFoundError: No module named 'tqdm'`

Установить:

```bash
python -m pip install tqdm
```

### `python -m pip: No module named pip`

Значит `.venv` создан криво. Пересоздать:

```bash
deactivate 2>/dev/null
rm -rf .venv

python -m venv .venv --upgrade-deps
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_search.txt
```

### `pip` ставит пакеты не в `.venv`

Проверить:

```bash
which python
which pip
python -m pip --version
```

Если `pip` указывает на `pyenv`, используй только:

```bash
python -m pip install ...
```

---

## 20. Короткое резюме

Основные команды:

```bash
python -m src.scripts.prepare_golden
python -m src.scripts.build_search_cache
python -m src.scripts.build_training_cache
python -m src.scripts.test_boost_ranker
python -m src.scripts.run_search_top20
python -m src.scripts.check_search_output
python -m src.scripts.run_one_query_loop
```

Самые важные папки:

```text
data/search_cache/        — готовая embedding-база товаров
data/training_cache/      — готовая матрица обучения XGBoost
trained_boost_ranker/     — обученные веса ранжировщика
```

Главная идея оптимизации:

```text
Embeddings товаров считаются один раз в build_search_cache.
Обучающая матрица считается один раз в build_training_cache.
test_boost_ranker быстро обучает XGBoost по готовому кешу.
run_one_query_loop использует готовую базу и только обрабатывает новые запросы.
```
