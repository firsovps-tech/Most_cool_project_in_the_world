# Финальная версия: товары -> domain_characteristics -> LLM parser -> golden_standard -> веса -> top-20

## Что делает код

1. Читает список товарных файлов из `data/`.
2. В каждом товарном файле первый столбец считается доменом.
3. Названия характеристик берутся из первой строки файла.
4. Отдельного `description` у товаров не нужно.
5. `description` собирается автоматически из характеристик, исключая:
   - domain/category
   - id / ID / product_id
   - article / Артикул
   - price / Цена
6. После загрузки товаров создаётся/обновляется:

```text
data/domain_characteristics.json
```

7. Обучение идёт на:

```text
data/golden_standard.json
```

8. Каждый сырой запрос из `golden_standard.json` идёт в `project.py`.
9. `project.py` берёт `data/domain_characteristics.json`, обращается к LLM и возвращает нормальный запрос:

```json
{
  "query_id": "q1",
  "raw_query_text": "белый стул дерево",
  "query_text": "стул белый дерево",
  "domain": "furniture",
  "characteristics": {
    "product_type": "стул",
    "color": "белый",
    "material": "дерево"
  }
}
```

10. Поиск внутри домена:
    - BM25 по общей строке характеристик
    - FAISS `IndexHNSWFlat` по embedding общей строки
    - top-1000 кандидатов
    - embedding по отдельным характеристикам для top-1000
    - итоговый score = `sum(feature_value * learned_weight)`
    - возвращаем top-20 id

## Структура проекта

```text
project_ai360/
    search_core.py
    project.py
    boost_ranker.py
    test_boost_ranker.py
    run_search_top20.py
    check_search_output.py
    requirements_search.txt

    data/
        product_files.json              # опционально
        твои_товарные_файлы.csv
        golden_standard.json
        queries_for_search.json
```

## Как указать товарные файлы

Можно просто положить товарные CSV/JSON в `data/`. Код сам отфильтрует `queries`, `answers`, `golden`, `output`, `domain_characteristics`.

Надёжнее создать:

```text
data/product_files.json
```

Пример:

```json
[
  "1_таблица(1).csv",
  "2_таблица(2).csv",
  "3_таблица(1).csv"
]
```

## Формат golden_standard.json

Поддерживается несколько вариантов.

Лучший вариант:

```json
[
  {
    "query_id": "q1",
    "query_text": "белый деревянный стул",
    "relevant_ids": ["1", "6", "2"]
  },
  {
    "query_id": "q2",
    "query_text": "черный шкаф лофт",
    "relevant_ids": ["10", "15"]
  }
]
```

Если в golden уже есть разобранный запрос, LLM не будет нужен для этого item:

```json
{
  "query_id": "q1",
  "query_text": "стул белый дерево",
  "domain": "furniture",
  "characteristics": {
    "product_type": "стул",
    "color": "белый",
    "material": "дерево"
  },
  "relevant_ids": ["1", "6"]
}
```

## LLM-настройки

`project.py` берёт настройки из `.env`:

```text
OPENAI_BASE_URL=http://127.0.0.1:4010/v1
OPENAI_API_KEY=unused
OPENAI_MODEL=opencode/deepseek-v4-flash-free
```

## Установка

```bash
python -m pip install -r requirements_search.txt
```

## Обучение на golden_standard

```bash
rm -rf trained_boost_ranker
python test_boost_ranker.py
```

После обучения появится:

```text
data/domain_characteristics.json
data/golden_queries_parsed.json
data/golden_answers.json
trained_boost_ranker/<domain>/xgb_ranker.json
trained_boost_ranker/<domain>/feature_names.json
trained_boost_ranker/<domain>/boost_feature_weights.json
```

## Генерация top-20

```bash
python run_search_top20.py
```

Ответ:

```text
data/search_output_top_20.json
```

Формат:

```json
[
  {
    "query_id": "q1",
    "query_text": "стул",
    "top_20_product_ids": ["1", "6", "2"]
  }
]
```

## Проверка формата

```bash
python check_search_output.py
```
