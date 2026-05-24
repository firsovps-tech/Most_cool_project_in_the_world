# Pipeline проекта

## Что откуда взято

- Rita: основной алгоритм поиска, кэш поиска, BM25, embeddings, FAISS/HNSW, LLM-парсинг характеристик, Yandex GPT подключение, запуск одного запроса в терминале.
- Ilnur: товарные CSV, `overnight_gold_standard.json`, `smart_gold_standard.json`, обычные метрики и LLM-метрики.
- Rita datasets не используются.

## Датасеты

- `data/overnight_gold_standard.json` — train для подбора весов.
- `data/smart_gold_standard.json` — test для обычных метрик.
- `data/queries_for_search.json` — финальные запросы для `search_output_top_20.json`.

## Что изменено

Формула score не менялась:

```text
score = sum(feature_value * weight)
```

Изменён только способ выбора весов: оптимизированный перебор до 10 000 комбинаций. У каждого веса есть 9 кандидатных значений, но полный `9^N` перебор не запускается. Используются ручные профили, локальные переборы вокруг базовых весов и random search.

Перед Rita LLM-парсингом добавлен отдельный LLM-запрос для исправления орфографии. Запросы, похожие на артикулы/product id, в LLM-исправление не отправляются.

LLM-метрики вынесены в отдельный запуск `evaluate_smart_llm.py`.

## Настройка Yandex GPT

Создай `.env` по примеру `.env.example`:

```env
OPENAI_BASE_URL=https://llm.api.cloud.yandex.net/v1
OPENAI_API_KEY=your_yandex_api_key
OPENAI_MODEL=gpt://your_folder_id/yandexgpt-lite
OPENAI_PROJECT=your_folder_id
```

## Команды запуска

```bash
python -m pip install -r requirements_search.txt
```

Построить/сохранить кэш поиска Rita:

```bash
python build_search_cache.py
```

Подобрать веса на overnight и проверить обычные метрики на smart:

```bash
python tune_weights_grid.py
```

Отдельно проверить обычные метрики на smart:

```bash
python evaluate_smart.py
```

Отдельно проверить LLM-метрики на smart:

```bash
python evaluate_smart_llm.py
```

Сделать финальную выдачу top-20:

```bash
python run_search_top20.py
```

Проверить формат финальной выдачи:

```bash
python check_search_output.py
```

Запуск одного запроса в терминале:

```bash
python run_one_query_loop.py
```

## Файлы результатов

- `grid_best_params.json` — лучшие группы весов.
- `grid_best_summary.json` — train/test summary.
- `eval_summary_grid_train.json` — обычные метрики на overnight.
- `eval_summary_grid_test.json` — обычные метрики на smart.
- `eval_llm_summary_smart.json` — отдельные LLM-метрики на smart.
- `eval_llm_rows_smart.json` — подробные LLM-оценки по запросам Smart.
- `data/search_output_top_20.json` — финальная выдача.

## Если менялись признаки/поиск/данные

Удалить кэши:

```bash
rm -rf data/search_cache feature_cache data/spelling_correction_cache.json data/judge_cache.json trained_boost_ranker
```
