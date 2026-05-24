
import json
import random
import time
from pathlib import Path

import numpy as np

from src.core.search_cache import load_search_engine_cache
from src.core.search_core import save_json, field_value
from src.evaluation.metrics import RankingMetrics
from src.parsing.project import normalize_one_query
from src.tuning.feature_cache import get_or_build_feature_cache

try:
    from src.tuning.weight_grid import MAX_WEIGHT_COMBINATIONS
except Exception:
    MAX_WEIGHT_COMBINATIONS = 10000


DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "search_cache"
MODEL_DIR = Path("trained_boost_ranker")
FEATURE_CACHE_DIR = Path("feature_cache")

TRAIN_GOLD_PATH = DATA_DIR / "overnight_gold_standard.json"
TEST_GOLD_PATH = DATA_DIR / "smart_gold_standard.json"
TRAIN_FEATURE_CACHE = FEATURE_CACHE_DIR / "overnight_feature_cache.pkl"
TEST_FEATURE_CACHE = FEATURE_CACHE_DIR / "smart_feature_cache.pkl"

TRAIN_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "overnight_parsed_gold_rows.json"
TEST_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "smart_parsed_gold_rows.json"

TRAIN_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "overnight_parsed_gold_rows.json"
TEST_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "smart_parsed_gold_rows.json"

TRAIN_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "overnight_parsed_gold_rows.json"
TEST_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "smart_parsed_gold_rows.json"

TRAIN_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "overnight_parsed_gold_rows.json"
TEST_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "smart_parsed_gold_rows.json"

TRAIN_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "overnight_parsed_gold_rows.json"
TEST_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "smart_parsed_gold_rows.json"

TRAIN_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "overnight_parsed_gold_rows.json"
TEST_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "smart_parsed_gold_rows.json"

TRAIN_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "overnight_parsed_gold_rows.json"
TEST_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "smart_parsed_gold_rows.json"

TRAIN_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "overnight_parsed_gold_rows.json"
TEST_PARSED_GOLD_CACHE = FEATURE_CACHE_DIR / "smart_parsed_gold_rows.json"

TOP_K_VALUES = [5, 10, 20]
FIRST_STAGE_LIMIT = 1000
FINAL_CANDIDATE_LIMIT = None
USE_LLM_PARSER = True

RANDOM_SEED = 42


def load_json(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_gold_rows(path, dataset_name, parsed_cache_path=None):
    path = Path(path)

    if parsed_cache_path is not None:
        parsed_cache_path = Path(parsed_cache_path)

        if parsed_cache_path.exists():
            print(f"[LOAD GOLD CACHE] {dataset_name}: {parsed_cache_path}", flush=True)
            cached_rows = load_json(parsed_cache_path)

            if cached_rows:
                print(f"[LOAD GOLD CACHE] loaded rows={len(cached_rows)}", flush=True)
                return cached_rows

    raw_rows = load_json(path)
    rows = []

    print(f"[LOAD GOLD] {dataset_name}: rows={len(raw_rows)}", flush=True)

    for row_index, item in enumerate(raw_rows, start=1):
        query_id = field_value(item.get("query_id"))
        query_text = field_value(item.get("query_text")) or field_value(item.get("query"))
        ideal_ids = item.get("ideal_product_ids", [])

        if not query_id or not query_text or not ideal_ids:
            continue

        prepared_query = normalize_one_query(
            query_id=query_id,
            query_text=query_text,
            known_domain=field_value(item.get("domain")) or None,
            use_llm=USE_LLM_PARSER,
        )

        if row_index == 1 or row_index % 10 == 0 or row_index == len(raw_rows):
            print(
                f"[LOAD GOLD] {dataset_name}: parsed {row_index}/{len(raw_rows)} | "
                f"last_query={query_id} | domain={prepared_query.get('domain')}",
                flush=True,
            )

        rows.append({
            "dataset": dataset_name,
            "query_id": query_id,
            "query_text": query_text,
            "prepared_query": prepared_query,
            "ideal_product_ids": [
                field_value(product_id)
                for product_id in ideal_ids
                if field_value(product_id)
            ],
        })

    if not rows:
        raise ValueError(f"Не удалось загрузить gold rows из {path}")

    if parsed_cache_path is not None:
        parsed_cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(parsed_cache_path, rows)
        print(f"[LOAD GOLD CACHE] saved: {parsed_cache_path}", flush=True)

    return rows


def collect_feature_names_from_cache(feature_cache):
    result = []
    seen = set()

    for query_cache in feature_cache:
        for feature_name in query_cache["feature_names"]:
            if feature_name not in seen:
                seen.add(feature_name)
                result.append(feature_name)

    return result


def candidates_for_feature(feature_name):
    """
    Для каждого реального признака даём 9 вариантов веса.

    Теперь признаки характеристик не получают один общий w_field_embedding.
    Каждый признак вида:
        Цвет_embedding_score
        Материал_embedding_score
        product_has_Цвет
    подбирается отдельно.
    """
    if feature_name == "exact_id_score":
        return [0.0, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0]

    if feature_name == "bm25_score":
        return [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    if feature_name == "general_embedding_score":
        return [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    if feature_name == "detail_embedding_score":
        return [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    if feature_name == "price_score":
        return [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    if feature_name in {"has_query_id", "has_price_filter"}:
        return [0.0, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

    if feature_name.startswith("product_has_"):
        return [0.0, 0.005, 0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25]

    if feature_name.endswith("_embedding_score"):
        return [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    return [0.0, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


def base_weight_for_feature(feature_name):
    if feature_name == "exact_id_score":
        return 100.0

    if feature_name == "bm25_score":
        return 2.0

    if feature_name == "general_embedding_score":
        return 0.5

    if feature_name == "detail_embedding_score":
        return 0.5

    if feature_name == "price_score":
        return 0.0

    if feature_name in {"has_query_id", "has_price_filter"}:
        return 0.0

    if feature_name.startswith("product_has_"):
        return 0.05

    if feature_name.endswith("_embedding_score"):
        return 0.5

    return 0.0


def make_base_feature_weights(feature_names):
    return {
        feature_name: base_weight_for_feature(feature_name)
        for feature_name in feature_names
    }


def generate_feature_weight_candidates(feature_names, max_combinations):
    """
    Оптимизированный подбор до max_combinations.

    Полный перебор невозможен:
        9 ** num_features

    Поэтому делаем:
    1. базовый вектор весов;
    2. одиночные переборы каждого признака;
    3. random search по всем признакам.
    """
    rng = random.Random(RANDOM_SEED)

    feature_names = list(feature_names)
    candidate_values = {
        feature_name: candidates_for_feature(feature_name)
        for feature_name in feature_names
    }

    base_weights = make_base_feature_weights(feature_names)

    candidates = []
    seen = set()

    def add_candidate(weights):
        key = tuple(float(weights.get(feature_name, 0.0)) for feature_name in feature_names)

        if key in seen:
            return

        seen.add(key)
        candidates.append({
            feature_name: float(weights.get(feature_name, 0.0))
            for feature_name in feature_names
        })

    add_candidate(base_weights)

    # Сначала проверяем влияние каждого признака отдельно.
    for feature_name in feature_names:
        for value in candidate_values[feature_name]:
            candidate = dict(base_weights)
            candidate[feature_name] = float(value)
            add_candidate(candidate)

            if len(candidates) >= max_combinations:
                return candidates, candidate_values

    # Потом случайные комбинации всех отдельных весов.
    max_attempts = max_combinations * 50
    attempts = 0

    while len(candidates) < max_combinations and attempts < max_attempts:
        attempts += 1

        candidate = {
            feature_name: float(rng.choice(candidate_values[feature_name]))
            for feature_name in feature_names
        }

        add_candidate(candidate)

    return candidates, candidate_values


def weights_dict_to_saved_list(weights):
    total = sum(abs(value) for value in weights.values())
    result = []

    for feature, value in weights.items():
        normalized = abs(value) / total if total > 0 else 0.0
        result.append({
            "feature": feature,
            "importance": float(value),
            "normalized_weight": float(normalized),
        })

    result.sort(key=lambda item: item["normalized_weight"], reverse=True)
    return result


def rank_from_cache(query_cache, feature_weights, top_k=20):
    product_ids = query_cache["product_ids"]
    features = query_cache["features"]
    feature_names = query_cache["feature_names"]

    if len(product_ids) == 0:
        return []

    weights_vector = np.asarray(
        [feature_weights.get(name, 0.0) for name in feature_names],
        dtype=np.float32,
    )

    scores = features @ weights_vector
    rows = []

    for product_id, score in zip(product_ids, scores):
        rows.append({
            "product_id": str(product_id),
            "ranker_score": float(score),
        })

    rows.sort(key=lambda item: item["ranker_score"], reverse=True)
    return rows[:top_k]


def evaluate_from_cache(feature_cache, feature_weights):
    rows = []

    metric_by_k = {
        k: RankingMetrics(k=k)
        for k in TOP_K_VALUES
    }

    for query_cache in feature_cache:
        ranked_rows = rank_from_cache(
            query_cache=query_cache,
            feature_weights=feature_weights,
            top_k=max(TOP_K_VALUES),
        )

        predicted_ids = [row["product_id"] for row in ranked_rows]
        ideal_ids = query_cache["ideal_product_ids"]

        row = {
            "dataset": query_cache["dataset"],
            "query_id": query_cache["query_id"],
            "query_text": query_cache["query_text"],
            "corrected_query_text": query_cache.get("corrected_query_text", query_cache["query_text"]),
            "top_20_product_ids": predicted_ids[:20],
            "ideal_product_ids": ideal_ids,
        }

        for k, metric in metric_by_k.items():
            result = metric.calculate_traditional(predicted_ids, ideal_ids)
            row[f"ndcg@{k}"] = float(result["ndcg"])
            row[f"hit@{k}"] = float(result["hit"])
            row[f"rank@{k}"] = result["rank"]
            row[f"error_hit@{k}"] = 1.0 - float(result["hit"])

        rows.append(row)

    summary = summarize_rows(rows)
    return summary, rows


def summarize_rows(rows):
    summary = {}
    metric_names = []

    for k in TOP_K_VALUES:
        metric_names.extend([
            f"ndcg@{k}",
            f"hit@{k}",
            f"error_hit@{k}",
        ])

    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        summary[metric_name] = float(np.mean(values)) if values else 0.0

    for k in TOP_K_VALUES:
        ranks = [row[f"rank@{k}"] for row in rows if row[f"rank@{k}"] is not None]
        summary[f"rank@{k}_mean_found"] = float(np.mean(ranks)) if ranks else None
        summary[f"rank@{k}_missing"] = sum(1 for row in rows if row[f"rank@{k}"] is None)

    summary["queries"] = len(rows)

    by_dataset = {}

    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    summary["by_dataset"] = {}

    for dataset_name, dataset_rows in by_dataset.items():
        summary["by_dataset"][dataset_name] = summarize_rows_without_by_dataset(dataset_rows)

    return summary


def summarize_rows_without_by_dataset(rows):
    summary = {}
    metric_names = []

    for k in TOP_K_VALUES:
        metric_names.extend([
            f"ndcg@{k}",
            f"hit@{k}",
            f"error_hit@{k}",
        ])

    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        summary[metric_name] = float(np.mean(values)) if values else 0.0

    for k in TOP_K_VALUES:
        ranks = [row[f"rank@{k}"] for row in rows if row[f"rank@{k}"] is not None]
        summary[f"rank@{k}_mean_found"] = float(np.mean(ranks)) if ranks else None
        summary[f"rank@{k}_missing"] = sum(1 for row in rows if row[f"rank@{k}"] is None)

    summary["queries"] = len(rows)
    return summary


def selection_score(summary):
    return (
        0.35 * summary["ndcg@10"]
        + 0.25 * summary["hit@10"]
        + 0.25 * summary["ndcg@20"]
        + 0.15 * summary["hit@20"]
    )


def better_summary(current, best):
    if best is None:
        return True

    current_score = selection_score(current)
    best_score = selection_score(best)

    if current_score > best_score + 1e-12:
        return True

    if current_score < best_score - 1e-12:
        return False

    priority = ["ndcg@10", "hit@10", "ndcg@20", "hit@20", "ndcg@5", "hit@5"]

    for metric_name in priority:
        if current[metric_name] > best[metric_name] + 1e-12:
            return True

        if current[metric_name] < best[metric_name] - 1e-12:
            return False

    return False


def save_best_weights(search_engine, best_feature_weights, train_summary, test_summary, candidate_values):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    saved_summary = {
        "selection_metric": "0.35*ndcg@10 + 0.25*hit@10 + 0.25*ndcg@20 + 0.15*hit@20",
        "selection_score_train": selection_score(train_summary),
        "train_summary": train_summary,
        "test_summary": test_summary,
        "best_feature_weights": best_feature_weights,
    }

    for domain in search_engine.domain_configs:
        feature_names = search_engine.get_feature_names(domain)

        domain_weights = {
            feature_name: float(best_feature_weights.get(feature_name, 0.0))
            for feature_name in feature_names
        }

        domain_candidate_values = {
            feature_name: candidate_values.get(feature_name, [])
            for feature_name in feature_names
        }

        domain_dir = MODEL_DIR / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        save_json(domain_dir / "feature_names.json", feature_names)
        save_json(domain_dir / "boost_feature_weights.json", weights_dict_to_saved_list(domain_weights))
        save_json(domain_dir / "boost_feature_weights_dict.json", domain_weights)
        save_json(domain_dir / "feature_weight_candidates.json", domain_candidate_values)
        save_json(domain_dir / "grid_best_params.json", domain_weights)
        save_json(domain_dir / "grid_best_summary.json", saved_summary)


def main():
    print("=== FEATURE-LEVEL WEIGHT TUNING: up to 10000 combinations ===")
    print("Train dataset:", TRAIN_GOLD_PATH)
    print("Test dataset:", TEST_GOLD_PATH)

    print("\nLoad cached Rita search engine:", CACHE_DIR)
    search_engine = load_search_engine_cache(CACHE_DIR)

    print("\nPrepare train gold rows...")
    train_rows = load_gold_rows(TRAIN_GOLD_PATH, TRAIN_GOLD_PATH.name, TRAIN_PARSED_GOLD_CACHE)
    print("Train queries:", len(train_rows))

    print("\nPrepare test gold rows...")
    test_rows = load_gold_rows(TEST_GOLD_PATH, TEST_GOLD_PATH.name, TEST_PARSED_GOLD_CACHE)
    print("Test queries:", len(test_rows))

    print("\nBuild/load feature cache for train...")
    train_feature_cache = get_or_build_feature_cache(
        TRAIN_FEATURE_CACHE,
        search_engine=search_engine,
        gold_rows=train_rows,
        first_stage_limit=FIRST_STAGE_LIMIT,
        final_candidate_limit=FINAL_CANDIDATE_LIMIT,
    )

    print("\nBuild/load feature cache for test...")
    test_feature_cache = get_or_build_feature_cache(
        TEST_FEATURE_CACHE,
        search_engine=search_engine,
        gold_rows=test_rows,
        first_stage_limit=FIRST_STAGE_LIMIT,
        final_candidate_limit=FINAL_CANDIDATE_LIMIT,
    )

    feature_names = collect_feature_names_from_cache(train_feature_cache)

    print("\nFeature-level weights:")
    print("Total features:", len(feature_names))

    for feature_name in feature_names:
        print(f"- {feature_name}: 9 values")

    candidates, candidate_values = generate_feature_weight_candidates(
        feature_names=feature_names,
        max_combinations=MAX_WEIGHT_COMBINATIONS,
    )

    print("\nTotal feature-weight combinations:", len(candidates))

    best_feature_weights = None
    best_train_summary = None
    best_train_rows = None
    all_results = []
    start_time = time.time()

    for index, feature_weights in enumerate(candidates, start=1):
        train_summary, train_eval_rows = evaluate_from_cache(
            train_feature_cache,
            feature_weights=feature_weights,
        )

        current_selection_score = selection_score(train_summary)

        all_results.append({
            "feature_weights": feature_weights,
            "selection_score_train": current_selection_score,
            "train_summary": train_summary,
        })

        if better_summary(train_summary, best_train_summary):
            best_feature_weights = feature_weights
            best_train_summary = train_summary
            best_train_rows = train_eval_rows

            print(
                f"[best {index}/{len(candidates)}] "
                f"score={current_selection_score:.4f}, "
                f"ndcg@10={train_summary['ndcg@10']:.4f}, "
                f"hit@10={train_summary['hit@10']:.4f}, "
                f"ndcg@20={train_summary['ndcg@20']:.4f}, "
                f"hit@20={train_summary['hit@20']:.4f}",
                flush=True,
            )

        if index % 100 == 0 or index == len(candidates):
            elapsed = time.time() - start_time
            eta = elapsed / index * (len(candidates) - index)
            print(
                f"progress {index}/{len(candidates)}, "
                f"elapsed={elapsed:.1f}s, eta={eta:.1f}s",
                flush=True,
            )

    test_summary, test_eval_rows = evaluate_from_cache(
        test_feature_cache,
        feature_weights=best_feature_weights,
    )

    save_best_weights(
        search_engine=search_engine,
        best_feature_weights=best_feature_weights,
        train_summary=best_train_summary,
        test_summary=test_summary,
        candidate_values=candidate_values,
    )

    final_summary = {
        "best_feature_weights": best_feature_weights,
        "selection_metric": "0.35*ndcg@10 + 0.25*hit@10 + 0.25*ndcg@20 + 0.15*hit@20",
        "selection_score_train": selection_score(best_train_summary),
        "train_summary": best_train_summary,
        "test_summary": test_summary,
    }

    save_json("grid_search_results.json", all_results)
    save_json("grid_best_params.json", best_feature_weights)
    save_json("grid_best_feature_weights.json", best_feature_weights)
    save_json("feature_weight_candidates.json", candidate_values)
    save_json("grid_best_summary.json", final_summary)
    save_json("eval_summary_grid_train.json", best_train_summary)
    save_json("eval_rows_grid_train.json", best_train_rows)
    save_json("eval_summary_grid_test.json", test_summary)
    save_json("eval_rows_grid_test.json", test_eval_rows)
    save_json("eval_summary_grid.json", test_summary)
    save_json("eval_rows_grid.json", test_eval_rows)

    print("\n=== BEST FEATURE WEIGHTS ===")
    print(json.dumps(best_feature_weights, ensure_ascii=False, indent=2))

    print("\n=== TRAIN BEST SUMMARY ===")
    print(json.dumps(best_train_summary, ensure_ascii=False, indent=2))

    print("\n=== TEST SUMMARY ON SMART ===")
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
