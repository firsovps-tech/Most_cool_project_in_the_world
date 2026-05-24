import json
import time
from pathlib import Path

import numpy as np

from src.core.search_cache import load_search_engine_cache
from src.core.search_core import save_json, field_value
from src.evaluation.metrics import RankingMetrics
from src.parsing.project import normalize_one_query
from src.tuning.feature_cache import get_or_build_feature_cache
from src.tuning.weight_grid import generate_weight_candidates, WEIGHT_GRID, MAX_WEIGHT_COMBINATIONS

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "search_cache"
MODEL_DIR = Path("trained_boost_ranker")
FEATURE_CACHE_DIR = Path("feature_cache")

TRAIN_GOLD_PATH = DATA_DIR / "overnight_gold_standard.json"
TEST_GOLD_PATH = DATA_DIR / "smart_gold_standard.json"
TRAIN_FEATURE_CACHE = FEATURE_CACHE_DIR / "overnight_feature_cache.pkl"
TEST_FEATURE_CACHE = FEATURE_CACHE_DIR / "smart_feature_cache.pkl"

TOP_K_VALUES = [5, 10, 20]
FIRST_STAGE_LIMIT = 1000
FINAL_CANDIDATE_LIMIT = None
USE_LLM_PARSER = True


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_gold_rows(path, dataset_name):
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
            "ideal_product_ids": [field_value(product_id) for product_id in ideal_ids if field_value(product_id)],
        })

    if not rows:
        raise ValueError(f"Не удалось загрузить gold rows из {path}")

    return rows


def make_weights_for_domain(feature_names, params):
    weights = {}

    for name in feature_names:
        if name == "exact_id_score":
            weights[name] = params["w_exact_id"]
        elif name == "bm25_score":
            weights[name] = params["w_bm25"]
        elif name == "general_embedding_score":
            weights[name] = params["w_general_embedding"]
        elif name == "detail_embedding_score":
            weights[name] = params["w_detail_embedding"]
        elif name == "price_score":
            weights[name] = params["w_price"]
        elif name.endswith("_embedding_score"):
            weights[name] = params["w_field_embedding"]
        elif name.startswith("product_has_"):
            weights[name] = params["w_product_has_field"]
        else:
            weights[name] = 0.0

    return weights


def weights_dict_to_saved_list(weights):
    total = sum(abs(value) for value in weights.values())
    result = []

    for feature, value in weights.items():
        normalized = value / total if total > 0 else 0.0
        result.append({
            "feature": feature,
            "importance": float(value),
            "normalized_weight": float(normalized),
        })

    result.sort(key=lambda item: item["normalized_weight"], reverse=True)
    return result


def rank_from_cache(query_cache, params, top_k=20):
    product_ids = query_cache["product_ids"]
    features = query_cache["features"]

    if len(product_ids) == 0:
        return []

    weights_dict = make_weights_for_domain(query_cache["feature_names"], params)
    weights_vector = np.asarray(
        [weights_dict.get(name, 0.0) for name in query_cache["feature_names"]],
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


def evaluate_from_cache(feature_cache, params):
    rows = []

    metric_by_k = {
        k: RankingMetrics(k=k)
        for k in TOP_K_VALUES
    }

    for query_cache in feature_cache:
        ranked_rows = rank_from_cache(query_cache, params=params, top_k=max(TOP_K_VALUES))
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
    # Оптимизируем верх выдачи и контролируем top-20.
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


def save_best_weights(search_engine, best_params, train_summary, test_summary):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    saved_summary = {
        "selection_metric": "0.35*ndcg@10 + 0.25*hit@10 + 0.25*ndcg@20 + 0.15*hit@20",
        "selection_score_train": selection_score(train_summary),
        "train_summary": train_summary,
        "test_summary": test_summary,
        "best_params": best_params,
    }

    for domain in search_engine.domain_configs:
        feature_names = search_engine.get_feature_names(domain)
        weights = make_weights_for_domain(feature_names, best_params)
        domain_dir = MODEL_DIR / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        save_json(domain_dir / "feature_names.json", feature_names)
        save_json(domain_dir / "boost_feature_weights.json", weights_dict_to_saved_list(weights))
        save_json(domain_dir / "grid_best_params.json", best_params)
        save_json(domain_dir / "grid_best_summary.json", saved_summary)


def main():
    print("=== WEIGHT TUNING: optimized grid/random search up to 10000 combinations ===")
    print("Train dataset:", TRAIN_GOLD_PATH)
    print("Test dataset:", TEST_GOLD_PATH)
    print("Weight candidates per weight:")
    for name, values in WEIGHT_GRID.items():
        print(f"- {name}: {len(values)} values")

    print("\nLoad cached Rita search engine:", CACHE_DIR)
    search_engine = load_search_engine_cache(CACHE_DIR)

    print("\nPrepare train gold rows...")
    train_rows = load_gold_rows(TRAIN_GOLD_PATH, TRAIN_GOLD_PATH.name)
    print("Train queries:", len(train_rows))

    print("\nPrepare test gold rows...")
    test_rows = load_gold_rows(TEST_GOLD_PATH, TEST_GOLD_PATH.name)
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

    candidates = generate_weight_candidates(max_combinations=MAX_WEIGHT_COMBINATIONS)
    print("\nTotal weight combinations:", len(candidates))

    best_params = None
    best_train_summary = None
    best_train_rows = None
    all_results = []
    start_time = time.time()

    for index, params in enumerate(candidates, start=1):
        train_summary, train_eval_rows = evaluate_from_cache(train_feature_cache, params)
        current_selection_score = selection_score(train_summary)

        all_results.append({
            "params": params,
            "selection_score_train": current_selection_score,
            "train_summary": train_summary,
        })

        if better_summary(train_summary, best_train_summary):
            best_params = params
            best_train_summary = train_summary
            best_train_rows = train_eval_rows
            print(
                f"[best {index}/{len(candidates)}] "
                f"score={current_selection_score:.4f}, "
                f"ndcg@10={train_summary['ndcg@10']:.4f}, "
                f"hit@10={train_summary['hit@10']:.4f}, "
                f"ndcg@20={train_summary['ndcg@20']:.4f}, "
                f"hit@20={train_summary['hit@20']:.4f}, "
                f"params={params}",
                flush=True,
            )

        if index % 100 == 0 or index == len(candidates):
            elapsed = time.time() - start_time
            eta = elapsed / index * (len(candidates) - index)
            print(f"progress {index}/{len(candidates)}, elapsed={elapsed:.1f}s, eta={eta:.1f}s", flush=True)

    test_summary, test_eval_rows = evaluate_from_cache(test_feature_cache, best_params)
    save_best_weights(search_engine, best_params, best_train_summary, test_summary)

    final_summary = {
        "best_params": best_params,
        "selection_metric": "0.35*ndcg@10 + 0.25*hit@10 + 0.25*ndcg@20 + 0.15*hit@20",
        "selection_score_train": selection_score(best_train_summary),
        "train_summary": best_train_summary,
        "test_summary": test_summary,
    }

    save_json("grid_search_results.json", all_results)
    save_json("grid_best_params.json", best_params)
    save_json("grid_best_summary.json", final_summary)
    save_json("eval_summary_grid_train.json", best_train_summary)
    save_json("eval_rows_grid_train.json", best_train_rows)
    save_json("eval_summary_grid_test.json", test_summary)
    save_json("eval_rows_grid_test.json", test_eval_rows)
    save_json("eval_summary_grid.json", test_summary)
    save_json("eval_rows_grid.json", test_eval_rows)

    print("\n=== BEST PARAMS ===")
    print(json.dumps(best_params, ensure_ascii=False, indent=2))
    print("\n=== TRAIN BEST SUMMARY ===")
    print(json.dumps(best_train_summary, ensure_ascii=False, indent=2))
    print("\n=== TEST SUMMARY ON SMART ===")
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
