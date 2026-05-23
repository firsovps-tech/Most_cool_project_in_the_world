import json
from pathlib import Path

import numpy as np

from src.core.search_cache import load_search_engine_cache
from src.core.search_core import field_value, save_json, load_json
from src.evaluation.llm_judge import evaluate_pair, load_cache
from src.evaluation.metrics import RankingMetrics
from src.ranking.boost_ranker import load_weights_by_domain, WeightedBoostProductSearch
from src.scripts.run_search_top20 import normalize_query_item

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "search_cache"
MODEL_DIR = Path("trained_boost_ranker")
SMART_GOLD_PATH = DATA_DIR / "smart_gold_standard.json"
OUTPUT_ROWS_PATH = Path("eval_llm_rows_smart.json")
OUTPUT_SUMMARY_PATH = Path("eval_llm_summary_smart.json")

TOP_K = 20
FIRST_STAGE_LIMIT = 1000
FINAL_CANDIDATE_LIMIT = None


def build_product_maps(products):
    by_id = {}
    article_to_id = {}

    for product in products:
        product_id = field_value(product.get("id"))
        if not product_id:
            continue

        by_id[product_id] = product

        article = field_value(product.get("article"))
        if article:
            article_to_id[article.strip().upper()] = product_id

    return by_id, article_to_id


def load_smart_rows():
    rows = load_json(SMART_GOLD_PATH)
    result = []

    for item in rows:
        query_id = field_value(item.get("query_id"))
        query_text = field_value(item.get("query_text")) or field_value(item.get("query"))
        if query_id and query_text:
            result.append({
                "query_id": query_id,
                "query_text": query_text,
                "ideal_product_ids": [field_value(x) for x in item.get("ideal_product_ids", []) if field_value(x)],
            })

    return result


def summarize_llm_rows(rows):
    metrics = ["llm_ndcg@20", "llm_precision@5"]
    summary = {}

    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        summary[metric] = float(np.mean(values)) if values else 0.0

    summary["queries"] = len(rows)
    summary["dataset"] = SMART_GOLD_PATH.name
    return summary


def main():
    print("=== Separate LLM metrics on Smart ===")
    print("Load cached Rita search engine:", CACHE_DIR)
    search_engine = load_search_engine_cache(CACHE_DIR)

    weights_by_domain = load_weights_by_domain(MODEL_DIR)
    search_system = WeightedBoostProductSearch(search_engine, weights_by_domain)

    product_by_id, article_to_id = build_product_maps(search_engine.products)
    smart_rows = load_smart_rows()
    judge_cache = load_cache()
    metric = RankingMetrics(k=TOP_K)

    output_rows = []

    for number, row in enumerate(smart_rows, start=1):
        print(f"[{number}/{len(smart_rows)}] {row['query_id']} | {row['query_text']}", flush=True)

        query_id, query_text, prepared_query = normalize_query_item(row)
        results = search_system.search(
            prepared_query=prepared_query,
            top_k=TOP_K,
            first_stage_limit=FIRST_STAGE_LIMIT,
            final_candidate_limit=FINAL_CANDIDATE_LIMIT,
        )

        llm_scores = []
        llm_reasons = []
        top_ids = []

        for result in results[:TOP_K]:
            product_id = field_value(result.get("product_id"))
            product = product_by_id.get(product_id, {})
            top_ids.append(product_id)

            characteristics = product.get("characteristics", {})
            product_name = (
                field_value(characteristics.get("Наименование"))
                or field_value(characteristics.get("Name"))
                or field_value(product.get("article"))
                or product_id
            )
            product_desc = field_value(product.get("description"))
            product_article = field_value(product.get("article"))

            score, reason = evaluate_pair(
                query_id=query_id,
                query_text=row["query_text"],
                product_id=product_id,
                product_name=product_name,
                product_desc=product_desc,
                cache=judge_cache,
                product_article=product_article,
                article_to_id=article_to_id,
            )

            if score is None:
                score = 0

            llm_scores.append(int(score))
            llm_reasons.append(reason)

        llm_metrics = metric.calculate_llm(llm_scores)

        output_rows.append({
            "dataset": SMART_GOLD_PATH.name,
            "query_id": query_id,
            "query_text": row["query_text"],
            "top_20_product_ids": top_ids,
            "llm_scores": llm_scores,
            "llm_reasons": llm_reasons,
            "llm_ndcg@20": llm_metrics["ndcg"],
            "llm_precision@5": llm_metrics["precision@5"],
        })

    summary = summarize_llm_rows(output_rows)
    save_json(OUTPUT_SUMMARY_PATH, summary)
    save_json(OUTPUT_ROWS_PATH, output_rows)

    print("Saved:")
    print("-", OUTPUT_SUMMARY_PATH)
    print("-", OUTPUT_ROWS_PATH)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
