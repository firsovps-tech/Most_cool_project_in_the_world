from pathlib import Path

from search_core import (
    discover_product_files,
    load_products_from_files,
    save_json,
    field_value,
    require_query_domain,
)
from search_core import NeuralHybridProductSearch
from boost_ranker import (
    load_json,
    load_weights_by_domain,
    WeightedBoostProductSearch,
)
from project import normalize_one_query


DATA_DIR = Path("data")
MODEL_DIR = Path("trained_boost_ranker")

QUERIES_PATH = DATA_DIR / "queries_for_search.json"
OUTPUT_PATH = DATA_DIR / "search_output_top_20.json"

TOP_K = 20
FIRST_STAGE_LIMIT = 1000
FINAL_CANDIDATE_LIMIT = None
USE_LLM_PARSER = True


def build_query_text(item):
    for key in ["query_text", "text", "description", "query"]:
        value = item.get(key)

        if isinstance(value, str) and field_value(value):
            return field_value(value)

    nested_query = item.get("query")

    if isinstance(nested_query, dict):
        for key in ["query_text", "text", "description"]:
            value = field_value(nested_query.get(key))

            if value:
                return value

        characteristics = nested_query.get("characteristics", {})

        if isinstance(characteristics, dict):
            parts = []

            for value in characteristics.values():
                value = field_value(value)

                if value:
                    parts.append(value)

            if parts:
                return " ".join(parts)

    characteristics = item.get("characteristics", {})

    if isinstance(characteristics, dict):
        parts = []

        for value in characteristics.values():
            value = field_value(value)

            if value:
                parts.append(value)

        if parts:
            return " ".join(parts)

    return ""


def normalize_query_item(item):
    query_id = field_value(item.get("query_id")) or field_value(item.get("id"))

    if not query_id:
        raise ValueError("У каждого запроса должен быть query_id")

    if isinstance(item.get("query"), dict):
        prepared_query = dict(item["query"])
    elif item.get("domain") and isinstance(item.get("characteristics"), dict):
        prepared_query = {
            "id": item.get("id", ""),
            "domain": item.get("domain", ""),
            "category": item.get("category", ""),
            "characteristics": item.get("characteristics", {}),
        }

        for key in ["price_min", "price_max"]:
            if key in item:
                prepared_query[key] = item[key]
    else:
        query_text = build_query_text(item)

        if not query_text:
            raise ValueError(f"Пустой текст запроса у {query_id}")

        prepared_query = normalize_one_query(
            query_id=query_id,
            query_text=query_text,
            known_domain=field_value(item.get("domain")) or None,
            use_llm=USE_LLM_PARSER,
        )

    if "characteristics" not in prepared_query:
        prepared_query["characteristics"] = {}

    if "id" not in prepared_query:
        prepared_query["id"] = ""

    require_query_domain(prepared_query)
    query_text = build_query_text(prepared_query)

    return query_id, query_text, prepared_query


def run_search():
    product_files = discover_product_files(DATA_DIR)

    print("Product files:")

    for path in product_files:
        print("-", path)

    products = load_products_from_files(product_files)

    print()
    print("Loaded products:", len(products))

    search_engine = NeuralHybridProductSearch(products=products)

    print("Domains:", list(search_engine.domain_configs.keys()))

    weights_by_domain = load_weights_by_domain(MODEL_DIR)

    print("Loaded weights:", list(weights_by_domain.keys()))

    search_system = WeightedBoostProductSearch(
        search_engine=search_engine,
        weights_by_domain=weights_by_domain,
    )

    queries = load_json(QUERIES_PATH)
    output_rows = []

    for item in queries:
        query_id, query_text, prepared_query = normalize_query_item(item)

        results = search_system.search(
            prepared_query=prepared_query,
            top_k=TOP_K,
            first_stage_limit=FIRST_STAGE_LIMIT,
            final_candidate_limit=FINAL_CANDIDATE_LIMIT,
        )

        top_ids = [
            str(result["product_id"])
            for result in results
        ]

        output_rows.append({
            "query_id": query_id,
            "query_text": query_text,
            "top_20_product_ids": top_ids,
        })

    save_json(OUTPUT_PATH, output_rows)

    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    run_search()
