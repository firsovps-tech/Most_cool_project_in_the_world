from pathlib import Path

from search_core import (
    discover_product_files,
    load_products_from_files,
    save_json,
)
from search_core import NeuralHybridProductSearch
from boost_ranker import (
    load_answers_by_query,
    train_rankers_by_domain,
    WeightedBoostProductSearch,
    evaluate_search,
)
from project import normalize_golden_standard


DATA_DIR = Path("data")
GOLDEN_STANDARD_PATH = DATA_DIR / "golden_standard.json"
PARSED_GOLDEN_QUERIES_PATH = DATA_DIR / "golden_queries_parsed.json"
PARSED_GOLDEN_ANSWERS_PATH = DATA_DIR / "golden_answers.json"


# Если хочешь не дергать LLM повторно, поставь False после первого успешного запуска.
REPARSE_GOLDEN_STANDARD = True
USE_LLM_PARSER = True


def main():
    product_files = discover_product_files(DATA_DIR)
    products = load_products_from_files(product_files)

    search_engine = NeuralHybridProductSearch(products=products)

    print("Loaded products:", len(products))
    print("Product files:", product_files)
    print("Domains:", list(search_engine.domain_configs.keys()))

    for domain in search_engine.domain_configs:
        print()
        print("Domain:", domain)
        print("Fields:", search_engine.domain_configs[domain].searchable_fields)
        print("Feature names:", search_engine.get_feature_names(domain))

    if REPARSE_GOLDEN_STANDARD or not PARSED_GOLDEN_QUERIES_PATH.exists():
        train_queries, train_answers_list = normalize_golden_standard(
            input_path=GOLDEN_STANDARD_PATH,
            output_queries_path=PARSED_GOLDEN_QUERIES_PATH,
            output_answers_path=PARSED_GOLDEN_ANSWERS_PATH,
            use_llm=USE_LLM_PARSER,
        )
    else:
        from boost_ranker import load_json
        train_queries = load_json(PARSED_GOLDEN_QUERIES_PATH)
        train_answers_list = load_json(PARSED_GOLDEN_ANSWERS_PATH)

    train_answers = {
        item["query_id"]: set(str(product_id) for product_id in item.get("relevant_ids", []))
        for item in train_answers_list
    }

    weights_by_domain = train_rankers_by_domain(
        search_engine=search_engine,
        queries=train_queries,
        answers_by_query=train_answers,
        output_dir="trained_boost_ranker",
        first_stage_limit=1000,
        final_candidate_limit=None,
    )

    search_system = WeightedBoostProductSearch(
        search_engine=search_engine,
        weights_by_domain=weights_by_domain,
    )

    summary, rows = evaluate_search(
        search_system=search_system,
        queries=train_queries,
        answers_by_query=train_answers,
        k=10,
    )

    print()
    print("Evaluation on golden_standard:")

    for key, value in summary.items():
        print(key, round(value, 4))

    save_json("eval_summary_boost.json", summary)
    save_json("eval_rows_boost.json", rows)

    print()
    print("Saved:")
    print("- data/domain_characteristics.json")
    print("- data/golden_queries_parsed.json")
    print("- data/golden_answers.json")
    print("- trained_boost_ranker/<domain>/boost_feature_weights.json")
    print("- eval_summary_boost.json")
    print("- eval_rows_boost.json")


if __name__ == "__main__":
    main()
