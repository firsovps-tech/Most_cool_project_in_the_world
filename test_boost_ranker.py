from search_core import load_products_from_files, save_json
from search_core import NeuralHybridProductSearch
from boost_ranker import (
    load_json,
    load_answers_by_query,
    train_rankers_by_domain,
    XGBoostProductSearch,
    evaluate_search,
)


def main():
    product_files = [
        "data/products_300.json",
        # сюда потом можно добавлять новые файлы:
        # "data/furniture.csv",
        # "data/food.csv",
        # "data/electronics.csv",
    ]

    train_queries_path = "data/queries_300.json"
    train_answers_path = "data/answers_300.json"

    test_queries_path = "data/new_test_queries_120_products300.json"
    test_answers_path = "data/new_test_answers_120_products300.json"

    products = load_products_from_files(product_files)

    train_queries = load_json(train_queries_path)
    train_answers = load_answers_by_query(train_answers_path)

    test_queries = load_json(test_queries_path)
    test_answers = load_answers_by_query(test_answers_path)

    search_engine = NeuralHybridProductSearch(products=products)

    print("Loaded products:", len(products))
    print("Domains:", list(search_engine.domain_configs.keys()))

    for domain in search_engine.domain_configs:
        print()
        print("Domain:", domain)
        print("Fields:", search_engine.domain_configs[domain].searchable_fields)
        print("Feature names:", search_engine.get_feature_names(domain))

    models_by_domain = train_rankers_by_domain(
        search_engine=search_engine,
        queries=train_queries,
        answers_by_query=train_answers,
        output_dir="trained_boost_ranker",
        first_stage_limit=1000,
    )

    boosted_search = XGBoostProductSearch(
        search_engine=search_engine,
        models_by_domain=models_by_domain,
    )

    summary, rows = evaluate_search(
        search_system=boosted_search,
        queries=test_queries,
        answers_by_query=test_answers,
        k=10,
    )

    print()
    print("Evaluation summary:")

    for key, value in summary.items():
        print(key, round(value, 4))

    save_json("eval_summary_boost.json", summary)
    save_json("eval_rows_boost.json", rows)

    print()
    print("Models saved to trained_boost_ranker/<domain>/")
    print("Evaluation saved to eval_summary_boost.json and eval_rows_boost.json")

    print()
    print("Example total search:")

    example_query = test_queries[0]["query"]

    results = boosted_search.search(
        prepared_query=example_query,
        top_k=10,
        per_domain_top_k=100,
        first_stage_limit=1000,
    )

    for item in results:
        print()
        print("ID:", item["product_id"])
        print("Article:", item.get("article", ""))
        print("Domain:", item["domain"])
        print("Score:", item["ranker_score"])
        print("Description:", item["description"])
        print("Price:", item["price"])


if __name__ == "__main__":
    main()
