from search_core_old_vertion import NeuralHybridProductSearch
from boost_ranker import (
    load_json,
    save_json,
    load_answers_by_query,
    build_training_dataset,
    train_xgb_ranker,
    save_ranker,
    print_boost_feature_weights,
    XGBoostProductSearch,
    evaluate_search,
)


def main():
    # Все файлы с данными лежат в папке data.
    products_path = "../data/products_300.json"

    train_queries_path = "../data/queries_300.json"
    train_answers_path = "../data/answers_300.json"

    test_queries_path = "../data/new_test_queries_120_products300.json"
    test_answers_path = "../data/new_test_answers_120_products300.json"

    products = load_json(products_path)

    train_queries = load_json(train_queries_path)
    train_answers = load_answers_by_query(train_answers_path)

    test_queries = load_json(test_queries_path)
    test_answers = load_answers_by_query(test_answers_path)

    search_engine = NeuralHybridProductSearch(
        products=products,
        domain_name="furniture",
    )

    X, y, qid, debug_rows = build_training_dataset(
        search_engine=search_engine,
        queries=train_queries,
        answers_by_query=train_answers,
        bm25_limit=100,
        embedding_limit=100,
    )

    print("Train rows:", len(X))
    print("Feature count:", X.shape[1])
    print("Queries:", len(set(qid)))
    print("Positive labels:", int(y.sum()))
    print("Feature names:", search_engine.feature_names)

    model = train_xgb_ranker(
        X=X,
        y=y,
        qid=qid,
    )

    save_ranker(
        model=model,
        feature_names=search_engine.feature_names,
        output_dir="../trained_boost_ranker",
    )

    print_boost_feature_weights(
        model=model,
        feature_names=search_engine.feature_names,
    )

    boosted_search = XGBoostProductSearch(
        search_engine=search_engine,
        ranker_model=model,
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

    save_json("../eval_summary_boost.json", summary)
    save_json("../eval_rows_boost.json", rows)

    print()
    print("Model saved to trained_boost_ranker/xgb_ranker.json")
    print("Feature names saved to trained_boost_ranker/feature_names.json")
    print("Boost weights saved to trained_boost_ranker/boost_feature_weights.json")
    print("Evaluation saved to eval_summary_boost.json and eval_rows_boost.json")

    print()
    print("Example boosted search:")

    example_query = test_queries[0]["query"]
    results = boosted_search.search(example_query, top_k=5)

    for item in results:
        print()
        print("ID:", item["product_id"])
        print("Score:", item["ranker_score"])
        print("Description:", item["description"])
        print("Price:", item["price"])


if __name__ == "__main__":
    main()
