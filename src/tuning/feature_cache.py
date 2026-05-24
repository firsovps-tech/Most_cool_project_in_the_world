import pickle
from pathlib import Path

import numpy as np


def build_feature_cache(search_engine, gold_rows, first_stage_limit=1000, final_candidate_limit=None):
    cache = []

    for number, row in enumerate(gold_rows, start=1):
        prepared_query = row["prepared_query"]
        domain = prepared_query["domain"]
        relevant_ids = set(row["ideal_product_ids"])

        print(
            f"[{number}/{len(gold_rows)}] candidates for "
            f"{row['dataset']}::{row['query_id']} | {row['query_text']}",
            flush=True,
        )

        candidate_indexes, all_scores = search_engine.get_candidate_indexes_for_domain(
            prepared_query=prepared_query,
            domain=domain,
            relevant_ids=relevant_ids,
            first_stage_limit=first_stage_limit,
            final_candidate_limit=final_candidate_limit,
        )

        feature_names = search_engine.get_feature_names(domain)
        product_ids = []
        product_indexes = []
        feature_vectors = []

        for product_index in candidate_indexes:
            product = search_engine.products[product_index]
            product_indexes.append(product_index)
            product_ids.append(str(product.get("id", "")))
            feature_vectors.append(
                search_engine.build_feature_vector(
                    prepared_query=prepared_query,
                    product_index=product_index,
                    all_scores=all_scores,
                )
            )

        cache.append({
            "dataset": row["dataset"],
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "corrected_query_text": prepared_query.get("corrected_query_text", row["query_text"]),
            "prepared_query": prepared_query,
            "ideal_product_ids": row["ideal_product_ids"],
            "domain": domain,
            "feature_names": feature_names,
            "product_ids": product_ids,
            "product_indexes": product_indexes,
            "features": np.asarray(feature_vectors, dtype=np.float32),
        })

    return cache


def get_or_build_feature_cache(cache_path, search_engine, gold_rows, first_stage_limit=1000, final_candidate_limit=None):
    cache_path = Path(cache_path)

    if cache_path.exists():
        print(f"Load feature cache: {cache_path}", flush=True)
        with cache_path.open("rb") as file:
            return pickle.load(file)

    cache = build_feature_cache(
        search_engine=search_engine,
        gold_rows=gold_rows,
        first_stage_limit=first_stage_limit,
        final_candidate_limit=final_candidate_limit,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(cache, file)

    print(f"Saved feature cache: {cache_path}", flush=True)
    return cache
