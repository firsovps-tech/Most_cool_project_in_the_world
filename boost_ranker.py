import json
from pathlib import Path

import numpy as np
from xgboost import XGBRanker

from search_core import field_value


def load_json(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path = Path(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_answers_by_query(path):
    raw_answers = load_json(path)
    answers_by_query = {}

    for item in raw_answers:
        query_id = item["query_id"]
        relevant_ids = item.get("relevant_ids", [])

        answers_by_query[query_id] = set(
            field_value(product_id)
            for product_id in relevant_ids
        )

    return answers_by_query


def build_training_dataset(
    search_engine,
    queries,
    answers_by_query,
    bm25_limit=100,
    embedding_limit=100,
):
    X = []
    y = []
    qid = []
    debug_rows = []

    for query_number, item in enumerate(queries):
        query_id = item["query_id"]
        query = item["query"]

        relevant_ids = answers_by_query.get(query_id, set())

        candidate_indexes, all_scores = search_engine.get_candidate_indexes(
            prepared_query=query,
            relevant_ids=relevant_ids,
            bm25_limit=bm25_limit,
            embedding_limit=embedding_limit,
        )

        for product_index in candidate_indexes:
            product = search_engine.products[product_index]
            product_id = field_value(product.get("id"))

            label = 1 if product_id in relevant_ids else 0

            feature_vector = search_engine.build_feature_vector(
                prepared_query=query,
                product_index=product_index,
                all_scores=all_scores,
            )

            X.append(feature_vector)
            y.append(label)
            qid.append(query_number)

            debug_rows.append({
                "query_id": query_id,
                "product_id": product_id,
                "label": label,
            })

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(qid, dtype=np.int32),
        debug_rows,
    )


def train_xgb_ranker(X, y, qid):
    if len(X) == 0:
        raise ValueError("Training matrix is empty")

    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
    )

    model.fit(
        X,
        y,
        qid=qid,
        verbose=False,
    )

    return model


def get_boost_feature_weights(model, feature_names):
    raw_importances = np.array(model.feature_importances_, dtype=np.float32)

    total = float(np.sum(raw_importances))

    if total <= 0:
        normalized = np.zeros_like(raw_importances)
    else:
        normalized = raw_importances / total

    result = []

    for name, raw, norm in zip(feature_names, raw_importances, normalized):
        result.append({
            "feature": name,
            "importance": float(raw),
            "normalized_weight": float(norm),
        })

    result.sort(
        key=lambda item: item["normalized_weight"],
        reverse=True,
    )

    return result


def save_ranker(model, feature_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(output_dir / "xgb_ranker.json"))
    save_json(output_dir / "feature_names.json", feature_names)

    boost_weights = get_boost_feature_weights(
        model=model,
        feature_names=feature_names,
    )

    save_json(output_dir / "boost_feature_weights.json", boost_weights)


def print_boost_feature_weights(model, feature_names):
    boost_weights = get_boost_feature_weights(
        model=model,
        feature_names=feature_names,
    )

    print()
    print("Boost feature weights:")

    for item in boost_weights:
        print(
            item["feature"],
            "importance=",
            round(item["importance"], 6),
            "normalized_weight=",
            round(item["normalized_weight"], 6),
        )


class XGBoostProductSearch:
    def __init__(self, search_engine, ranker_model):
        self.search_engine = search_engine
        self.ranker_model = ranker_model

    def search(self, prepared_query, top_k=100, bm25_limit=100, embedding_limit=100):
        candidate_indexes, all_scores = self.search_engine.get_candidate_indexes(
            prepared_query=prepared_query,
            relevant_ids=[],
            bm25_limit=bm25_limit,
            embedding_limit=embedding_limit,
        )

        if not candidate_indexes:
            return []

        X = []

        for product_index in candidate_indexes:
            feature_vector = self.search_engine.build_feature_vector(
                prepared_query=prepared_query,
                product_index=product_index,
                all_scores=all_scores,
            )
            X.append(feature_vector)

        X = np.array(X, dtype=np.float32)
        predicted_scores = self.ranker_model.predict(X)

        results = []

        for product_index, predicted_score in zip(candidate_indexes, predicted_scores):
            product = self.search_engine.products[product_index]

            results.append({
                "product_id": product["id"],
                "description": product.get("description", ""),
                "characteristics": product.get("characteristics", {}),
                "price": product.get("price"),
                "ranker_score": round(float(predicted_score), 6),
            })

        results.sort(
            key=lambda item: item["ranker_score"],
            reverse=True,
        )

        return results[:top_k]


def precision_at_k(results, relevant_ids, k):
    if not results:
        return 0.0

    found = 0

    for item in results[:k]:
        if item["product_id"] in relevant_ids:
            found += 1

    return found / k


def recall_at_k(results, relevant_ids, k):
    if not relevant_ids:
        return 0.0

    found = 0

    for item in results[:k]:
        if item["product_id"] in relevant_ids:
            found += 1

    return found / len(relevant_ids)


def hit_at_k(results, relevant_ids, k):
    for item in results[:k]:
        if item["product_id"] in relevant_ids:
            return 1.0

    return 0.0


def mrr_at_k(results, relevant_ids, k):
    for position, item in enumerate(results[:k], start=1):
        if item["product_id"] in relevant_ids:
            return 1.0 / position

    return 0.0


def dcg_at_k(results, relevant_ids, k):
    score = 0.0

    for position, item in enumerate(results[:k], start=1):
        relevance = 1.0 if item["product_id"] in relevant_ids else 0.0
        score += relevance / np.log2(position + 1)

    return score


def ndcg_at_k(results, relevant_ids, k):
    if not relevant_ids:
        return 0.0

    ideal_count = min(len(relevant_ids), k)
    ideal_dcg = 0.0

    for position in range(1, ideal_count + 1):
        ideal_dcg += 1.0 / np.log2(position + 1)

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(results, relevant_ids, k) / ideal_dcg


def evaluate_search(search_system, queries, answers_by_query, k=10):
    rows = []

    for item in queries:
        query_id = item["query_id"]
        query = item["query"]
        relevant_ids = answers_by_query.get(query_id, set())

        results = search_system.search(
            prepared_query=query,
            top_k=k,
        )

        row = {
            "query_id": query_id,
            f"precision@{k}": precision_at_k(results, relevant_ids, k),
            f"recall@{k}": recall_at_k(results, relevant_ids, k),
            f"hit@{k}": hit_at_k(results, relevant_ids, k),
            f"mrr@{k}": mrr_at_k(results, relevant_ids, k),
            f"ndcg@{k}": ndcg_at_k(results, relevant_ids, k),
            "top_ids": [result["product_id"] for result in results],
            "relevant_ids": sorted(relevant_ids),
        }

        rows.append(row)

    summary = {}

    metric_names = [
        f"precision@{k}",
        f"recall@{k}",
        f"hit@{k}",
        f"mrr@{k}",
        f"ndcg@{k}",
    ]

    for metric_name in metric_names:
        values = [row[metric_name] for row in rows]
        summary[metric_name] = float(np.mean(values)) if values else 0.0

    summary[f"error_hit@{k}"] = 1.0 - summary[f"hit@{k}"]
    summary[f"error_mrr@{k}"] = 1.0 - summary[f"mrr@{k}"]

    return summary, rows
