import json
from pathlib import Path

from src.core.search_core import field_value, require_query_domain


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def weights_list_to_dict(weights_list):
    result = {}
    for item in weights_list:
        feature = item.get("feature")
        if not feature:
            continue
        result[feature] = float(item.get("normalized_weight", item.get("importance", 0.0)))
    return result


def load_weights_by_domain(model_dir="trained_boost_ranker"):
    model_dir = Path(model_dir)
    result = {}

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Папка с весами не найдена: {model_dir}. "
            "Сначала запусти: python -m src.scripts.tune_weights_grid"
        )

    for domain_dir in sorted(model_dir.iterdir()):
        if not domain_dir.is_dir():
            continue

        weights_path = domain_dir / "boost_feature_weights.json"
        if not weights_path.exists():
            continue

        data = load_json(weights_path)
        if isinstance(data, list):
            result[domain_dir.name] = weights_list_to_dict(data)
        elif isinstance(data, dict):
            result[domain_dir.name] = {str(k): float(v) for k, v in data.items()}

    if not result:
        raise FileNotFoundError(
            f"В папке {model_dir} не найдено boost_feature_weights.json."
        )

    return result


class WeightedBoostProductSearch:
    """
    Финальный ранжировщик Rita-стиля.
    Формула score не меняется:
        score = sum(feature_value * weight)
    Веса берутся из trained_boost_ranker/<domain>/boost_feature_weights.json.
    """

    def __init__(self, search_engine, weights_by_domain):
        self.search_engine = search_engine
        self.weights_by_domain = weights_by_domain

    def _weighted_score(self, feature_names, feature_vector, domain):
        weights = self.weights_by_domain.get(domain, {})
        score = 0.0

        for name, value in zip(feature_names, feature_vector):
            score += float(value) * float(weights.get(name, 0.0))

        return score

    def search(
        self,
        prepared_query,
        top_k=20,
        first_stage_limit=1000,
        final_candidate_limit=None,
    ):
        domain = require_query_domain(prepared_query)

        if domain not in self.weights_by_domain:
            return []

        candidate_indexes, all_scores = self.search_engine.get_candidate_indexes_for_domain(
            prepared_query=prepared_query,
            domain=domain,
            relevant_ids=[],
            first_stage_limit=first_stage_limit,
            final_candidate_limit=final_candidate_limit,
        )

        if not candidate_indexes:
            return []

        feature_names = self.search_engine.get_feature_names(domain)
        results = []

        for product_index in candidate_indexes:
            feature_vector = self.search_engine.build_feature_vector(
                prepared_query=prepared_query,
                product_index=product_index,
                all_scores=all_scores,
            )

            score = self._weighted_score(
                feature_names=feature_names,
                feature_vector=feature_vector,
                domain=domain,
            )

            product = self.search_engine.products[product_index]

            results.append({
                "product_id": field_value(product.get("id")),
                "article": field_value(product.get("article")),
                "domain": domain,
                "description": field_value(product.get("description")),
                "characteristics": product.get("characteristics", {}),
                "price": product.get("price"),
                "ranker_score": round(float(score), 6),
            })

        results.sort(key=lambda item: item["ranker_score"], reverse=True)
        return results[:top_k]
