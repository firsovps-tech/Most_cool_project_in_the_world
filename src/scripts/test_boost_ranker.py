import base64
from pathlib import Path

import numpy as np

from src.core.search_cache import load_search_engine_cache
from src.ranking.boost_ranker import (
    load_json,
    save_json,
    load_answers_by_query,
    train_xgb_ranker,
    save_domain_ranker,
    print_boost_feature_weights,
    WeightedBoostProductSearch,
    evaluate_search,
)


DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "search_cache"
TRAINING_CACHE_DIR = DATA_DIR / "training_cache"
MODEL_DIR = Path("trained_boost_ranker")

PARSED_GOLDEN_QUERIES_PATH = DATA_DIR / "golden_queries_parsed.json"
PARSED_GOLDEN_ANSWERS_PATH = DATA_DIR / "golden_answers.json"


def safe_name(value: str) -> str:
    raw = value.encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def load_domain_training_cache(domain):
    domain_dir = TRAINING_CACHE_DIR / safe_name(domain)

    if not domain_dir.exists():
        raise FileNotFoundError(
            f"Нет training cache для домена {domain}: {domain_dir}. "
            "Сначала запусти: python -m src.scripts.build_training_cache"
        )

    X = np.load(domain_dir / "X.npy")
    y = np.load(domain_dir / "y.npy")
    qid = np.load(domain_dir / "qid.npy")
    feature_names = load_json(domain_dir / "feature_names.json")

    return X, y, qid, feature_names


def train_from_training_cache(search_engine):
    weights_by_domain = {}

    for domain in search_engine.domain_configs:
        X, y, qid, feature_names = load_domain_training_cache(domain)

        if len(X) == 0:
            print()
            print("Domain:", domain)
            print("No training rows, skip")
            continue

        print()
        print("Domain:", domain)
        print("Train rows:", len(X))
        print("Feature count:", X.shape[1])
        print("Queries:", len(set(qid.tolist())))
        print("Positive labels:", int(y.sum()))
        print("Feature names:", feature_names)

        if int(y.sum()) == 0:
            raise ValueError(
                f"Positive labels = 0 for domain {domain}. "
                "Проверь data/golden_answers.json и ID товаров."
            )

        model = train_xgb_ranker(
            X=X,
            y=y,
            qid=qid,
        )

        weights_by_domain[domain] = save_domain_ranker(
            model=model,
            feature_names=feature_names,
            output_dir=MODEL_DIR,
            domain=domain,
        )

        print_boost_feature_weights(
            model=model,
            feature_names=feature_names,
        )

    return weights_by_domain


def main():
    if not TRAINING_CACHE_DIR.exists():
        raise FileNotFoundError(
            f"Нет папки {TRAINING_CACHE_DIR}. "
            "Сначала запусти: python -m src.scripts.build_training_cache"
        )

    if not PARSED_GOLDEN_QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Нет файла {PARSED_GOLDEN_QUERIES_PATH}. "
            "Сначала запусти: python -m src.scripts.prepare_golden"
        )

    if not PARSED_GOLDEN_ANSWERS_PATH.exists():
        raise FileNotFoundError(
            f"Нет файла {PARSED_GOLDEN_ANSWERS_PATH}. "
            "Сначала запусти: python -m src.scripts.prepare_golden"
        )

    print("Load search engine from cache:", CACHE_DIR)
    search_engine = load_search_engine_cache(CACHE_DIR)

    print("Loaded products:", len(search_engine.products))
    print("Domains:", list(search_engine.domain_configs.keys()))

    for domain in search_engine.domain_configs:
        print()
        print("Domain:", domain)
        print("Fields:", search_engine.domain_configs[domain].searchable_fields)
        print("Feature names:", search_engine.get_feature_names(domain))

    train_queries = load_json(PARSED_GOLDEN_QUERIES_PATH)
    train_answers = load_answers_by_query(PARSED_GOLDEN_ANSWERS_PATH)

    print()
    print("Loaded parsed golden queries:", len(train_queries))
    print("Loaded parsed golden answers:", len(train_answers))

    weights_by_domain = train_from_training_cache(search_engine)

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
    print("Evaluation summary:")

    for key, value in summary.items():
        print(key, round(value, 4))

    save_json("eval_summary_boost.json", summary)
    save_json("eval_rows_boost.json", rows)

    print()
    print("Models and weights saved to trained_boost_ranker/<domain>/")
    print("Evaluation saved to eval_summary_boost.json and eval_rows_boost.json")


if __name__ == "__main__":
    main()
