import base64
from pathlib import Path

import numpy as np

from src.core.search_cache import load_search_engine_cache
from src.ranking.boost_ranker import (
    load_json,
    save_json,
    load_answers_by_query,
    build_training_dataset_for_domain,
)


DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "search_cache"
TRAINING_CACHE_DIR = DATA_DIR / "training_cache"

PARSED_GOLDEN_QUERIES_PATH = DATA_DIR / "golden_queries_parsed.json"
PARSED_GOLDEN_ANSWERS_PATH = DATA_DIR / "golden_answers.json"

FIRST_STAGE_LIMIT = 1000
FINAL_CANDIDATE_LIMIT = None


def safe_name(value: str) -> str:
    raw = value.encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def save_domain_training_cache(domain, X, y, qid, debug_rows, feature_names):
    domain_dir = TRAINING_CACHE_DIR / safe_name(domain)
    domain_dir.mkdir(parents=True, exist_ok=True)

    np.save(domain_dir / "X.npy", X)
    np.save(domain_dir / "y.npy", y)
    np.save(domain_dir / "qid.npy", qid)

    save_json(domain_dir / "debug_rows.json", debug_rows)
    save_json(domain_dir / "feature_names.json", feature_names)
    save_json(domain_dir / "metadata.json", {
        "domain": domain,
        "train_rows": int(len(X)),
        "feature_count": int(X.shape[1]) if len(X) else 0,
        "queries": int(len(set(qid.tolist()))) if len(qid) else 0,
        "positive_labels": int(np.sum(y)) if len(y) else 0,
        "first_stage_limit": FIRST_STAGE_LIMIT,
        "final_candidate_limit": FINAL_CANDIDATE_LIMIT,
    })


def main():
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

    queries = load_json(PARSED_GOLDEN_QUERIES_PATH)
    answers_by_query = load_answers_by_query(PARSED_GOLDEN_ANSWERS_PATH)

    print()
    print("Loaded parsed golden queries:", len(queries))
    print("Loaded parsed golden answers:", len(answers_by_query))
    print("Build training cache...")

    TRAINING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    domains_metadata = []

    for domain in search_engine.domain_configs:
        print()
        print("Domain:", domain)

        X, y, qid, debug_rows = build_training_dataset_for_domain(
            search_engine=search_engine,
            queries=queries,
            answers_by_query=answers_by_query,
            domain=domain,
            first_stage_limit=FIRST_STAGE_LIMIT,
            final_candidate_limit=FINAL_CANDIDATE_LIMIT,
        )

        if len(X) == 0:
            print("No training rows, skip")
            continue

        feature_names = search_engine.get_feature_names(domain)

        print("Train rows:", len(X))
        print("Feature count:", X.shape[1])
        print("Queries:", len(set(qid.tolist())))
        print("Positive labels:", int(y.sum()))
        print("Feature names:", feature_names)

        save_domain_training_cache(
            domain=domain,
            X=X,
            y=y,
            qid=qid,
            debug_rows=debug_rows,
            feature_names=feature_names,
        )

        domains_metadata.append({
            "domain": domain,
            "safe_domain": safe_name(domain),
            "train_rows": int(len(X)),
            "feature_count": int(X.shape[1]),
            "queries": int(len(set(qid.tolist()))),
            "positive_labels": int(y.sum()),
        })

    save_json(TRAINING_CACHE_DIR / "metadata.json", {
        "domains": domains_metadata,
        "source_queries": str(PARSED_GOLDEN_QUERIES_PATH),
        "source_answers": str(PARSED_GOLDEN_ANSWERS_PATH),
        "search_cache": str(CACHE_DIR),
    })

    print()
    print("Done")
    print("Training cache dir:", TRAINING_CACHE_DIR)


if __name__ == "__main__":
    main()
