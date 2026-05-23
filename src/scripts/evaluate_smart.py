from pathlib import Path

from src.core.search_cache import load_search_engine_cache
from src.core.search_core import load_json, save_json
from src.scripts.tune_weights_grid import (
    TEST_GOLD_PATH,
    TEST_FEATURE_CACHE,
    FIRST_STAGE_LIMIT,
    FINAL_CANDIDATE_LIMIT,
    load_gold_rows,
    evaluate_from_cache,
)
from src.tuning.feature_cache import get_or_build_feature_cache

CACHE_DIR = Path("data") / "search_cache"
BEST_PARAMS_PATH = Path("grid_best_params.json")


def main():
    print("=== Ordinary Smart evaluation ===")
    print("Load best params:", BEST_PARAMS_PATH)
    best_params = load_json(BEST_PARAMS_PATH)

    print("Load cached Rita search engine:", CACHE_DIR)
    search_engine = load_search_engine_cache(CACHE_DIR)

    print("Load Smart rows:", TEST_GOLD_PATH)
    smart_rows = load_gold_rows(TEST_GOLD_PATH, TEST_GOLD_PATH.name)

    print("Build/load Smart feature cache:", TEST_FEATURE_CACHE)
    smart_feature_cache = get_or_build_feature_cache(
        TEST_FEATURE_CACHE,
        search_engine=search_engine,
        gold_rows=smart_rows,
        first_stage_limit=FIRST_STAGE_LIMIT,
        final_candidate_limit=FINAL_CANDIDATE_LIMIT,
    )

    summary, rows = evaluate_from_cache(smart_feature_cache, best_params)

    save_json("eval_summary_smart.json", summary)
    save_json("eval_rows_smart.json", rows)

    print("Saved:")
    print("- eval_summary_smart.json")
    print("- eval_rows_smart.json")
    print(summary)


if __name__ == "__main__":
    main()
