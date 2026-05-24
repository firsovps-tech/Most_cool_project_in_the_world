from pathlib import Path

from src.core import search_cache

from src.ranking.boost_ranker import load_weights_by_domain, WeightedBoostProductSearch
from src.parsing.project import normalize_one_query
from src.core.search_core  import field_value


DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "search_cache"
MODEL_DIR = Path("trained_boost_ranker")

TOP_K = 20
FIRST_STAGE_LIMIT = 1000
FINAL_CANDIDATE_LIMIT = None
USE_LLM_PARSER = True


def clean_text(value):
    if value is None:
        return ""

    text = str(value)

    # Убираем битые Unicode surrogate-символы: \ud800...\udfff
    text = "".join(
        char
        for char in text
        if not (0xD800 <= ord(char) <= 0xDFFF)
    )

    # Убираем всё, что нельзя нормально закодировать в UTF-8
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    return text.strip()


def clean_prepared_query(prepared_query):
    prepared_query["query_id"] = clean_text(prepared_query.get("query_id", "single_query")) or "single_query"
    prepared_query["raw_query_text"] = clean_text(prepared_query.get("raw_query_text", ""))
    prepared_query["query_text"] = clean_text(prepared_query.get("query_text", ""))

    domain = clean_text(prepared_query.get("domain", ""))
    prepared_query["domain"] = domain or "Мебель"

    characteristics = prepared_query.get("characteristics", {})

    if not isinstance(characteristics, dict):
        characteristics = {}

    cleaned_characteristics = {}

    for key, value in characteristics.items():
        clean_key = clean_text(key)
        clean_value = clean_text(value)

        if clean_key:
            cleaned_characteristics[clean_key] = clean_value or "-"

    prepared_query["characteristics"] = cleaned_characteristics

    return prepared_query


def print_results(results):
    top_ids = [str(result["product_id"]) for result in results]

    print()
    print(f"Top {TOP_K} product ids:")
    print(top_ids)

    print()
    print("Detailed results:")

    for position, result in enumerate(results, start=1):
        product_id = field_value(result.get("product_id"))
        score = result.get("ranker_score")
        description = field_value(result.get("description"))

        print(f"{position}. id={product_id} score={score}")
        print(f"   {description[:250]}")
        print()


def main():
    print("Loading cached search engine...")
    search_engine = search_cache.load_search_engine_cache(CACHE_DIR)

    print("Loading trained weights...")
    weights_by_domain = load_weights_by_domain(MODEL_DIR)

    search_system = WeightedBoostProductSearch(
        search_engine=search_engine,
        weights_by_domain=weights_by_domain,
    )

    print()
    print("Ready. Вводи запросы. Для выхода напиши: exit")
    print()

    while True:
        raw_input_text = input("query> ")
        query_text = clean_text(raw_input_text)

        if query_text.lower() in {"exit", "quit", "q"}:
            break

        if not query_text:
            continue

        print("Clean query:", repr(query_text))
        print("Parsing query by LLM...")

        try:
            prepared_query = normalize_one_query(
                query_id="single_query",
                query_text=query_text,
                known_domain="Мебель",
                use_llm=USE_LLM_PARSER,
            )
        except Exception as error:
            print("LLM parser failed, using simple fallback:", error)

            prepared_query = {
                "query_id": "single_query",
                "raw_query_text": query_text,
                "query_text": query_text,
                "domain": "Мебель",
                "characteristics": {
                    "Тип товара": query_text,
                    "Наименование": "-",
                    "Габариты": "-",
                    "Цвет": "-",
                    "Особенности": "-",
                    "Материал": "-",
                    "Product Type": "-",
                    "Dimensions": "-",
                    "Features": "-",
                    "Color": "-",
                    "Material": "-",
                    "Name": "-",
                },
            }

        prepared_query = clean_prepared_query(prepared_query)

        print()
        print("Prepared query:")
        print(prepared_query)

        try:
            results = search_system.search(
                prepared_query=prepared_query,
                top_k=TOP_K,
                first_stage_limit=FIRST_STAGE_LIMIT,
                final_candidate_limit=FINAL_CANDIDATE_LIMIT,
            )

            print_results(results)

        except Exception as error:
            print()
            print("Search failed:", repr(error))
            print("Prepared query was:")
            print(prepared_query)


if __name__ == "__main__":
    main()