from pathlib import Path

from src.core.search_core  import (
    discover_product_files,
    load_products_from_files,
    load_domain_characteristics,
)
from src.parsing.project import normalize_golden_standard


DATA_DIR = Path("data")

GOLDEN_STANDARD_PATH = DATA_DIR / "golden_standard.json"
PARSED_GOLDEN_QUERIES_PATH = DATA_DIR / "golden_queries_parsed.json"
PARSED_GOLDEN_ANSWERS_PATH = DATA_DIR / "golden_answers.json"
DOMAIN_CHARACTERISTICS_PATH = DATA_DIR / "domain_characteristics.json"


def main():
    product_files = discover_product_files(DATA_DIR)
    products = load_products_from_files(product_files)

    print("Loaded products:", len(products))
    print("Product files:", product_files)

    domain_characteristics = load_domain_characteristics(DOMAIN_CHARACTERISTICS_PATH)

    print()
    print("Domain characteristics:")
    for domain, fields in domain_characteristics.items():
        print(domain, "->", fields)

    print()
    print("Start parsing golden_standard.json by LLM")

    queries, answers = normalize_golden_standard(
        input_path=GOLDEN_STANDARD_PATH,
        output_queries_path=PARSED_GOLDEN_QUERIES_PATH,
        output_answers_path=PARSED_GOLDEN_ANSWERS_PATH,
        domain_characteristics_path=DOMAIN_CHARACTERISTICS_PATH,
        use_llm=True,
    )

    print()
    print("Done")
    print("Parsed queries:", len(queries))
    print("Answers:", len(answers))
    print("Saved:", PARSED_GOLDEN_QUERIES_PATH)
    print("Saved:", PARSED_GOLDEN_ANSWERS_PATH)


if __name__ == "__main__":
    main()
