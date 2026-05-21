from pathlib import Path

from src.core.search_core import discover_product_files, load_products_from_files
from src.core.search_core import NeuralHybridProductSearch
from src.core.search_cache import save_search_engine_cache


DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "search_cache"


def main():
    product_files = discover_product_files(DATA_DIR)

    print("Product files:")
    for path in product_files:
        print("-", path)

    products = load_products_from_files(product_files)

    print()
    print("Loaded products:", len(products))
    print("Build search engine: BM25 + embeddings + FAISS")

    search_engine = NeuralHybridProductSearch(products=products)

    print("Domains:", list(search_engine.domain_configs.keys()))
    print("Save cache...")

    save_search_engine_cache(
        search_engine=search_engine,
        cache_dir=CACHE_DIR,
    )

    print()
    print("Done")
    print("Cache dir:", CACHE_DIR)


if __name__ == "__main__":
    main()
