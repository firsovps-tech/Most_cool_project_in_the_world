from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.manifold import TSNE

import search_cache
from search_core import field_value


DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "search_cache"

OUTPUT_IMAGE = "vector_db_visualization_from_cache.png"


def get_product_type(product):
    characteristics = product.get("characteristics", {})

    for key in ["Тип товара", "Product Type", "Наименование", "Name"]:
        value = field_value(characteristics.get(key))
        if value:
            return value.lower()

    return "unknown"


def reduce_to_2d(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)

    pca_components = min(50, vectors.shape[1], len(vectors) - 1)

    print("PCA components:", pca_components)

    pca = PCA(n_components=pca_components, random_state=42)
    reduced = pca.fit_transform(vectors)

    if HAS_UMAP:
        print("Using UMAP...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(reduced)
        method_name = "UMAP"
    else:
        print("Using t-SNE...")
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42,
            init="pca",
        )
        coords = reducer.fit_transform(reduced)
        method_name = "t-SNE"

    return coords, method_name


def main():
    print("Loading cached search engine...")
    engine = search_cache.load_search_engine_cache(CACHE_DIR)

    print("Domains:", list(engine.domain_configs.keys()))

    # У нас сейчас основной домен — Мебель
    domain = "Мебель"

    if domain not in engine.general_embeddings_by_domain:
        domain = list(engine.general_embeddings_by_domain.keys())[0]

    print("Using domain:", domain)

    vectors = engine.general_embeddings_by_domain[domain]
    product_indexes = engine.domain_to_product_indexes[domain]

    products = [
        engine.products[index]
        for index in product_indexes
    ]

    print("Products in domain:", len(products))
    print("Vectors shape:", vectors.shape)

    print("Reducing vectors to 2D...")
    coords, method_name = reduce_to_2d(vectors)

    type_labels = [
        get_product_type(product)
        for product in products
    ]

    counter = Counter(type_labels)
    top_types = [name for name, _ in counter.most_common(10)]

    plot_labels = [
        label if label in top_types else "other"
        for label in type_labels
    ]

    unique_labels = sorted(set(plot_labels))

    print("Top labels:")
    for label, count in counter.most_common(15):
        print(label, count)

    print("Drawing plot...")

    plt.figure(figsize=(14, 10))

    for label in unique_labels:
        mask = np.array([current == label for current in plot_labels])

        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.7,
            label=label,
        )

    plt.title(f"Визуализация векторной базы товаров из cache ({method_name})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(fontsize=8, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_IMAGE, dpi=200)

    print("Saved:", OUTPUT_IMAGE)


if __name__ == "__main__":
    main()