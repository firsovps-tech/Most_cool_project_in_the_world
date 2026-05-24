
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.search_core import (
    NeuralHybridProductSearch,
    discover_product_files,
    field_value,
    load_products_from_files,
)


OUTPUT_DIR = Path("embedding_database")


def characteristics_to_text(product):
    characteristics = product.get("characteristics", {})

    if not isinstance(characteristics, dict):
        return ""

    parts = []

    for key, value in characteristics.items():
        key = field_value(key)
        value = field_value(value)

        if key and value:
            parts.append(f"{key}: {value}")

    return "; ".join(parts)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== EXPORT EMBEDDING DATABASE ===", flush=True)

    print("Load product files...", flush=True)
    product_files = discover_product_files("data")

    for file_path in product_files:
        print("-", file_path, flush=True)

    print("Load products...", flush=True)
    products = load_products_from_files(product_files)
    print("Products loaded:", len(products), flush=True)

    print("Build/load search engine...", flush=True)
    search_engine = NeuralHybridProductSearch(products=products)

    vectors = []
    rows = []

    print("Collect embeddings from search_engine.general_embeddings_by_domain...", flush=True)

    for domain, domain_embeddings in search_engine.general_embeddings_by_domain.items():
        product_indexes = search_engine.domain_to_product_indexes.get(domain, [])

        print(
            f"Domain: {domain} | products={len(product_indexes)} | embeddings_shape={domain_embeddings.shape}",
            flush=True,
        )

        for local_index, product_index in enumerate(product_indexes):
            product = search_engine.products[product_index]
            vector = domain_embeddings[local_index].astype(np.float32)

            vector_row = len(vectors)
            vectors.append(vector)

            rows.append({
                "vector_row": vector_row,
                "product_index": product_index,
                "id": field_value(product.get("id")),
                "article": field_value(product.get("article")),
                "domain": field_value(product.get("domain")),
                "price": field_value(product.get("price")),
                "description": field_value(product.get("description")),
                "characteristics_text": characteristics_to_text(product),
            })

    if not vectors:
        raise RuntimeError("Embedding vectors were not collected")

    embedding_matrix = np.vstack(vectors).astype(np.float32)
    metadata = pd.DataFrame(rows)

    print("Final matrix shape:", embedding_matrix.shape, flush=True)

    vectors_path = OUTPUT_DIR / "product_embeddings.npy"
    metadata_path = OUTPUT_DIR / "product_embeddings_metadata.csv"
    full_parquet_path = OUTPUT_DIR / "product_embeddings_full.parquet"
    info_path = OUTPUT_DIR / "product_embeddings_info.json"

    print("Save vectors:", vectors_path, flush=True)
    np.save(vectors_path, embedding_matrix)

    print("Save metadata:", metadata_path, flush=True)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8")

    print("Save full parquet:", full_parquet_path, flush=True)
    full_df = metadata.copy()
    full_df["embedding"] = [vector.tolist() for vector in embedding_matrix]
    full_df.to_parquet(full_parquet_path, index=False)

    info = {
        "source": "search_engine.general_embeddings_by_domain",
        "num_products": int(embedding_matrix.shape[0]),
        "embedding_dim": int(embedding_matrix.shape[1]),
        "vectors_file": str(vectors_path),
        "metadata_file": str(metadata_path),
        "full_parquet_file": str(full_parquet_path),
        "domains": sorted(metadata["domain"].dropna().unique().tolist()),
        "description": "Full product embedding database exported from Rita search engine general product embeddings.",
    }

    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved files:", flush=True)
    print("-", vectors_path, flush=True)
    print("-", metadata_path, flush=True)
    print("-", full_parquet_path, flush=True)
    print("-", info_path, flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
