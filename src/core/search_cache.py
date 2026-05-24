import base64
import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.core.search_core import (
    NeuralHybridProductSearch,
    SearchDomainConfig,
    tokenize,
)


CACHE_VERSION = 1


def _safe_name(value: str) -> str:
    raw = value.encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_search_engine_cache(search_engine: NeuralHybridProductSearch, cache_dir="data/search_cache") -> None:
    """
    Сохраняет готовую поисковую базу:
    товары, domain configs, feature names, embeddings товаров и FAISS.
    Embedding-модель не сохраняется, она нужна только для кодирования новых запросов.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    domains = list(search_engine.domain_configs.keys())

    _save_json(cache_dir / "metadata.json", {
        "cache_version": CACHE_VERSION,
        "product_count": len(search_engine.products),
        "domains": domains,
    })

    _save_json(cache_dir / "products.json", search_engine.products)

    domain_configs = {}
    for domain, config in search_engine.domain_configs.items():
        domain_configs[domain] = {
            "searchable_fields": config.searchable_fields,
            "missing_field_scores": config.missing_field_scores,
        }

    _save_json(cache_dir / "domain_configs.json", domain_configs)
    _save_json(cache_dir / "feature_names_by_domain.json", search_engine.feature_names_by_domain)
    _save_json(cache_dir / "domain_to_product_indexes.json", search_engine.domain_to_product_indexes)

    embeddings_dir = cache_dir / "embeddings"
    faiss_dir = cache_dir / "faiss"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        safe_domain = _safe_name(domain)

        general_embeddings = search_engine.general_embeddings_by_domain.get(domain)
        if general_embeddings is not None:
            np.save(embeddings_dir / f"general_{safe_domain}.npy", general_embeddings)

        field_embeddings = search_engine.field_embeddings_by_domain.get(domain, {})
        if field_embeddings:
            safe_field_names = []
            arrays = {}

            for field_name, vectors in field_embeddings.items():
                safe_field = _safe_name(field_name)
                safe_field_names.append({
                    "field_name": field_name,
                    "safe_field": safe_field,
                })
                arrays[safe_field] = vectors

            np.savez_compressed(embeddings_dir / f"fields_{safe_domain}.npz", **arrays)
            _save_json(embeddings_dir / f"fields_{safe_domain}_names.json", safe_field_names)

        faiss_index = search_engine.faiss_general_index_by_domain.get(domain)
        if faiss_index is not None:
            faiss.write_index(faiss_index, str(faiss_dir / f"general_{safe_domain}.index"))

    print(f"Search cache saved to: {cache_dir}")


def _load_embedding_model():
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=False,
    )


def load_search_engine_cache(cache_dir="data/search_cache") -> NeuralHybridProductSearch:
    """
    Загружает поисковую базу без пересчета embeddings товаров.
    __init__ не вызывается, чтобы не строить базу заново.
    """

    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache not found: {cache_dir}. Сначала запусти: python -m src.scripts.build_search_cache"
        )

    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Cache metadata not found: {metadata_path}. Пересобери cache."
        )

    metadata = _load_json(metadata_path)

    if metadata.get("cache_version") != CACHE_VERSION:
        raise RuntimeError(
            f"Unsupported cache version: {metadata.get('cache_version')}. "
            f"Expected: {CACHE_VERSION}. Пересобери cache."
        )

    engine = NeuralHybridProductSearch.__new__(NeuralHybridProductSearch)

    engine.products = _load_json(cache_dir / "products.json")

    raw_domain_configs = _load_json(cache_dir / "domain_configs.json")
    engine.domain_configs = {}

    for domain, config in raw_domain_configs.items():
        engine.domain_configs[domain] = SearchDomainConfig(
            searchable_fields=list(config.get("searchable_fields", [])),
            missing_field_scores=dict(config.get("missing_field_scores", {})),
        )

    engine.feature_names_by_domain = _load_json(cache_dir / "feature_names_by_domain.json")

    raw_domain_to_product_indexes = _load_json(cache_dir / "domain_to_product_indexes.json")
    engine.domain_to_product_indexes = {
        domain: [int(index) for index in indexes]
        for domain, indexes in raw_domain_to_product_indexes.items()
    }

    # BM25 быстрый, его можно перестроить. Главное — не пересчитывать embeddings.
    engine.bm25_by_domain = {}
    for domain, product_indexes in engine.domain_to_product_indexes.items():
        tokens = []

        for product_index in product_indexes:
            product = engine.products[product_index]
            tokens.append(tokenize(product.get("description", "")))

        engine.bm25_by_domain[domain] = BM25Okapi(tokens)

    embeddings_dir = cache_dir / "embeddings"
    faiss_dir = cache_dir / "faiss"

    engine.general_embeddings_by_domain = {}
    engine.field_embeddings_by_domain = {}
    engine.faiss_general_index_by_domain = {}

    for domain in engine.domain_configs:
        safe_domain = _safe_name(domain)

        general_path = embeddings_dir / f"general_{safe_domain}.npy"
        if general_path.exists():
            engine.general_embeddings_by_domain[domain] = np.load(general_path)
        else:
            engine.general_embeddings_by_domain[domain] = np.zeros((0, 0), dtype=np.float32)

        field_npz_path = embeddings_dir / f"fields_{safe_domain}.npz"
        field_names_path = embeddings_dir / f"fields_{safe_domain}_names.json"
        field_embeddings = {}

        if field_npz_path.exists() and field_names_path.exists():
            field_name_rows = _load_json(field_names_path)

            with np.load(field_npz_path) as data:
                for row in field_name_rows:
                    field_name = row["field_name"]
                    safe_field = row["safe_field"]
                    field_embeddings[field_name] = data[safe_field]

        engine.field_embeddings_by_domain[domain] = field_embeddings

        faiss_path = faiss_dir / f"general_{safe_domain}.index"
        if faiss_path.exists():
            engine.faiss_general_index_by_domain[domain] = faiss.read_index(str(faiss_path))

    # Модель нужна только для embeddings новых запросов.
    engine.embedding_model = _load_embedding_model()

    return engine
