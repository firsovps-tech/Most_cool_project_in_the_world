import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel


@dataclass
class SearchDomainConfig:
    searchable_fields: list[str]
    missing_field_scores: dict[str, float]


TECHNICAL_COLUMN_NAMES = {
    "domain",
    "Domain",
    "домен",
    "Домен",
    "category",
    "Category",
    "категория",
    "Категория",

    "id",
    "ID",
    "Id",
    "product_id",
    "Product ID",
    "productId",

    "article",
    "Article",
    "артикул",
    "Артикул",
    "ARTICLE",

    "price",
    "Price",
    "цена",
    "Цена",
    "Цена (₽)",
    "Цена ₽",
}

ID_COLUMN_NAMES = [
    "id",
    "ID",
    "Id",
    "product_id",
    "Product ID",
    "productId",
]

ARTICLE_COLUMN_NAMES = [
    "article",
    "Article",
    "артикул",
    "Артикул",
    "ARTICLE",
]

PRICE_COLUMN_NAMES = [
    "price",
    "Price",
    "цена",
    "Цена",
    "Цена (₽)",
    "Цена ₽",
]


def is_empty(value):
    return value is None or str(value).strip() == "" or str(value).strip() == "-"


def field_value(value):
    if is_empty(value):
        return ""

    return str(value).strip()


def tokenize(text):
    text = field_value(text).lower().replace("ё", "е")

    if not text:
        return []

    return text.split()


def normalize_vector(vector):
    vector = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def normalize_scores(scores):
    scores = np.array(scores, dtype=np.float32)

    if len(scores) == 0:
        return scores

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if max_score - min_score < 1e-9:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


def price_to_float(value):
    if is_empty(value):
        return None

    text = str(value).replace(" ", "").replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def calculate_price_score(product_price, price_min=None, price_max=None):
    if price_min is None and price_max is None:
        return 0.0

    product_price = price_to_float(product_price)

    if product_price is None:
        return 0.0

    price_min = price_to_float(price_min)
    price_max = price_to_float(price_max)

    if price_min is not None and price_max is not None:
        if price_min <= product_price <= price_max:
            return 1.0

        if product_price < price_min:
            diff = price_min - product_price
            return max(0.0, 1.0 - diff / price_min)

        diff = product_price - price_max
        return max(0.0, 1.0 - diff / price_max)

    if price_max is not None:
        if product_price <= price_max:
            return 1.0

        diff = product_price - price_max
        return max(0.0, 1.0 - diff / price_max)

    if price_min is not None:
        if product_price >= price_min:
            return 1.0

        diff = price_min - product_price
        return max(0.0, 1.0 - diff / price_min)

    return 0.0


def load_json(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path = Path(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def first_existing_value(row, column_names):
    for column_name in column_names:
        if column_name in row and field_value(row.get(column_name)):
            return field_value(row.get(column_name))

    return ""


def build_description_from_characteristics(characteristics):
    parts = []

    for value in characteristics.values():
        value = field_value(value)

        if value:
            parts.append(value)

    return " ".join(parts)


def get_product_domain(product):
    return (
        field_value(product.get("domain"))
        or field_value(product.get("category"))
        or "default"
    )


def get_query_domain(prepared_query):
    return (
        field_value(prepared_query.get("domain"))
        or field_value(prepared_query.get("category"))
        or ""
    )


def load_products_from_csv(path):
    """
    CSV формат:
    1. первая строка — заголовки;
    2. первый столбец — домен;
    3. id / article / price — технические поля;
    4. все остальные колонки — характеристики;
    5. description в файле не нужен: он строится из характеристик.
    """

    path = Path(path)
    products = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        columns = reader.fieldnames or []

        if not columns:
            return products

        domain_column = columns[0]

        for row_number, row in enumerate(reader, start=1):
            domain = field_value(row.get(domain_column)) or "default"

            product_id = first_existing_value(row, ID_COLUMN_NAMES)

            if not product_id:
                product_id = f"{domain}_{path.stem}_{row_number}"

            article = first_existing_value(row, ARTICLE_COLUMN_NAMES)
            price = first_existing_value(row, PRICE_COLUMN_NAMES)

            characteristics = {}

            for column in columns:
                if column == domain_column:
                    continue

                if column in TECHNICAL_COLUMN_NAMES:
                    continue

                value = field_value(row.get(column))

                if value:
                    characteristics[column] = value
                else:
                    characteristics[column] = "-"

            description = build_description_from_characteristics(characteristics)

            products.append({
                "id": product_id,
                "article": article,
                "domain": domain,
                "description": description,
                "characteristics": characteristics,
                "price": price,
            })

    return products


def normalize_product_from_json(raw_product, fallback_domain="default", fallback_id=""):
    domain = (
        field_value(raw_product.get("domain"))
        or field_value(raw_product.get("category"))
        or fallback_domain
    )

    product_id = (
        field_value(raw_product.get("id"))
        or field_value(raw_product.get("ID"))
        or field_value(raw_product.get("product_id"))
        or fallback_id
    )

    article = (
        field_value(raw_product.get("article"))
        or field_value(raw_product.get("Article"))
        or field_value(raw_product.get("Артикул"))
    )

    price = (
        raw_product.get("price")
        or raw_product.get("Price")
        or raw_product.get("Цена")
        or raw_product.get("Цена (₽)")
        or 0
    )

    raw_characteristics = raw_product.get("characteristics", {})
    characteristics = {}

    if isinstance(raw_characteristics, dict) and raw_characteristics:
        for field_name, field_val in raw_characteristics.items():
            if field_name in TECHNICAL_COLUMN_NAMES:
                continue

            value = field_value(field_val)

            if value:
                characteristics[field_name] = value
            else:
                characteristics[field_name] = "-"
    else:
        for key, value in raw_product.items():
            if key in TECHNICAL_COLUMN_NAMES:
                continue

            if key in {"description", "Описание", "characteristics"}:
                continue

            value = field_value(value)

            if value:
                characteristics[key] = value
            else:
                characteristics[key] = "-"

    description = build_description_from_characteristics(characteristics)

    return {
        "id": product_id,
        "article": article,
        "domain": domain,
        "description": description,
        "characteristics": characteristics,
        "price": price,
    }


def load_products_from_json(path):
    path = Path(path)
    raw_data = load_json(path)

    if isinstance(raw_data, dict):
        raw_products = raw_data.get("products", [])
    else:
        raw_products = raw_data

    products = []

    for index, raw_product in enumerate(raw_products, start=1):
        fallback_id = f"{path.stem}_{index}"

        product = normalize_product_from_json(
            raw_product=raw_product,
            fallback_domain="default",
            fallback_id=fallback_id,
        )

        products.append(product)

    return products


def load_products_from_file(path):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return load_products_from_csv(path)

    if path.suffix.lower() == ".json":
        return load_products_from_json(path)

    raise ValueError(f"Unsupported product file format: {path}")


def load_products_from_files(paths):
    products = []

    for path in paths:
        products.extend(load_products_from_file(path))

    return products


def infer_domain_configs(products):
    fields_by_domain = {}

    for product in products:
        domain = get_product_domain(product)

        if domain not in fields_by_domain:
            fields_by_domain[domain] = []

        seen = set(fields_by_domain[domain])
        characteristics = product.get("characteristics", {})

        for field_name, field_val in characteristics.items():
            if field_name in TECHNICAL_COLUMN_NAMES:
                continue

            if not field_value(field_val):
                continue

            if field_name not in seen:
                fields_by_domain[domain].append(field_name)
                seen.add(field_name)

    configs = {}

    for domain, fields in fields_by_domain.items():
        missing_field_scores = {}

        for field_name in fields:
            missing_field_scores[field_name] = 0.5

        configs[domain] = SearchDomainConfig(
            searchable_fields=fields,
            missing_field_scores=missing_field_scores,
        )

    return configs


class NeuralHybridProductSearch:
    """
    Главный класс для построения поисковых признаков.

    Что важно:
    - ручных field_embedding_weights тут нет;
    - признаки читаются из файлов;
    - каждый домен индексируется отдельно;
    - description строится автоматически из характеристик;
    - поиск идёт отдельно по каждому домену.
    """

    def __init__(self, products, domain_configs=None):
        self.products = products

        if domain_configs is None:
            domain_configs = infer_domain_configs(products)

        self.domain_configs = domain_configs
        self.feature_names_by_domain = {}

        for domain, config in self.domain_configs.items():
            self.feature_names_by_domain[domain] = self._build_feature_names(config)

        self.embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False,
        )

        self.domain_to_product_indexes = {}
        self.bm25_by_domain = {}
        self.field_embeddings_by_domain = {}

        self._build_indexes()

    def _build_feature_names(self, config):
        names = [
            "exact_id_score",
            "bm25_score",
            "price_score",
            "has_query_id",
            "has_price_filter",
        ]

        for field_name in config.searchable_fields:
            names.append(f"{field_name}_embedding_score")

        for field_name in config.searchable_fields:
            names.append(f"product_has_{field_name}")

        return names

    def get_domain_config(self, domain):
        if domain not in self.domain_configs:
            raise ValueError(f"Unknown domain: {domain}")

        return self.domain_configs[domain]

    def get_feature_names(self, domain):
        if domain not in self.feature_names_by_domain:
            raise ValueError(f"Unknown domain: {domain}")

        return self.feature_names_by_domain[domain]

    def _build_indexes(self):
        for index, product in enumerate(self.products):
            domain = get_product_domain(product)

            if domain not in self.domain_to_product_indexes:
                self.domain_to_product_indexes[domain] = []

            self.domain_to_product_indexes[domain].append(index)

        for domain in self.domain_configs:
            self._build_bm25_index_for_domain(domain)
            self._build_field_embedding_indexes_for_domain(domain)

    def _build_bm25_index_for_domain(self, domain):
        product_indexes = self.domain_to_product_indexes.get(domain, [])
        tokens = []

        for product_index in product_indexes:
            product = self.products[product_index]
            text = field_value(product.get("description", ""))
            tokens.append(tokenize(text))

        self.bm25_by_domain[domain] = BM25Okapi(tokens)

    def _build_field_embedding_indexes_for_domain(self, domain):
        config = self.get_domain_config(domain)
        product_indexes = self.domain_to_product_indexes.get(domain, [])

        field_embeddings = {}

        for field_name in config.searchable_fields:
            texts = []

            for product_index in product_indexes:
                product = self.products[product_index]
                value = self._get_product_field_value(product, field_name)
                texts.append(value)

            field_embeddings[field_name] = self._encode_texts(texts)

        self.field_embeddings_by_domain[domain] = field_embeddings

    def _get_product_field_value(self, product, field_name):
        characteristics = product.get("characteristics", {})
        return field_value(characteristics.get(field_name, ""))

    def _get_query_field_value(self, prepared_query, field_name):
        characteristics = prepared_query.get("characteristics", {})
        return field_value(characteristics.get(field_name, ""))

    def _encode_texts(self, texts):
        prepared_texts = []

        for text in texts:
            text = field_value(text)

            if not text:
                text = " "

            prepared_texts.append(text)

        output = self.embedding_model.encode(
            prepared_texts,
            batch_size=8,
            max_length=256,
        )

        vectors = output["dense_vecs"]
        normalized_vectors = []

        for vector in vectors:
            normalized_vectors.append(normalize_vector(vector))

        return np.array(normalized_vectors, dtype=np.float32)

    def _encode_one_text(self, text):
        return self._encode_texts([text])[0]

    def _query_text_from_characteristics(self, prepared_query, domain):
        config = self.get_domain_config(domain)
        parts = []

        for field_name in config.searchable_fields:
            value = self._get_query_field_value(prepared_query, field_name)

            if value:
                parts.append(value)

        return " ".join(parts)

    def _exact_id_scores_for_domain(self, query_id, domain):
        product_indexes = self.domain_to_product_indexes.get(domain, [])
        scores = np.zeros(len(product_indexes), dtype=np.float32)

        query_id = field_value(query_id)

        if not query_id:
            return scores

        for local_index, product_index in enumerate(product_indexes):
            product = self.products[product_index]
            product_id = field_value(product.get("id"))

            if product_id == query_id:
                scores[local_index] = 1.0

        return scores

    def _bm25_scores_for_domain(self, query_text, domain):
        product_indexes = self.domain_to_product_indexes.get(domain, [])
        query_text = field_value(query_text)

        if not query_text:
            return np.zeros(len(product_indexes), dtype=np.float32)

        query_tokens = tokenize(query_text)
        raw_scores = self.bm25_by_domain[domain].get_scores(query_tokens)

        return normalize_scores(raw_scores)

    def _field_embedding_scores_for_domain(self, prepared_query, domain):
        config = self.get_domain_config(domain)
        product_indexes = self.domain_to_product_indexes.get(domain, [])
        field_embeddings = self.field_embeddings_by_domain[domain]

        result = {}

        for field_name in config.searchable_fields:
            query_value = self._get_query_field_value(prepared_query, field_name)

            if not query_value:
                continue

            if field_name not in field_embeddings:
                continue

            query_vector = self._encode_one_text(query_value)
            product_vectors = field_embeddings[field_name]

            raw_scores = product_vectors @ query_vector
            scores = normalize_scores(raw_scores)

            missing_score = config.missing_field_scores.get(field_name, 0.5)

            for local_index, product_index in enumerate(product_indexes):
                product = self.products[product_index]
                product_value = self._get_product_field_value(product, field_name)

                if not product_value:
                    scores[local_index] = missing_score

            result[field_name] = scores

        return result

    def _price_scores_for_domain(self, prepared_query, domain):
        product_indexes = self.domain_to_product_indexes.get(domain, [])

        price_min = prepared_query.get("price_min")
        price_max = prepared_query.get("price_max")

        scores = np.zeros(len(product_indexes), dtype=np.float32)

        if price_min is None and price_max is None:
            return scores

        for local_index, product_index in enumerate(product_indexes):
            product = self.products[product_index]

            scores[local_index] = calculate_price_score(
                product_price=product.get("price"),
                price_min=price_min,
                price_max=price_max,
            )

        return scores

    def all_scores_for_query_domain(self, prepared_query, domain):
        product_indexes = self.domain_to_product_indexes.get(domain, [])

        query_text = self._query_text_from_characteristics(
            prepared_query=prepared_query,
            domain=domain,
        )

        exact_id_scores = self._exact_id_scores_for_domain(
            prepared_query.get("id", ""),
            domain,
        )

        bm25_scores = self._bm25_scores_for_domain(
            query_text=query_text,
            domain=domain,
        )

        field_scores = self._field_embedding_scores_for_domain(
            prepared_query=prepared_query,
            domain=domain,
        )

        price_scores = self._price_scores_for_domain(
            prepared_query=prepared_query,
            domain=domain,
        )

        return {
            "domain": domain,
            "product_indexes": product_indexes,
            "exact_id_score": exact_id_scores,
            "bm25_score": bm25_scores,
            "price_score": price_scores,
            "field_scores": field_scores,
        }

    def _top_local_indexes(self, scores, limit):
        limit = min(limit, len(scores))
        return [int(index) for index in np.argsort(scores)[::-1][:limit]]

    def _description_candidate_scores_for_domain(self, prepared_query, domain):
        product_indexes = self.domain_to_product_indexes.get(domain, [])

        query_text = self._query_text_from_characteristics(
            prepared_query=prepared_query,
            domain=domain,
        )

        if not query_text:
            return np.zeros(len(product_indexes), dtype=np.float32)

        query_vector = self._encode_one_text(query_text)

        config = self.get_domain_config(domain)
        field_embeddings = self.field_embeddings_by_domain[domain]

        score_parts = []

        for field_name in config.searchable_fields:
            if field_name not in field_embeddings:
                continue

            product_vectors = field_embeddings[field_name]
            raw_scores = product_vectors @ query_vector
            score_parts.append(normalize_scores(raw_scores))

        if not score_parts:
            return np.zeros(len(product_indexes), dtype=np.float32)

        stacked = np.vstack(score_parts)

        return np.max(stacked, axis=0)

    def get_candidate_indexes_for_domain(
        self,
        prepared_query,
        domain,
        relevant_ids=None,
        first_stage_limit=1000,
        id_limit=10,
    ):
        if relevant_ids is None:
            relevant_ids = []

        product_indexes = self.domain_to_product_indexes.get(domain, [])
        all_scores = self.all_scores_for_query_domain(prepared_query, domain)

        description_scores = self._description_candidate_scores_for_domain(
            prepared_query=prepared_query,
            domain=domain,
        )

        candidate_local_indexes = set(
            self._top_local_indexes(description_scores, first_stage_limit)
        )

        exact_id_scores = all_scores["exact_id_score"]

        for local_index in self._top_local_indexes(exact_id_scores, id_limit):
            if exact_id_scores[local_index] > 0:
                candidate_local_indexes.add(local_index)

        relevant_ids = set(field_value(item) for item in relevant_ids)

        for local_index, product_index in enumerate(product_indexes):
            product = self.products[product_index]
            product_id = field_value(product.get("id"))

            if product_id in relevant_ids:
                candidate_local_indexes.add(local_index)

        candidate_product_indexes = [
            product_indexes[local_index]
            for local_index in sorted(candidate_local_indexes)
        ]

        return candidate_product_indexes, all_scores

    def build_feature_vector(self, prepared_query, product_index, all_scores):
        domain = all_scores["domain"]
        product_indexes = all_scores["product_indexes"]
        feature_names = self.get_feature_names(domain)
        config = self.get_domain_config(domain)

        local_index_by_product_index = {
            current_product_index: local_index
            for local_index, current_product_index in enumerate(product_indexes)
        }

        local_index = local_index_by_product_index[product_index]
        product = self.products[product_index]

        has_query_id = 1.0 if field_value(prepared_query.get("id", "")) else 0.0

        has_price_filter = (
            prepared_query.get("price_min") is not None
            or prepared_query.get("price_max") is not None
        )

        feature_values = {
            "exact_id_score": float(all_scores["exact_id_score"][local_index]),
            "bm25_score": float(all_scores["bm25_score"][local_index]),
            "price_score": float(all_scores["price_score"][local_index]),
            "has_query_id": float(has_query_id),
            "has_price_filter": float(has_price_filter),
        }

        field_scores = all_scores["field_scores"]

        for field_name in config.searchable_fields:
            key = f"{field_name}_embedding_score"

            if field_name in field_scores:
                feature_values[key] = float(field_scores[field_name][local_index])
            else:
                feature_values[key] = 0.0

        for field_name in config.searchable_fields:
            key = f"product_has_{field_name}"
            product_value = self._get_product_field_value(product, field_name)

            feature_values[key] = 1.0 if product_value else 0.0

        return [
            feature_values.get(feature_name, 0.0)
            for feature_name in feature_names
        ]
