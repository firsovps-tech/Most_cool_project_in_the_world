from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel


@dataclass
class SearchDomainConfig:
    searchable_fields: list[str]
    field_embedding_weights: dict[str, float]
    final_weights: dict[str, float]
    missing_field_scores: dict[str, float]


DOMAIN_CONFIGS = {
    "furniture": SearchDomainConfig(
        searchable_fields=[
            "description",
            "product_type",
            "model",
            "dimensions",
            "color",
            "material",
            "features",
            "style_or_purpose",
        ],
        field_embedding_weights={
            "description": 0.30,
            "product_type": 0.20,
            "model": 0.10,
            "dimensions": 0.05,
            "color": 0.10,
            "material": 0.10,
            "features": 0.10,
            "style_or_purpose": 0.05,
        },
        final_weights={
            "exact_id_score": 0.35,
            "bm25_score": 0.20,
            "field_embedding_score": 0.35,
            "price_score": 0.10,
        },
        missing_field_scores={
            "description": 0.0,
            "product_type": 0.2,
            "model": 0.5,
            "dimensions": 0.5,
            "color": 0.5,
            "material": 0.4,
            "features": 0.4,
            "style_or_purpose": 0.5,
        },
    ),
}


def handle_new_domain(domain_name, observed_fields):
    return


def is_empty(value):
    return value is None or value == "" or value == "-"


def field_value(value):
    if is_empty(value):
        return ""

    return str(value)


def join_non_empty(parts):
    result = []

    for part in parts:
        part = field_value(part)

        if part:
            result.append(part)

    return " ".join(result)


def tokenize(text):
    text = field_value(text)

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

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    if max_score - min_score < 1e-9:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


def price_to_float(value):
    if is_empty(value):
        return None

    try:
        return float(value)
    except ValueError:
        return None


def calculate_price_score(product_price, price_min=None, price_max=None):
    if price_min is None and price_max is None:
        return None

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

    return None


class NeuralHybridProductSearch:
    def __init__(self, products, domain_name="furniture"):
        self.products = products
        self.domain_name = domain_name

        if domain_name not in DOMAIN_CONFIGS:
            observed_fields = self._get_observed_fields()
            handle_new_domain(domain_name, observed_fields)
            raise ValueError(f"Unknown search domain: {domain_name}")

        self.config = DOMAIN_CONFIGS[domain_name]

        self.embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False,
        )

        self.bm25 = None
        self.field_embeddings = {}

        self._build_indexes()

    def _get_observed_fields(self):
        fields = set()

        for product in self.products:
            if field_value(product.get("description")):
                fields.add("description")

            characteristics = product.get("characteristics", {})

            for field_name, field_val in characteristics.items():
                if field_value(field_val):
                    fields.add(field_name)

        return sorted(fields)

    def _build_indexes(self):
        self._build_bm25_index()
        self._build_field_embedding_indexes()

    def _build_bm25_index(self):
        texts = []

        for product in self.products:
            text = self._build_bm25_text_for_product(product)
            texts.append(text)

        tokens = []

        for text in texts:
            tokens.append(tokenize(text))

        self.bm25 = BM25Okapi(tokens)

    def _build_bm25_text_for_product(self, product):
        return field_value(product.get("description", ""))

    def _build_field_embedding_indexes(self):
        for field_name in self.config.searchable_fields:
            texts = []

            for product in self.products:
                value = self._get_product_field_value(product, field_name)
                texts.append(value)

            vectors = self._encode_texts(texts)
            self.field_embeddings[field_name] = vectors

    def _get_product_field_value(self, product, field_name):
        if field_name == "description":
            return field_value(product.get("description", ""))

        characteristics = product.get("characteristics", {})

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
        vectors = self._encode_texts([text])
        return vectors[0]

    def _exact_id_scores(self, query_id):
        query_id = field_value(query_id)

        scores = np.zeros(len(self.products), dtype=np.float32)

        if not query_id:
            return scores

        for index, product in enumerate(self.products):
            product_id = field_value(product.get("id"))

            if product_id == query_id:
                scores[index] = 1.0
            else:
                scores[index] = 0.0

        return scores

    def _bm25_scores(self, query_text):
        query_text = field_value(query_text)

        if not query_text:
            return np.zeros(len(self.products), dtype=np.float32)

        query_tokens = tokenize(query_text)
        raw_scores = self.bm25.get_scores(query_tokens)

        return normalize_scores(raw_scores)

    def _build_bm25_text_for_query(self, prepared_query):
        return field_value(prepared_query.get("description", ""))

    def _field_embedding_scores(self, prepared_query):
        result = {}

        query_description = field_value(
            prepared_query.get("description", "")
        )

        query_characteristics = prepared_query.get("characteristics", {})

        for field_name in self.config.searchable_fields:
            if field_name == "description":
                query_value = query_description
            else:
                query_value = field_value(
                    query_characteristics.get(field_name, "")
                )

            if not query_value:
                continue

            if field_name not in self.field_embeddings:
                continue

            query_vector = self._encode_one_text(query_value)

            product_vectors = self.field_embeddings[field_name]
            raw_scores = product_vectors @ query_vector

            scores = normalize_scores(raw_scores)

            missing_score = self.config.missing_field_scores.get(
                field_name,
                0.5,
            )

            for index, product in enumerate(self.products):
                product_value = self._get_product_field_value(
                    product,
                    field_name,
                )

                if not product_value:
                    scores[index] = missing_score

            result[field_name] = scores

        return result

    def _combine_field_embedding_scores(self, field_scores):
        final_scores = np.zeros(len(self.products), dtype=np.float32)
        used_weight_sum = 0.0

        for field_name, scores in field_scores.items():
            weight = self.config.field_embedding_weights.get(field_name, 0.0)

            if weight <= 0:
                continue

            final_scores += weight * scores
            used_weight_sum += weight

        if used_weight_sum > 0:
            final_scores = final_scores / used_weight_sum

        return final_scores

    def _price_scores(self, prepared_query):
        price_min = prepared_query.get("price_min")
        price_max = prepared_query.get("price_max")

        scores = np.zeros(len(self.products), dtype=np.float32)

        if price_min is None and price_max is None:
            return scores

        for index, product in enumerate(self.products):
            score = calculate_price_score(
                product_price=product.get("price"),
                price_min=price_min,
                price_max=price_max,
            )

            if score is None:
                score = 0.0

            scores[index] = score

        return scores

    def _calculate_final_scores(
        self,
        exact_id_scores,
        bm25_scores,
        field_embedding_scores,
        price_scores,
        has_query_id,
        has_price_filter,
    ):
        weights = dict(self.config.final_weights)

        if not has_query_id:
            weights["exact_id_score"] = 0.0

        if not has_price_filter:
            weights["price_score"] = 0.0

        used_weight_sum = sum(weights.values())

        if used_weight_sum <= 0:
            return np.zeros(len(self.products), dtype=np.float32)

        final_scores = (
            weights["exact_id_score"] * exact_id_scores
            + weights["bm25_score"] * bm25_scores
            + weights["field_embedding_score"] * field_embedding_scores
            + weights["price_score"] * price_scores
        )

        final_scores = final_scores / used_weight_sum

        return final_scores

    def search(self, prepared_query, top_k=100):
        query_id = prepared_query.get("id", "")

        exact_id_scores = self._exact_id_scores(query_id)

        bm25_query_text = self._build_bm25_text_for_query(prepared_query)
        bm25_scores = self._bm25_scores(bm25_query_text)

        field_scores = self._field_embedding_scores(prepared_query)

        field_embedding_scores = self._combine_field_embedding_scores(
            field_scores
        )

        price_scores = self._price_scores(prepared_query)

        final_scores = self._calculate_final_scores(
            exact_id_scores=exact_id_scores,
            bm25_scores=bm25_scores,
            field_embedding_scores=field_embedding_scores,
            price_scores=price_scores,
            has_query_id=bool(field_value(query_id)),
            has_price_filter=(
                prepared_query.get("price_min") is not None
                or prepared_query.get("price_max") is not None
            ),
        )

        results = []

        for index, product in enumerate(self.products):
            item = {
                "product_id": product["id"],
                "description": product.get("description", ""),
                "characteristics": product.get("characteristics", {}),
                "price": product.get("price"),
                "exact_id_score": round(float(exact_id_scores[index]), 4),
                "bm25_score": round(float(bm25_scores[index]), 4),
                "field_embedding_score": round(float(field_embedding_scores[index]), 4),
                "price_score": round(float(price_scores[index]), 4),
                "final_score": round(float(final_scores[index]), 4),
            }

            for field_name, scores in field_scores.items():
                item[f"{field_name}_embedding_score"] = round(
                    float(scores[index]),
                    4,
                )

            results.append(item)

        results.sort(
            key=lambda item: item["final_score"],
            reverse=True,
        )

        return results[:top_k]


if __name__ == "__main__":
    products = [
        {
            "id": "1001",
            "description": "черный угловой диван из экокожи механизм дельфин стиль лофт",
            "characteristics": {
                "product_type": "диван угловой",
                "model": "лофт",
                "dimensions": "-",
                "color": "черный",
                "material": "экокожа",
                "features": "механизм дельфин",
                "style_or_purpose": "лофт",
            },
            "price": 44900,
        },
        {
            "id": "1002",
            "description": "серый прямой диван рогожка раскладной книжка скандинавский стиль",
            "characteristics": {
                "product_type": "диван прямой",
                "model": "стокгольм",
                "dimensions": "180x85",
                "color": "серый",
                "material": "рогожка",
                "features": "раскладной книжка",
                "style_or_purpose": "скандинавский",
            },
            "price": 28900,
        },
        {
            "id": "1003",
            "description": "обеденный стол орех шпон ореха раздвижной",
            "characteristics": {
                "product_type": "стол обеденный",
                "model": "турин",
                "dimensions": "160x90",
                "color": "орех",
                "material": "шпон ореха",
                "features": "раздвижной",
                "style_or_purpose": "-",
            },
            "price": 38700,
        },
    ]

    search_engine = NeuralHybridProductSearch(
        products=products,
        domain_name="furniture",
    )

    prepared_query = {
        "id": "",
        "description": "черный угловой диван из экокожи в стиле лофт",
        "characteristics": {
            "product_type": "диван угловой",
            "model": "-",
            "dimensions": "-",
            "color": "черный",
            "material": "экокожа",
            "features": "механизм дельфин",
            "style_or_purpose": "лофт",
        },
        "price_max": 50000,
    }

    results = search_engine.search(
        prepared_query,
        top_k=100,
    )

    for item in results:
        print()
        print("ID:", item["product_id"])
        print("Описание:", item["description"])
        print("Характеристики:", item["characteristics"])
        print("Цена:", item["price"])
        print("Exact ID:", item["exact_id_score"])
        print("BM25:", item["bm25_score"])
        print("Field embedding:", item["field_embedding_score"])
        print("Price:", item["price_score"])
        print("Final:", item["final_score"])