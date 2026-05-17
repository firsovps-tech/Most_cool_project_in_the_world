from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel


@dataclass
class SearchDomainConfig:
    searchable_fields: list[str]
    field_embedding_weights: dict[str, float]
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


class NeuralHybridProductSearch:
    """
    Этот класс НЕ задаёт ручные финальные веса.

    Он делает 2 вещи:
    1. Считает признаки для XGBoost.
    2. Делает двухэтапный отбор кандидатов:
       сначала top-1000 по общему description embedding,
       потом top-100 по подробным признакам.

    Финальную сортировку делает отдельный файл с бустингом.
    """

    def __init__(self, products, domain_name="furniture"):
        self.products = products
        self.domain_name = domain_name

        if domain_name not in DOMAIN_CONFIGS:
            observed_fields = self._get_observed_fields()
            handle_new_domain(domain_name, observed_fields)
            raise ValueError(f"Unknown search domain: {domain_name}")

        self.config = DOMAIN_CONFIGS[domain_name]
        self.feature_names = self._build_feature_names()

        self.embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False,
        )

        self.bm25 = None
        self.field_embeddings = {}

        self._build_indexes()

    def _build_feature_names(self):
        names = [
            "exact_id_score",
            "bm25_score",
            "field_embedding_score",
            "price_score",
            "has_query_id",
            "has_price_filter",
        ]

        for field_name in self.config.searchable_fields:
            names.append(f"{field_name}_embedding_score")

        for field_name in self.config.searchable_fields:
            names.append(f"product_has_{field_name}")

        return names

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
        tokens = []

        for product in self.products:
            text = self._build_bm25_text_for_product(product)
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

            self.field_embeddings[field_name] = self._encode_texts(texts)

    def _get_product_field_value(self, product, field_name):
        if field_name == "description":
            return field_value(product.get("description", ""))

        characteristics = product.get("characteristics", {})
        return field_value(characteristics.get(field_name, ""))

    def _get_query_field_value(self, prepared_query, field_name):
        if field_name == "description":
            return field_value(prepared_query.get("description", ""))

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

    def _exact_id_scores(self, query_id):
        query_id = field_value(query_id)
        scores = np.zeros(len(self.products), dtype=np.float32)

        if not query_id:
            return scores

        for index, product in enumerate(self.products):
            product_id = field_value(product.get("id"))

            if product_id == query_id:
                scores[index] = 1.0

        return scores

    def _bm25_scores(self, query_text):
        query_text = field_value(query_text)

        if not query_text:
            return np.zeros(len(self.products), dtype=np.float32)

        query_tokens = tokenize(query_text)
        raw_scores = self.bm25.get_scores(query_tokens)

        return normalize_scores(raw_scores)

    def _field_embedding_scores(self, prepared_query):
        result = {}

        for field_name in self.config.searchable_fields:
            query_value = self._get_query_field_value(prepared_query, field_name)

            if not query_value:
                continue

            if field_name not in self.field_embeddings:
                continue

            query_vector = self._encode_one_text(query_value)
            product_vectors = self.field_embeddings[field_name]

            raw_scores = product_vectors @ query_vector
            scores = normalize_scores(raw_scores)

            missing_score = self.config.missing_field_scores.get(field_name, 0.5)

            for index, product in enumerate(self.products):
                product_value = self._get_product_field_value(product, field_name)

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
            scores[index] = calculate_price_score(
                product_price=product.get("price"),
                price_min=price_min,
                price_max=price_max,
            )

        return scores

    def all_scores_for_query(self, prepared_query):
        query_id = prepared_query.get("id", "")

        exact_id_scores = self._exact_id_scores(query_id)
        bm25_scores = self._bm25_scores(prepared_query.get("description", ""))

        field_scores = self._field_embedding_scores(prepared_query)
        field_embedding_scores = self._combine_field_embedding_scores(field_scores)

        price_scores = self._price_scores(prepared_query)

        return {
            "exact_id_score": exact_id_scores,
            "bm25_score": bm25_scores,
            "field_embedding_score": field_embedding_scores,
            "price_score": price_scores,
            "field_scores": field_scores,
        }

    def build_feature_vector(self, prepared_query, product_index, all_scores):
        product = self.products[product_index]

        has_query_id = 1.0 if field_value(prepared_query.get("id", "")) else 0.0

        has_price_filter = (
            prepared_query.get("price_min") is not None
            or prepared_query.get("price_max") is not None
        )

        feature_values = {
            "exact_id_score": float(all_scores["exact_id_score"][product_index]),
            "bm25_score": float(all_scores["bm25_score"][product_index]),
            "field_embedding_score": float(all_scores["field_embedding_score"][product_index]),
            "price_score": float(all_scores["price_score"][product_index]),
            "has_query_id": float(has_query_id),
            "has_price_filter": float(has_price_filter),
        }

        field_scores = all_scores["field_scores"]

        for field_name in self.config.searchable_fields:
            key = f"{field_name}_embedding_score"

            if field_name in field_scores:
                feature_values[key] = float(field_scores[field_name][product_index])
            else:
                feature_values[key] = 0.0

        for field_name in self.config.searchable_fields:
            key = f"product_has_{field_name}"
            product_value = self._get_product_field_value(product, field_name)

            feature_values[key] = 1.0 if product_value else 0.0

        return [
            feature_values.get(feature_name, 0.0)
            for feature_name in self.feature_names
        ]

    def _top_indexes(self, scores, limit):
        limit = min(limit, len(scores))
        return [int(index) for index in np.argsort(scores)[::-1][:limit]]

    def _description_candidate_scores(self, prepared_query):
        """
        1-й этап: общий быстрый отбор по description embedding.

        Сейчас это считается через numpy.
        Потом именно этот метод можно заменить на Qdrant.
        """
        query_text = field_value(prepared_query.get("description", ""))

        if not query_text:
            return np.zeros(len(self.products), dtype=np.float32)

        query_vector = self._encode_one_text(query_text)
        product_vectors = self.field_embeddings["description"]

        raw_scores = product_vectors @ query_vector
        return normalize_scores(raw_scores)

    def _detail_candidate_scores(self, all_scores):
        """
        2-й этап: отбор top-100 среди top-1000.

        Это не финальное ранжирование.
        Это только фильтр перед XGBoost.
        """
        exact_id_scores = all_scores["exact_id_score"]
        bm25_scores = all_scores["bm25_score"]
        field_embedding_scores = all_scores["field_embedding_score"]
        price_scores = all_scores["price_score"]

        return (
            0.10 * exact_id_scores
            + 0.20 * bm25_scores
            + 0.65 * field_embedding_scores
            + 0.05 * price_scores
        )

    def get_candidate_indexes(
        self,
        prepared_query,
        relevant_ids=None,
        first_stage_limit=1000,
        final_candidate_limit=100,
        id_limit=10,
    ):
        """
        Новая схема:

        1. Берём top-1000 по общему description embedding.
        2. Внутри них отбираем top-100 по подробному score.
        3. На обучении обязательно добавляем relevant_ids,
           чтобы правильные ответы не потерялись.
        4. Дальше эти кандидаты идут в XGBoost.
        """
        if relevant_ids is None:
            relevant_ids = []

        all_scores = self.all_scores_for_query(prepared_query)

        description_scores = self._description_candidate_scores(prepared_query)
        first_stage_indexes = set(
            self._top_indexes(description_scores, first_stage_limit)
        )

        exact_id_scores = all_scores["exact_id_score"]

        for index in self._top_indexes(exact_id_scores, id_limit):
            if exact_id_scores[index] > 0:
                first_stage_indexes.add(index)

        detail_scores = self._detail_candidate_scores(all_scores)

        first_stage_indexes = list(first_stage_indexes)
        first_stage_indexes.sort(
            key=lambda index: float(detail_scores[index]),
            reverse=True,
        )

        candidate_indexes = set(first_stage_indexes[:final_candidate_limit])

        relevant_ids = set(field_value(item) for item in relevant_ids)

        for index, product in enumerate(self.products):
            product_id = field_value(product.get("id"))

            if product_id in relevant_ids:
                candidate_indexes.add(index)

        return sorted(candidate_indexes), all_scores
