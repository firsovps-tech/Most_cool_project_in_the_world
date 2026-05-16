import csv
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel


KNOWN_CATEGORIES = {
    "Ванная",
    "Гостиная",
    "Спальня",
    "Прихожая",
    "Кухня",
    "Детская",
    "Кабинет",
    "Декор",
    "Текстиль",
    "Балкон",
    "Освещение",
    "Гостиница",
    "Хозтовары",
}


NORMAL_SCORE_WEIGHTS = {
    "article_score": 0.00,
    "text_score": 0.30,
    "general_embedding_score": 0.30,
    "category_embedding_score": 0.20,
    "attribute_score": 0.20,
}


ARTICLE_SCORE_WEIGHTS = {
    "article_score": 0.45,
    "text_score": 0.20,
    "general_embedding_score": 0.15,
    "category_embedding_score": 0.05,
    "attribute_score": 0.15,
}


ATTRIBUTE_WEIGHTS = {
    "category": 0.12,
    "product_type": 0.20,
    "model": 0.10,
    "dimensions": 0.08,
    "color": 0.12,
    "material": 0.14,
    "features": 0.12,
    "style_or_purpose": 0.07,
    "price": 0.05,
}


def handle_new_categories(new_categories):
    return


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = text.replace("ё", "е")
    text = " ".join(text.split())

    if text in {"", "-", "none", "null", "nan"}:
        return ""

    return text


def clean_field(value):
    value = normalize_text(value)

    if value in {"", "-", "none", "null", "nan"}:
        return ""

    return value


def join_non_empty(parts):
    cleaned_parts = []

    for part in parts:
        part = clean_field(part)

        if part:
            cleaned_parts.append(part)

    return " ".join(cleaned_parts)


def tokenize(text):
    return normalize_text(text).split()


def load_products_from_csv(file_path):
    products = []
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            product_id = row["id"]
            article = row.get("Артикул", "")

            category = row["Категория"]
            product_type = row["Тип товара"]
            model = row["Модель"]
            dimensions = row["Габариты"]
            color = row["Цвет"]
            material = row["Материал"]
            features = row["Особенности"]
            style_or_purpose = row["Стиль/Назначение"]
            price = row["Цена (₽)"]

            name = join_non_empty([
                product_type,
                model,
            ])

            exact_search_text = join_non_empty([
                article,
                category,
                product_type,
                model,
                dimensions,
                color,
                material,
                features,
                style_or_purpose,
                price,
            ])

            general_embedding_text = join_non_empty([
                category,
                product_type,
                model,
                dimensions,
                color,
                material,
                features,
                style_or_purpose,
            ])

            category_embedding_text = join_non_empty([
                category,
                product_type,
                style_or_purpose,
            ])

            product = {
                "id": product_id,
                "article": clean_field(article),
                "name": name,
                "category": category,
                "exact_search_text": exact_search_text,
                "general_embedding_text": general_embedding_text,
                "category_embedding_text": category_embedding_text,
                "attributes": {
                    "category": clean_field(category),
                    "product_type": clean_field(product_type),
                    "model": clean_field(model),
                    "dimensions": clean_field(dimensions),
                    "color": clean_field(color),
                    "material": clean_field(material),
                    "features": clean_field(features),
                    "style_or_purpose": clean_field(style_or_purpose),
                    "price": price,
                },
            }

            products.append(product)

    return products


class HybridFurnitureSearch:
    def __init__(self, products):
        self.products = products

        self.new_categories = self._find_new_categories()

        if self.new_categories:
            handle_new_categories(self.new_categories)

        self.embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False
        )

        self.bm25 = None
        self.general_product_embeddings = None
        self.category_product_embeddings = None

        self._build_indexes()

    def _find_new_categories(self):
        found_categories = set()

        for product in self.products:
            category = product.get("category")

            if category:
                found_categories.add(category)

        return found_categories - KNOWN_CATEGORIES

    def _normalize_scores(self, scores):
        scores = np.array(scores, dtype=np.float32)

        min_score = float(np.min(scores))
        max_score = float(np.max(scores))

        if max_score - min_score < 1e-9:
            return np.zeros_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def _build_indexes(self):
        exact_texts = [
            product["exact_search_text"]
            for product in self.products
        ]

        exact_tokens = [
            tokenize(text)
            for text in exact_texts
        ]

        self.bm25 = BM25Okapi(exact_tokens)

        general_texts = [
            product["general_embedding_text"]
            for product in self.products
        ]

        general_output = self.embedding_model.encode(
            general_texts,
            batch_size=8,
            max_length=512
        )

        self.general_product_embeddings = np.array(
            general_output["dense_vecs"],
            dtype=np.float32
        )

        category_texts = [
            product["category_embedding_text"]
            for product in self.products
        ]

        category_output = self.embedding_model.encode(
            category_texts,
            batch_size=8,
            max_length=256
        )

        self.category_product_embeddings = np.array(
            category_output["dense_vecs"],
            dtype=np.float32
        )

    def _article_search_scores(self, query_article):
        query_article = clean_field(query_article)

        scores = np.zeros(len(self.products), dtype=np.float32)

        if not query_article:
            return scores

        for index, product in enumerate(self.products):
            product_article = clean_field(product.get("article", ""))

            if not product_article:
                continue

            if product_article == query_article:
                scores[index] = 1.0
            elif query_article in product_article:
                scores[index] = 0.7
            elif product_article in query_article:
                scores[index] = 0.7

        return scores

    def _exact_match(self, product_value, query_value):
        if query_value is None:
            return None

        product_value = clean_field(product_value)
        query_value = clean_field(query_value)

        if not query_value:
            return None

        if product_value == query_value:
            return 1.0

        return 0.0

    def _features_score(self, product_features, query_features):
        if not query_features:
            return None

        product_features = clean_field(product_features)
        query_features = clean_field(query_features)

        if not query_features:
            return None

        product_parts = set(
            part.strip()
            for part in product_features.split(",")
            if part.strip()
        )

        query_parts = set(
            part.strip()
            for part in query_features.split(",")
            if part.strip()
        )

        if not query_parts:
            return None

        matched = query_parts.intersection(product_parts)

        return len(matched) / len(query_parts)

    def _price_score(self, product_price, query_attributes):
        price_min = query_attributes.get("price_min")
        price_max = query_attributes.get("price_max")

        if price_min is None and price_max is None:
            return None

        if product_price is None:
            return 0.0

        product_price = float(product_price)

        if price_min is not None:
            price_min = float(price_min)

        if price_max is not None:
            price_max = float(price_max)

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

    def _attribute_score(self, product, query_attributes):
        product_attributes = product.get("attributes", {})

        total_score = 0.0
        total_weight = 0.0

        exact_fields = [
            "category",
            "product_type",
            "model",
            "dimensions",
            "color",
            "material",
            "style_or_purpose",
        ]

        for field in exact_fields:
            field_score = self._exact_match(
                product_attributes.get(field),
                query_attributes.get(field)
            )

            if field_score is None:
                continue

            weight = ATTRIBUTE_WEIGHTS[field]
            total_score += weight * field_score
            total_weight += weight

        features_score = self._features_score(
            product_attributes.get("features"),
            query_attributes.get("features")
        )

        if features_score is not None:
            weight = ATTRIBUTE_WEIGHTS["features"]
            total_score += weight * features_score
            total_weight += weight

        price_score = self._price_score(
            product_attributes.get("price"),
            query_attributes
        )

        if price_score is not None:
            weight = ATTRIBUTE_WEIGHTS["price"]
            total_score += weight * price_score
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def _get_allowed_indexes(self, categories):
        allowed_indexes = []

        for index, product in enumerate(self.products):
            product_category = product.get("category")

            if not categories or product_category in categories:
                allowed_indexes.append(index)

        return allowed_indexes

    def _top_indexes_from_scores(self, scores, allowed_indexes_set, limit):
        sorted_indexes = np.argsort(scores)[::-1]
        top_indexes = []

        for index in sorted_indexes:
            index = int(index)

            if index in allowed_indexes_set:
                top_indexes.append(index)

            if len(top_indexes) >= limit:
                break

        return top_indexes

    def search(
        self,
        prepared_query,
        top_k=100,
        article_candidates_count=20,
        bm25_candidates_count=200,
        general_embedding_candidates_count=200,
        category_embedding_candidates_count=200
    ):
        query_article = prepared_query.get("article", "")

        text_for_exact = normalize_text(
            prepared_query.get("text_for_exact", "")
        )

        text_for_embedding = normalize_text(
            prepared_query.get("text_for_embedding", "")
        )

        text_for_category_embedding = normalize_text(
            prepared_query.get("text_for_category_embedding", "")
        )

        categories = prepared_query.get("categories", [])
        query_attributes = prepared_query.get("attributes", {})

        allowed_indexes = self._get_allowed_indexes(categories)

        if not allowed_indexes:
            return []

        allowed_indexes_set = set(allowed_indexes)

        article_scores = self._article_search_scores(query_article)

        exact_tokens = tokenize(text_for_exact)

        bm25_raw_scores = self.bm25.get_scores(exact_tokens)
        text_scores = self._normalize_scores(bm25_raw_scores)

        general_query_output = self.embedding_model.encode(
            [text_for_embedding],
            batch_size=1,
            max_length=512
        )

        general_query_vector = np.array(
            general_query_output["dense_vecs"][0],
            dtype=np.float32
        )

        general_embedding_raw_scores = (
            self.general_product_embeddings @ general_query_vector
        )

        general_embedding_scores = self._normalize_scores(
            general_embedding_raw_scores
        )

        category_query_output = self.embedding_model.encode(
            [text_for_category_embedding],
            batch_size=1,
            max_length=256
        )

        category_query_vector = np.array(
            category_query_output["dense_vecs"][0],
            dtype=np.float32
        )

        category_embedding_raw_scores = (
            self.category_product_embeddings @ category_query_vector
        )

        category_embedding_scores = self._normalize_scores(
            category_embedding_raw_scores
        )

        article_top_indexes = self._top_indexes_from_scores(
            article_scores,
            allowed_indexes_set,
            article_candidates_count
        )

        bm25_top_indexes = self._top_indexes_from_scores(
            text_scores,
            allowed_indexes_set,
            bm25_candidates_count
        )

        general_embedding_top_indexes = self._top_indexes_from_scores(
            general_embedding_scores,
            allowed_indexes_set,
            general_embedding_candidates_count
        )

        category_embedding_top_indexes = self._top_indexes_from_scores(
            category_embedding_scores,
            allowed_indexes_set,
            category_embedding_candidates_count
        )

        candidate_indexes = set()

        for index in article_top_indexes:
            candidate_indexes.add(index)

        for index in bm25_top_indexes:
            candidate_indexes.add(index)

        for index in general_embedding_top_indexes:
            candidate_indexes.add(index)

        for index in category_embedding_top_indexes:
            candidate_indexes.add(index)

        if query_article:
            weights = ARTICLE_SCORE_WEIGHTS
        else:
            weights = NORMAL_SCORE_WEIGHTS

        results = []

        for index in candidate_indexes:
            product = self.products[index]

            article_score = float(article_scores[index])
            text_score = float(text_scores[index])
            general_embedding_score = float(general_embedding_scores[index])
            category_embedding_score = float(category_embedding_scores[index])

            attribute_score = self._attribute_score(
                product,
                query_attributes
            )

            final_score = (
                weights["article_score"] * article_score
                + weights["text_score"] * text_score
                + weights["general_embedding_score"] * general_embedding_score
                + weights["category_embedding_score"] * category_embedding_score
                + weights["attribute_score"] * attribute_score
            )

            results.append({
                "product_id": product["id"],
                "article": product["article"],
                "name": product["name"],
                "category": product["category"],
                "price": product["attributes"].get("price"),
                "article_score": round(article_score, 4),
                "text_score": round(text_score, 4),
                "general_embedding_score": round(general_embedding_score, 4),
                "category_embedding_score": round(category_embedding_score, 4),
                "attribute_score": round(attribute_score, 4),
                "final_score": round(float(final_score), 4),
            })

        results.sort(key=lambda item: item["final_score"], reverse=True)

        return results[:top_k]


if __name__ == "__main__":
    file_path = "Qwen_csv_20260516_iyqyxkuif.txt"

    products = load_products_from_csv(file_path)

    search_engine = HybridFurnitureSearch(products)

    prepared_query = {
        "article": "",
        "text_for_exact": "диван угловой лофт черный механизм дельфин до 50000",
        "text_for_embedding": "диван угловой черный экокожа механизм дельфин стиль лофт",
        "text_for_category_embedding": "гостиная диван угловой лофт",
        "categories": ["Гостиная"],
        "attributes": {
            "category": "Гостиная",
            "product_type": "Диван угловой",
            "color": "чёрный",
            "material": "экокожа",
            "features": "механизм дельфин",
            "style_or_purpose": "лофт",
            "price_max": 50000,
        }
    }

    results = search_engine.search(prepared_query, top_k=100)

    print("Новые категории:", search_engine.new_categories)

    for item in results:
        print()
        print("Товар:", item["name"])
        print("Артикул:", item["article"])
        print("Категория:", item["category"])
        print("Цена:", item["price"])
        print("Article:", item["article_score"])
        print("BM25:", item["text_score"])
        print("General embedding:", item["general_embedding_score"])
        print("Category embedding:", item["category_embedding_score"])
        print("Attributes:", item["attribute_score"])
        print("Final:", item["final_score"])