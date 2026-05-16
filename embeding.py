import numpy as np

from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel


class HybridProductSearch:
    """
    Гибридный поиск товаров.

    Используем:
    1. BM25 — поиск по точным словам.
    2. BAAI/bge-m3 — сильная embedding-модель для поиска по смыслу.
    3. Attribute score — совпадение точных характеристик товара.
    """

    def __init__(self, products):
        """
        products — список уже подготовленных товаров.

        Важно:
        очистка текста, подготовка категорий и извлечение признаков
        происходят отдельно.

        Здесь поиск получает уже готовые:
        - search_text
        - category
        - attributes
        """

        self.products = products

        # Сильная multilingual embedding-модель.
        # На CPU лучше use_fp16=False.
        # Если есть GPU, можно попробовать use_fp16=True.
        self.embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False
        )

        # Тексты товаров для поиска.
        self.product_texts = []

        # Токены товаров для BM25.
        self.product_tokens = []

        # Embedding-векторы товаров.
        self.product_embeddings = None

        # BM25-индекс.
        self.bm25 = None

        # Строим индексы.
        self._build_index()

    def _tokenize(self, text):
        """
        Разбивает подготовленный текст на слова.

        Мы считаем, что search_text уже очищен заранее.
        """
        return text.split()

    def _normalize_scores(self, scores):
        """
        Приводит scores к диапазону 0...1.

        Это нужно, потому что BM25 и embeddings
        дают оценки в разных масштабах.
        """

        scores = np.array(scores, dtype=np.float32)

        min_score = float(np.min(scores))
        max_score = float(np.max(scores))

        if max_score - min_score < 1e-9:
            return np.zeros_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def _build_index(self):
        """
        Строит:
        1. BM25 индекс.
        2. Embedding-векторы товаров.
        """

        # Берём готовые search_text у товаров.
        self.product_texts = [
            product["search_text"]
            for product in self.products
        ]

        # Разбиваем search_text на слова для BM25.
        self.product_tokens = [
            self._tokenize(text)
            for text in self.product_texts
        ]

        # Создаём BM25 индекс.
        self.bm25 = BM25Okapi(self.product_tokens)

        # Считаем dense embeddings товаров через BGE-M3.
        output = self.embedding_model.encode(
            self.product_texts,
            batch_size=8,
            max_length=512
        )

        # Нам нужны именно dense-векторы.
        dense_vectors = output["dense_vecs"]

        self.product_embeddings = np.array(
            dense_vectors,
            dtype=np.float32
        )

    def _exact_attribute_score(self, product_attributes, query_attributes, attr_name):
        """
        Проверяет точное совпадение одного атрибута.

        Например:
        query color = "бежевый"
        product color = "бежевый"

        Тогда score = 1.

        Если не совпало — 0.
        """

        query_value = query_attributes.get(attr_name)

        if query_value is None:
            return None

        product_value = product_attributes.get(attr_name)

        if product_value == query_value:
            return 1.0

        return 0.0

    def _list_attribute_score(self, product_attributes, query_attributes, attr_name):
        """
        Проверяет атрибуты, где может быть список значений.

        Например особенности:

        query:
        ["раскладной", "с ящиком"]

        product:
        ["раскладной", "мягкий", "с ящиком"]

        Совпало 2 из 2.
        score = 1.0
        """

        query_values = query_attributes.get(attr_name)

        if not query_values:
            return None

        product_values = product_attributes.get(attr_name, [])

        if isinstance(query_values, str):
            query_values = [query_values]

        if isinstance(product_values, str):
            product_values = [product_values]

        query_set = set(query_values)
        product_set = set(product_values)

        if not query_set:
            return None

        matched = query_set.intersection(product_set)

        return len(matched) / len(query_set)

    def _price_score(self, product_attributes, query_attributes):
        """
        Считает совпадение по цене.

        Поддерживаем такой формат запроса:

        "price_min": 10000
        "price_max": 30000

        Если цена товара внутри диапазона — 1.
        Если цена чуть выше/ниже — постепенно уменьшаем score.
        """

        price_min = query_attributes.get("price_min")
        price_max = query_attributes.get("price_max")

        if price_min is None and price_max is None:
            return None

        product_price = product_attributes.get("price")

        if product_price is None:
            return 0.0

        product_price = float(product_price)

        if price_min is not None:
            price_min = float(price_min)

        if price_max is not None:
            price_max = float(price_max)

        # Если есть и минимум, и максимум.
        if price_min is not None and price_max is not None:
            if price_min <= product_price <= price_max:
                return 1.0

            # Если товар дешевле минимума.
            if product_price < price_min:
                diff = price_min - product_price
                return max(0.0, 1.0 - diff / price_min)

            # Если товар дороже максимума.
            diff = product_price - price_max
            return max(0.0, 1.0 - diff / price_max)

        # Если задан только максимум.
        if price_max is not None:
            if product_price <= price_max:
                return 1.0

            diff = product_price - price_max
            return max(0.0, 1.0 - diff / price_max)

        # Если задан только минимум.
        if price_min is not None:
            if product_price >= price_min:
                return 1.0

            diff = price_min - product_price
            return max(0.0, 1.0 - diff / price_min)

        return None

    def _furniture_attribute_score(self, product, query_attributes):
        """
        Attribute score именно для мебели.

        У мебели важные признаки:

        category
        product_type
        model
        dimensions
        color
        material
        features
        style_or_purpose
        price

        Здесь мы считаем, насколько товар совпал
        с признаками из запроса.
        """

        product_attributes = product.get("attributes", {})

        # Веса признаков внутри мебели.
        # Потом можно будет менять.
        furniture_attr_weights = {
            "product_type": 0.25,       # диван, стол, шкаф, кровать
            "model": 0.10,              # конкретная модель
            "dimensions": 0.10,         # размеры
            "color": 0.10,              # цвет
            "material": 0.15,           # дерево, ЛДСП, металл, ткань
            "features": 0.15,           # раскладной, с ящиками, ортопедический
            "style_or_purpose": 0.10,   # лофт, спальня, гостиная, офис
            "price": 0.05,              # цена
        }

        total_score = 0.0
        total_weight = 0.0

        # Простые точные атрибуты.
        exact_attrs = [
            "product_type",
            "model",
            "dimensions",
            "color",
            "material",
            "style_or_purpose",
        ]

        for attr_name in exact_attrs:
            attr_score = self._exact_attribute_score(
                product_attributes,
                query_attributes,
                attr_name
            )

            if attr_score is None:
                continue

            weight = furniture_attr_weights[attr_name]
            total_score += weight * attr_score
            total_weight += weight

        # Особенности могут быть списком.
        features_score = self._list_attribute_score(
            product_attributes,
            query_attributes,
            "features"
        )

        if features_score is not None:
            weight = furniture_attr_weights["features"]
            total_score += weight * features_score
            total_weight += weight

        # Цена считается отдельно.
        price_score = self._price_score(
            product_attributes,
            query_attributes
        )

        if price_score is not None:
            weight = furniture_attr_weights["price"]
            total_score += weight * price_score
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def _attribute_score(self, product, query_attributes):
        """
        Общая функция attribute_score.

        Сейчас отдельно обрабатываем furniture.
        Потом сюда можно добавить:
        smartphones
        shoes
        pet_food
        clothes
        и т.д.
        """

        category = product.get("category")

        if category == "furniture":
            return self._furniture_attribute_score(
                product,
                query_attributes
            )

        # Если категория другая, используем простой общий подсчёт.
        product_attributes = product.get("attributes", {})

        if not query_attributes:
            return 0.0

        score = 0
        max_score = 0

        for attr_name, query_value in query_attributes.items():
            if query_value is None:
                continue

            if attr_name in ["price_min", "price_max"]:
                continue

            max_score += 1

            product_value = product_attributes.get(attr_name)

            if product_value == query_value:
                score += 1

        if max_score == 0:
            return 0.0

        return score / max_score

    def _get_category_weights(self, category):
        """
        Веса для итоговой формулы.

        text_score — BM25.
        embedding_score — BGE-M3.
        attribute_score — точные характеристики.
        """

        default_weights = {
            "text_score": 0.40,
            "embedding_score": 0.45,
            "attribute_score": 0.15,
        }

        category_weights = {
            # Для мебели атрибуты довольно важны:
            # тип товара, материал, цвет, размеры, цена.
            "furniture": {
                "text_score": 0.35,
                "embedding_score": 0.35,
                "attribute_score": 0.30,
            },
        }

        return category_weights.get(category, default_weights)

    def _filter_by_categories(self, categories):
        """
        Оставляет только товары из нужных категорий.

        Если categories пустой список,
        ищем по всем товарам.
        """

        if not categories:
            return list(range(len(self.products)))

        result_indexes = []

        for index, product in enumerate(self.products):
            if product.get("category") in categories:
                result_indexes.append(index)

        return result_indexes

    def search(
        self,
        prepared_query,
        top_k=10,
        bm25_candidates_count=50,
        embedding_candidates_count=50
    ):
        """
        Главная функция поиска.

        prepared_query должен быть уже подготовлен отдельно.

        Пример:

        {
            "text": "бежевый раскладной диван ткань гостиная до 30000",
            "categories": ["furniture"],
            "attributes": {
                "product_type": "диван",
                "color": "бежевый",
                "material": "ткань",
                "features": ["раскладной"],
                "style_or_purpose": "гостиная",
                "price_max": 30000
            }
        }
        """

        query_text = prepared_query["text"]
        query_categories = prepared_query.get("categories", [])
        query_attributes = prepared_query.get("attributes", {})

        allowed_indexes = self._filter_by_categories(query_categories)

        if not allowed_indexes:
            return []

        query_tokens = self._tokenize(query_text)

        # 1. BM25 score.
        bm25_raw_scores = self.bm25.get_scores(query_tokens)
        bm25_scores = self._normalize_scores(bm25_raw_scores)

        # 2. Embedding score через BGE-M3.
        query_output = self.embedding_model.encode(
            [query_text],
            batch_size=1,
            max_length=512
        )

        query_embedding = query_output["dense_vecs"][0]
        query_embedding = np.array(query_embedding, dtype=np.float32)

        embedding_raw_scores = self.product_embeddings @ query_embedding
        embedding_scores = self._normalize_scores(embedding_raw_scores)

        # 3. Берём кандидатов только из нужных категорий.
        allowed_indexes_set = set(allowed_indexes)

        bm25_sorted_indexes = np.argsort(bm25_scores)[::-1]
        embedding_sorted_indexes = np.argsort(embedding_scores)[::-1]

        bm25_top_indexes = []

        for index in bm25_sorted_indexes:
            index = int(index)

            if index in allowed_indexes_set:
                bm25_top_indexes.append(index)

            if len(bm25_top_indexes) >= bm25_candidates_count:
                break

        embedding_top_indexes = []

        for index in embedding_sorted_indexes:
            index = int(index)

            if index in allowed_indexes_set:
                embedding_top_indexes.append(index)

            if len(embedding_top_indexes) >= embedding_candidates_count:
                break

        # 4. Объединяем кандидатов.
        candidate_indexes = set()

        for index in bm25_top_indexes:
            candidate_indexes.add(index)

        for index in embedding_top_indexes:
            candidate_indexes.add(index)

        # 5. Считаем final_score.
        results = []

        for index in candidate_indexes:
            product = self.products[index]

            text_score = float(bm25_scores[index])
            embedding_score = float(embedding_scores[index])
            attribute_score = self._attribute_score(
                product,
                query_attributes
            )

            category = product.get("category")
            weights = self._get_category_weights(category)

            final_score = (
                weights["text_score"] * text_score
                + weights["embedding_score"] * embedding_score
                + weights["attribute_score"] * attribute_score
            )

            results.append({
                "product_id": product["id"],
                "name": product["name"],
                "category": product.get("category"),
                "text_score": round(text_score, 4),
                "embedding_score": round(embedding_score, 4),
                "attribute_score": round(attribute_score, 4),
                "final_score": round(float(final_score), 4),
                "price": product.get("attributes", {}).get("price"),
            })

        results.sort(key=lambda item: item["final_score"], reverse=True)

        return results[:top_k]


if __name__ == "__main__":
    # Пример товаров мебели.
    #
    # Эти товары уже подготовлены.
    # search_text должен быть очищен отдельным модулем.
    products = [
        {
            "id": 1,
            "name": "Диван Милан раскладной бежевый",
            "category": "furniture",
            "search_text": (
                "мебель диван милан раскладной бежевый ткань "
                "для гостиной современный мягкий 220x95x90"
            ),
            "attributes": {
                "product_type": "диван",
                "model": "милан",
                "dimensions": "220x95x90",
                "color": "бежевый",
                "material": "ткань",
                "features": ["раскладной", "мягкий"],
                "style_or_purpose": "гостиная",
                "price": 29990,
            }
        },
        {
            "id": 2,
            "name": "Диван Лофт серый прямой",
            "category": "furniture",
            "search_text": (
                "мебель диван лофт прямой серый велюр "
                "для гостиной современный 210x90x85"
            ),
            "attributes": {
                "product_type": "диван",
                "model": "лофт",
                "dimensions": "210x90x85",
                "color": "серый",
                "material": "велюр",
                "features": ["мягкий"],
                "style_or_purpose": "гостиная",
                "price": 34990,
            }
        },
        {
            "id": 3,
            "name": "Обеденный стол Норд дуб",
            "category": "furniture",
            "search_text": (
                "мебель стол обеденный норд дуб дерево "
                "для кухни скандинавский 140x80x75"
            ),
            "attributes": {
                "product_type": "стол",
                "model": "норд",
                "dimensions": "140x80x75",
                "color": "дуб",
                "material": "дерево",
                "features": ["обеденный"],
                "style_or_purpose": "кухня",
                "price": 18990,
            }
        },
        {
            "id": 4,
            "name": "Шкаф Прага белый двухдверный",
            "category": "furniture",
            "search_text": (
                "мебель шкаф прага белый лдсп двухдверный "
                "для спальни хранение 180x80x50"
            ),
            "attributes": {
                "product_type": "шкаф",
                "model": "прага",
                "dimensions": "180x80x50",
                "color": "белый",
                "material": "лдсп",
                "features": ["двухдверный", "для хранения"],
                "style_or_purpose": "спальня",
                "price": 15990,
            }
        },
        {
            "id": 5,
            "name": "Кровать Соната с ящиками",
            "category": "furniture",
            "search_text": (
                "мебель кровать соната с ящиками белый лдсп "
                "для спальни 160x200 хранение"
            ),
            "attributes": {
                "product_type": "кровать",
                "model": "соната",
                "dimensions": "160x200",
                "color": "белый",
                "material": "лдсп",
                "features": ["с ящиками", "для хранения"],
                "style_or_purpose": "спальня",
                "price": 27990,
            }
        },
    ]

    search_engine = HybridProductSearch(products)

    # Пример уже подготовленного запроса.
    #
    # Пользователь мог написать:
    # "бежевый раскладной диван в гостиную до 30000"
    #
    # А отдельный preprocessing-модуль сделал:
    prepared_query = {
        "text": "бежевый раскладной диван ткань гостиная до 30000",
        "categories": ["furniture"],
        "attributes": {
            "product_type": "диван",
            "color": "бежевый",
            "material": "ткань",
            "features": ["раскладной"],
            "style_or_purpose": "гостиная",
            "price_max": 30000,
        }
    }

    results = search_engine.search(prepared_query, top_k=5)

    for item in results:
        print()
        print("Товар:", item["name"])
        print("Категория:", item["category"])
        print("Цена:", item["price"])
        print("BM25:", item["text_score"])
        print("Embedding:", item["embedding_score"])
        print("Attributes:", item["attribute_score"])
        print("Final:", item["final_score"])