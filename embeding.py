import csv
import hashlib
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel


# ============================================================
# 1. НАСТРОЙКИ
# ============================================================

# Известные категории.
# Сюда можно заранее записать категории, которые мы уже умеем обрабатывать.
# Если в файле появится новая категория, которой нет здесь,
# код её найдёт и вызовет функцию-заглушку.
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


# Веса итогового скоринга.
# Потом их можно поменять после тестов.
#
# text_score:
#   поиск по точным словам, BM25.
#   Здесь учитываем артикул.
#
# general_embedding_score:
#   общий смысловой embedding товара.
#   Здесь артикул НЕ учитываем.
#
# category_embedding_score:
#   embedding по категориальному описанию.
#   Здесь тоже артикул НЕ учитываем.
#
# attribute_score:
#   точное совпадение признаков: тип, модель, цвет, материал, цена и т.д.
SCORE_WEIGHTS = {
    "text_score": 0.30,
    "general_embedding_score": 0.30,
    "category_embedding_score": 0.20,
    "attribute_score": 0.20,
}


# Веса признаков для мебели.
# Это внутренняя формула attribute_score.
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


# ============================================================
# 2. ФУНКЦИЯ-ЗАГЛУШКА ДЛЯ НОВЫХ КАТЕГОРИЙ
# ============================================================

def handle_new_categories(new_categories):
    """
    Заглушка для обработки новых категорий.

    Сейчас она ничего не делает, только return.

    Потом сюда можно вставить любую логику:
    - записать новые категории в файл;
    - отправить уведомление;
    - автоматически создать веса;
    - добавить категорию в базу;
    - запустить ручную разметку.

    new_categories — это set с названиями новых категорий.
    Например:
    {"Сад", "Офис", "Гараж"}
    """

    # Пока специально ничего не делаем.
    # Это место для будущей логики.
    return


# ============================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def safe_str(value):
    """
    Аккуратно превращает значение в строку.

    Если значение None или пустое — вернём пустую строку.
    Это нужно, чтобы код не падал на пустых ячейках.
    """

    if value is None:
        return ""

    value = str(value).strip()

    if value == "-":
        return ""

    return value


def normalize_text(text):
    """
    Небольшая нормализация текста.

    Важно:
    тут не делаем сложную очистку, потому что ты сказала,
    что очистка и подготовка категорий происходят отдельно.

    Но базово:
    - приводим к нижнему регистру;
    - заменяем ё на е;
    - убираем лишние пробелы.
    """

    text = safe_str(text).lower()
    text = text.replace("ё", "е")
    text = " ".join(text.split())

    return text


def tokenize(text):
    """
    Разбиваем текст на токены для BM25.

    BM25 работает не с целой строкой, а со списком слов.

    Пример:
    "диван раскладной бежевый"
    станет:
    ["диван", "раскладной", "бежевый"]
    """

    text = normalize_text(text)
    return text.split()


def make_article(row, row_number):
    """
    Получаем артикул товара.

    Если в файле есть колонка:
    - Артикул
    - article
    - sku

    то берём её.

    Если артикула нет, генерируем стабильный технический артикул
    на основе номера строки.

    Почему это нужно:
    пользователь может искать товар по артикулу.
    BM25 должен это учитывать.

    Но embedding НЕ должен учитывать артикул,
    потому что артикул не несёт смысл товара.
    """

    possible_article_fields = [
        "Артикул",
        "article",
        "sku",
        "SKU",
        "id",
        "ID",
    ]

    for field in possible_article_fields:
        value = safe_str(row.get(field))
        if value:
            return value

    # Если артикула в файле нет, делаем свой.
    # Например: AUTO-000001
    return f"AUTO-{row_number:06d}"


def make_product_id(article):
    """
    Делаем технический product_id из артикула.

    product_id нужен для объединения кандидатов.

    В реальной базе product_id обычно уже есть.
    Здесь мы делаем его сами.
    """

    raw = article.encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def parse_price(value):
    """
    Превращает цену в число.

    В файле цена может быть строкой:
    "15700"
    "15 700"
    "15 700 ₽"

    Функция оставляет только цифры.
    """

    value = safe_str(value)

    digits = ""

    for char in value:
        if char.isdigit():
            digits += char

    if not digits:
        return None

    return int(digits)


def split_features(value):
    """
    Разбивает особенности товара на список.

    Например:
    "механический, с подножкой"
    станет:
    ["механический", "с подножкой"]

    Если особенностей нет, вернём пустой список.
    """

    value = safe_str(value)

    if not value:
        return []

    parts = []

    for part in value.split(","):
        part = normalize_text(part)
        if part:
            parts.append(part)

    return parts


# ============================================================
# 4. ЗАГРУЗКА ТОВАРОВ ИЗ ФАЙЛА
# ============================================================

def load_products_from_semicolon_csv(file_path):
    """
    Загружает товары из CSV/TXT файла с разделителем ';'.

    В твоём файле колонки такие:

    Категория
    Тип товара
    Модель
    Габариты
    Цвет
    Материал
    Особенности
    Стиль/Назначение
    Цена (₽)

    Функция превращает каждую строку в словарь товара.
    """

    products = []

    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row_number, row in enumerate(reader, start=1):
            article = make_article(row, row_number)
            product_id = make_product_id(article)

            category = safe_str(row.get("Категория"))
            product_type = safe_str(row.get("Тип товара"))
            model = safe_str(row.get("Модель"))
            dimensions = safe_str(row.get("Габариты"))
            color = safe_str(row.get("Цвет"))
            material = safe_str(row.get("Материал"))
            features_raw = safe_str(row.get("Особенности"))
            style_or_purpose = safe_str(row.get("Стиль/Назначение"))
            price = parse_price(row.get("Цена (₽)"))

            features = split_features(features_raw)

            # Название товара собираем из типа товара и модели.
            # Например:
            # "Диван угловой Лофт"
            name_parts = [product_type, model]
            name = " ".join(part for part in name_parts if part)

            if not name:
                name = f"Товар {article}"

            # ------------------------------------------------
            # ТЕКСТ ДЛЯ ТОЧНОГО ПОИСКА BM25
            # ------------------------------------------------
            #
            # Здесь мы УЧИТЫВАЕМ АРТИКУЛ.
            #
            # Почему:
            # если пользователь вводит артикул, BM25 должен быстро найти товар.
            #
            # Но этот текст НЕ идёт в embedding.
            exact_search_text = " ".join([
                article,
                category,
                product_type,
                model,
                dimensions,
                color,
                material,
                features_raw,
                style_or_purpose,
                str(price) if price is not None else "",
            ])

            # ------------------------------------------------
            # ОБЩИЙ ТЕКСТ ДЛЯ EMBEDDING
            # ------------------------------------------------
            #
            # Здесь АРТИКУЛ НЕ УЧИТЫВАЕМ.
            #
            # Почему:
            # артикул — это технический код.
            # Он не помогает понять смысл товара.
            #
            # Embedding должен понимать:
            # диван, материал, цвет, стиль, назначение, особенности.
            general_embedding_text = " ".join([
                category,
                product_type,
                model,
                dimensions,
                color,
                material,
                features_raw,
                style_or_purpose,
            ])

            # ------------------------------------------------
            # КАТЕГОРИАЛЬНЫЙ ТЕКСТ ДЛЯ EMBEDDING
            # ------------------------------------------------
            #
            # Это отдельный embedding, который помогает понять,
            # к какой группе относится товар.
            #
            # Здесь тоже НЕ учитываем артикул.
            #
            # Берём более категориальные поля:
            # категория, тип товара, стиль/назначение.
            category_embedding_text = " ".join([
                category,
                product_type,
                style_or_purpose,
            ])

            product = {
                "id": product_id,
                "article": article,
                "name": name,

                # Основная категория из файла:
                # Ванная, Гостиная, Спальня, Кухня и т.д.
                "category": category,

                # Текст для BM25.
                # В нём есть артикул.
                "exact_search_text": normalize_text(exact_search_text),

                # Текст для общего embedding.
                # В нём нет артикула.
                "general_embedding_text": normalize_text(general_embedding_text),

                # Текст для категориального embedding.
                # В нём нет артикула.
                "category_embedding_text": normalize_text(category_embedding_text),

                # Структурированные признаки товара.
                "attributes": {
                    "category": category,
                    "product_type": product_type,
                    "model": model,
                    "dimensions": dimensions,
                    "color": color,
                    "material": material,
                    "features": features,
                    "style_or_purpose": style_or_purpose,
                    "price": price,
                }
            }

            products.append(product)

    return products


# ============================================================
# 5. ПОИСКОВЫЙ КЛАСС
# ============================================================

class HybridFurnitureSearch:
    """
    Гибридный поиск по товарам.

    Внутри есть 4 сигнала:

    1. text_score:
       BM25 по exact_search_text.
       Тут учитывается артикул.

    2. general_embedding_score:
       Общий embedding по товару.
       Артикул НЕ учитывается.

    3. category_embedding_score:
       Категориальный embedding.
       Артикул НЕ учитывается.

    4. attribute_score:
       Точное совпадение признаков.
    """

    def __init__(self, products):
        """
        products — список товаров после load_products_from_semicolon_csv().
        """

        self.products = products

        # ----------------------------------------------------
        # Проверяем новые категории
        # ----------------------------------------------------
        #
        # Если в файле есть категория, которой нет в KNOWN_CATEGORIES,
        # мы её считаем новой.
        self.new_categories = self._find_new_categories()

        # Если новые категории есть — вызываем заглушку.
        if self.new_categories:
            handle_new_categories(self.new_categories)

        # ----------------------------------------------------
        # Загружаем BGE-M3
        # ----------------------------------------------------
        #
        # Это сильная embedding-модель.
        #
        # use_fp16=False — безопаснее для CPU.
        # Если есть GPU, можно поставить True.
        self.embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False
        )

        # BM25 индекс.
        self.bm25 = None

        # Embedding-векторы.
        self.general_product_embeddings = None
        self.category_product_embeddings = None

        # Строим все индексы.
        self._build_indexes()

    def _find_new_categories(self):
        """
        Ищет новые категории в товарах.

        Новая категория — это категория, которой нет в KNOWN_CATEGORIES.

        Возвращает set.
        """

        found_categories = set()

        for product in self.products:
            category = product.get("category")
            if category:
                found_categories.add(category)

        new_categories = found_categories - KNOWN_CATEGORIES

        return new_categories

    def _normalize_scores(self, scores):
        """
        Приводит список оценок к диапазону 0...1.

        Это нужно, чтобы BM25 score, embedding score
        и attribute score можно было смешивать.
        """

        scores = np.array(scores, dtype=np.float32)

        min_score = float(np.min(scores))
        max_score = float(np.max(scores))

        if max_score - min_score < 1e-9:
            return np.zeros_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def _build_indexes(self):
        """
        Строит 3 индекса:

        1. BM25 по exact_search_text.
        2. Общие embedding-векторы по general_embedding_text.
        3. Категориальные embedding-векторы по category_embedding_text.
        """

        # ----------------------------------------------------
        # 1. BM25
        # ----------------------------------------------------
        #
        # BM25 должен учитывать артикул.
        # Поэтому используем exact_search_text.
        exact_texts = [
            product["exact_search_text"]
            for product in self.products
        ]

        exact_tokens = [
            tokenize(text)
            for text in exact_texts
        ]

        self.bm25 = BM25Okapi(exact_tokens)

        # ----------------------------------------------------
        # 2. Общий embedding
        # ----------------------------------------------------
        #
        # Тут артикул не учитывается.
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

        # ----------------------------------------------------
        # 3. Категориальный embedding
        # ----------------------------------------------------
        #
        # Тут тоже артикул не учитывается.
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

    def _exact_match(self, product_value, query_value):
        """
        Проверяет точное совпадение двух значений.

        Если query_value пустой, возвращаем None,
        потому что этот признак не надо учитывать.

        Если совпало — 1.0.
        Если не совпало — 0.0.
        """

        if query_value is None:
            return None

        query_value = safe_str(query_value)
        product_value = safe_str(product_value)

        if not query_value:
            return None

        if normalize_text(product_value) == normalize_text(query_value):
            return 1.0

        return 0.0

    def _features_score(self, product_features, query_features):
        """
        Считает совпадение особенностей.

        Например:
        query_features = ["раздвижной", "с подсветкой"]

        product_features = ["раздвижной", "с подсветкой", "мягкий"]

        Совпало 2 из 2 → score = 1.0
        """

        if not query_features:
            return None

        if isinstance(query_features, str):
            query_features = [query_features]

        if isinstance(product_features, str):
            product_features = [product_features]

        query_set = set(normalize_text(x) for x in query_features if x)
        product_set = set(normalize_text(x) for x in product_features if x)

        if not query_set:
            return None

        matched = query_set.intersection(product_set)

        return len(matched) / len(query_set)

    def _price_score(self, product_price, query_attributes):
        """
        Считает совпадение по цене.

        Поддерживаем:
        price_min
        price_max

        Если цена товара внутри диапазона — 1.0.
        Если цена вне диапазона — score плавно уменьшается.
        """

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
        """
        Считает точное совпадение признаков товара с запросом.

        query_attributes может быть таким:

        {
            "category": "Гостиная",
            "product_type": "Диван угловой",
            "color": "чёрный",
            "material": "экокожа",
            "features": ["механизм дельфин"],
            "style_or_purpose": "лофт",
            "price_max": 50000
        }
        """

        product_attributes = product.get("attributes", {})

        total_score = 0.0
        total_weight = 0.0

        # Простые точные признаки.
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

        # Особенности считаем отдельно, потому что это список.
        features_score = self._features_score(
            product_attributes.get("features", []),
            query_attributes.get("features")
        )

        if features_score is not None:
            weight = ATTRIBUTE_WEIGHTS["features"]
            total_score += weight * features_score
            total_weight += weight

        # Цена считается отдельно, потому что это диапазон.
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

    def search(
        self,
        prepared_query,
        top_k=10,
        bm25_candidates_count=80,
        general_embedding_candidates_count=80,
        category_embedding_candidates_count=80
    ):
        """
        Главная функция поиска.

        prepared_query — уже подготовленный запрос.

        Пример:

        prepared_query = {
            "text_for_exact": "ART-001 диван угловой лофт черный",
            "text_for_embedding": "диван угловой черный экокожа лофт",
            "text_for_category_embedding": "гостиная диван лофт",
            "categories": ["Гостиная"],
            "attributes": {
                "category": "Гостиная",
                "product_type": "Диван угловой",
                "color": "чёрный",
                "material": "экокожа",
                "style_or_purpose": "лофт",
                "price_max": 50000
            }
        }

        Важно:
        - text_for_exact может содержать артикул.
        - text_for_embedding НЕ должен содержать артикул.
        - text_for_category_embedding НЕ должен содержать артикул.
        """

        # Текст для точного поиска.
        # Тут артикул можно учитывать.
        text_for_exact = normalize_text(
            prepared_query.get("text_for_exact", "")
        )

        # Текст для общего embedding.
        # Тут артикула быть не должно.
        text_for_embedding = normalize_text(
            prepared_query.get("text_for_embedding", "")
        )

        # Текст для категориального embedding.
        # Тут артикула тоже быть не должно.
        text_for_category_embedding = normalize_text(
            prepared_query.get("text_for_category_embedding", "")
        )

        # Категории, где надо искать.
        categories = prepared_query.get("categories", [])

        # Структурированные признаки из запроса.
        query_attributes = prepared_query.get("attributes", {})

        # ----------------------------------------------------
        # 1. BM25 score
        # ----------------------------------------------------
        #
        # Здесь учитывается артикул, если он есть в text_for_exact.
        exact_tokens = tokenize(text_for_exact)

        bm25_raw_scores = self.bm25.get_scores(exact_tokens)
        text_scores = self._normalize_scores(bm25_raw_scores)

        # ----------------------------------------------------
        # 2. Общий embedding score
        # ----------------------------------------------------
        #
        # Здесь артикул не учитываем.
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

        # ----------------------------------------------------
        # 3. Категориальный embedding score
        # ----------------------------------------------------
        #
        # Здесь тоже артикул не учитываем.
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

        # ----------------------------------------------------
        # 4. Фильтр по категориям
        # ----------------------------------------------------
        #
        # Если categories пустой список, ищем по всем товарам.
        allowed_indexes = []

        for index, product in enumerate(self.products):
            product_category = product.get("category")

            if not categories or product_category in categories:
                allowed_indexes.append(index)

        allowed_indexes_set = set(allowed_indexes)

        if not allowed_indexes:
            return []

        # ----------------------------------------------------
        # 5. Берём кандидатов из трёх источников
        # ----------------------------------------------------

        # Топ по BM25.
        bm25_sorted_indexes = np.argsort(text_scores)[::-1]

        bm25_top_indexes = []

        for index in bm25_sorted_indexes:
            index = int(index)

            if index in allowed_indexes_set:
                bm25_top_indexes.append(index)

            if len(bm25_top_indexes) >= bm25_candidates_count:
                break

        # Топ по общему embedding.
        general_embedding_sorted_indexes = np.argsort(
            general_embedding_scores
        )[::-1]

        general_embedding_top_indexes = []

        for index in general_embedding_sorted_indexes:
            index = int(index)

            if index in allowed_indexes_set:
                general_embedding_top_indexes.append(index)

            if len(general_embedding_top_indexes) >= general_embedding_candidates_count:
                break

        # Топ по категориальному embedding.
        category_embedding_sorted_indexes = np.argsort(
            category_embedding_scores
        )[::-1]

        category_embedding_top_indexes = []

        for index in category_embedding_sorted_indexes:
            index = int(index)

            if index in allowed_indexes_set:
                category_embedding_top_indexes.append(index)

            if len(category_embedding_top_indexes) >= category_embedding_candidates_count:
                break

        # ----------------------------------------------------
        # 6. Объединяем кандидатов
        # ----------------------------------------------------
        #
        # Один и тот же товар может прийти из BM25,
        # общего embedding и категориального embedding.
        #
        # Поэтому используем set, чтобы товар не дублировался.
        candidate_indexes = set()

        for index in bm25_top_indexes:
            candidate_indexes.add(index)

        for index in general_embedding_top_indexes:
            candidate_indexes.add(index)

        for index in category_embedding_top_indexes:
            candidate_indexes.add(index)

        # ----------------------------------------------------
        # 7. Финальный скоринг
        # ----------------------------------------------------

        results = []

        for index in candidate_indexes:
            product = self.products[index]

            text_score = float(text_scores[index])
            general_embedding_score = float(general_embedding_scores[index])
            category_embedding_score = float(category_embedding_scores[index])

            attribute_score = self._attribute_score(
                product,
                query_attributes
            )

            final_score = (
                SCORE_WEIGHTS["text_score"] * text_score
                + SCORE_WEIGHTS["general_embedding_score"] * general_embedding_score
                + SCORE_WEIGHTS["category_embedding_score"] * category_embedding_score
                + SCORE_WEIGHTS["attribute_score"] * attribute_score
            )

            results.append({
                "product_id": product["id"],
                "article": product["article"],
                "name": product["name"],
                "category": product["category"],
                "price": product["attributes"].get("price"),
                "text_score": round(text_score, 4),
                "general_embedding_score": round(general_embedding_score, 4),
                "category_embedding_score": round(category_embedding_score, 4),
                "attribute_score": round(attribute_score, 4),
                "final_score": round(float(final_score), 4),
            })

        results.sort(key=lambda item: item["final_score"], reverse=True)

        return results[:top_k]


# ============================================================
# 6. ПРИМЕР ЗАПУСКА
# ============================================================

if __name__ == "__main__":
    # Укажи путь к своему файлу.
    #
    # Например:
    # file_path = "Qwen_csv_20260516_iyqyxkuif.txt"
    #
    # Если файл лежит рядом с этим Python-файлом, так и оставляй.
    file_path = "Qwen_csv_20260516_iyqyxkuif.txt"

    products = load_products_from_semicolon_csv(file_path)

    search_engine = HybridFurnitureSearch(products)

    # Пример подготовленного запроса.
    #
    # Важно:
    # prepared_query должен приходить уже после твоего модуля очистки.
    #
    # text_for_exact:
    #   для BM25, можно включать артикул.
    #
    # text_for_embedding:
    #   для общего embedding, артикул НЕ включаем.
    #
    # text_for_category_embedding:
    #   для категориального embedding, артикул НЕ включаем.
    prepared_query = {
        "text_for_exact": "диван угловой лофт черный механизм дельфин до 50000",
        "text_for_embedding": "диван угловой черный экокожа механизм дельфин стиль лофт",
        "text_for_category_embedding": "гостиная диван угловой лофт",
        "categories": ["Гостиная"],
        "attributes": {
            "category": "Гостиная",
            "product_type": "Диван угловой",
            "color": "чёрный",
            "material": "экокожа",
            "features": ["механизм дельфин"],
            "style_or_purpose": "лофт",
            "price_max": 50000,
        }
    }

    results = search_engine.search(prepared_query, top_k=10)

    print("Новые категории:", search_engine.new_categories)

    for item in results:
        print()
        print("Товар:", item["name"])
        print("Артикул:", item["article"])
        print("Категория:", item["category"])
        print("Цена:", item["price"])
        print("BM25:", item["text_score"])
        print("General embedding:", item["general_embedding_score"])
        print("Category embedding:", item["category_embedding_score"])
        print("Attributes:", item["attribute_score"])
        print("Final:", item["final_score"])