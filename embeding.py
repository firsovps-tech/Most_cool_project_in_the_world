import csv
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel


# ============================================================
# 1. ИЗВЕСТНЫЕ КАТЕГОРИИ
# ============================================================

# Это категории, которые система уже знает.
# Если в данных появится категория, которой нет в этом списке,
# мы считаем её новой.
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


# ============================================================
# 2. ВЕСА ИТОГОВОГО СКОРА
# ============================================================

# Это веса для финальной формулы.
#
# text_score:
#   BM25, точный поиск по словам.
#   Здесь можно учитывать артикул.
#
# general_embedding_score:
#   общий embedding товара.
#   Здесь артикул НЕ учитываем.
#
# category_embedding_score:
#   embedding категории/типа товара.
#   Здесь артикул тоже НЕ учитываем.
#
# attribute_score:
#   точные признаки: категория, тип товара, модель, цвет, материал и т.д.
SCORE_WEIGHTS = {
    "text_score": 0.30,
    "general_embedding_score": 0.30,
    "category_embedding_score": 0.20,
    "attribute_score": 0.20,
}


# ============================================================
# 3. ВЕСА ПРИЗНАКОВ ДЛЯ ATTRIBUTE_SCORE
# ============================================================

# Эти веса используются внутри attribute_score.
#
# Например, если в запросе указан тип товара "Диван угловой",
# то совпадение по product_type весит 0.20.
#
# Если указан цвет, совпадение по color весит 0.12.
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
# 4. ЗАГЛУШКА ДЛЯ НОВЫХ КАТЕГОРИЙ
# ============================================================

def handle_new_categories(new_categories):
    """
    Функция-заглушка.

    Она вызывается, если в данных найдены новые категории,
    которых нет в KNOWN_CATEGORIES.

    Сейчас она ничего не делает.

    Потом сюда можно вставить:
    - сохранение новых категорий в файл;
    - логирование;
    - отправку уведомления;
    - автоматическое добавление категории;
    - запуск отдельной обработки.
    """

    return


# ============================================================
# 5. НОРМАЛИЗАЦИЯ ТЕКСТА
# ============================================================

def normalize_text(text):
    """
    Минимальная нормализация текста.

    Так как ты сказала, что данные идеально подготовлены,
    здесь нет сложной очистки.

    Делаем только:
    - None -> "";
    - приводим к строке;
    - убираем пробелы по краям;
    - приводим к нижнему регистру;
    - заменяем "ё" на "е";
    - убираем лишние пробелы;
    - "-" считаем пустым значением.
    """

    if text is None:
        return ""

    text = str(text).strip().lower()
    text = text.replace("ё", "е")
    text = " ".join(text.split())

    if text == "-":
        return ""

    return text


def tokenize(text):
    """
    Токенизация для BM25.

    BM25 работает не с целой строкой,
    а со списком слов.

    Пример:
    "диван угловой лофт"
    ->
    ["диван", "угловой", "лофт"]
    """

    return normalize_text(text).split()


# ============================================================
# 6. ЗАГРУЗКА ГОТОВЫХ ТОВАРОВ ИЗ CSV/TXT
# ============================================================

def load_products_from_csv(file_path):
    """
    Загружает товары из файла с разделителем ';'.

    ВАЖНО:
    мы считаем, что данные уже идеально подготовлены.

    То есть:
    - id уже есть;
    - цена уже нормальная;
    - категории уже подготовлены;
    - признаки уже подготовлены.

    В файле должны быть колонки:

    id
    Артикул
    Категория
    Тип товара
    Модель
    Габариты
    Цвет
    Материал
    Особенности
    Стиль/Назначение
    Цена (₽)

    Если колонка "Артикул" отсутствует, article будет пустой строкой.
    """

    products = []
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")

        for row in reader:
            # Берём готовый уникальный id из файла.
            # Мы больше НЕ создаём make_product_id.
            product_id = row["id"]

            # Артикул нужен только для точного поиска BM25.
            # Если артикула нет, будет пустая строка.
            article = row.get("Артикул", "")

            # Основные поля товара.
            category = row["Категория"]
            product_type = row["Тип товара"]
            model = row["Модель"]
            dimensions = row["Габариты"]
            color = row["Цвет"]
            material = row["Материал"]
            features = row["Особенности"]
            style_or_purpose = row["Стиль/Назначение"]
            price = row["Цена (₽)"]

            # Название товара делаем из типа товара и модели.
            # Например:
            # Тип товара = "Диван угловой"
            # Модель = "Лофт"
            # name = "Диван угловой Лофт"
            name = " ".join(
                part for part in [product_type, model]
                if normalize_text(part)
            )

            # ------------------------------------------------
            # TEXT FOR EXACT SEARCH
            # ------------------------------------------------
            #
            # Это текст для BM25.
            #
            # Здесь МЫ УЧИТЫВАЕМ АРТИКУЛ.
            #
            # Почему:
            # если пользователь ввёл артикул, BM25 должен найти товар.
            exact_search_text = " ".join([
                article,
                category,
                product_type,
                model,
                dimensions,
                color,
                material,
                features,
                style_or_purpose,
                str(price),
            ])

            # ------------------------------------------------
            # TEXT FOR GENERAL EMBEDDING
            # ------------------------------------------------
            #
            # Это общий текст товара для embedding.
            #
            # Здесь АРТИКУЛ НЕ УЧИТЫВАЕМ.
            #
            # Почему:
            # артикул — технический код.
            # Он не несёт смысл товара.
            general_embedding_text = " ".join([
                category,
                product_type,
                model,
                dimensions,
                color,
                material,
                features,
                style_or_purpose,
            ])

            # ------------------------------------------------
            # TEXT FOR CATEGORY EMBEDDING
            # ------------------------------------------------
            #
            # Это отдельный текст для категориального embedding.
            #
            # Он помогает понять категорию/тип/назначение товара.
            #
            # Здесь тоже НЕТ артикула.
            category_embedding_text = " ".join([
                category,
                product_type,
                style_or_purpose,
            ])

            # Собираем товар в удобный словарь.
            product = {
                "id": product_id,
                "article": article,
                "name": name,
                "category": category,

                # Текст для BM25.
                "exact_search_text": normalize_text(exact_search_text),

                # Текст для общего embedding.
                "general_embedding_text": normalize_text(general_embedding_text),

                # Текст для категориального embedding.
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
                },
            }

            products.append(product)

    return products


# ============================================================
# 7. ГИБРИДНЫЙ ПОИСК
# ============================================================

class HybridFurnitureSearch:
    """
    Класс гибридного поиска.

    В нём есть 4 источника оценки:

    1. text_score
       BM25 по exact_search_text.
       Учитывает артикул.

    2. general_embedding_score
       Общий embedding по general_embedding_text.
       Не учитывает артикул.

    3. category_embedding_score
       Категориальный embedding по category_embedding_text.
       Не учитывает артикул.

    4. attribute_score
       Точное совпадение структурированных признаков.
    """

    def __init__(self, products):
        """
        Конструктор поисковика.

        При создании:
        - сохраняем товары;
        - ищем новые категории;
        - загружаем модель BGE-M3;
        - строим BM25;
        - строим embeddings.
        """

        self.products = products

        # Ищем новые категории.
        self.new_categories = self._find_new_categories()

        # Если новые категории есть, вызываем заглушку.
        if self.new_categories:
            handle_new_categories(self.new_categories)

        # Загружаем сильную embedding-модель.
        #
        # use_fp16=False — безопаснее для CPU.
        # Если будет GPU, можно попробовать use_fp16=True.
        self.embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False
        )

        # Здесь будет BM25-индекс.
        self.bm25 = None

        # Здесь будут embedding-векторы товаров.
        self.general_product_embeddings = None
        self.category_product_embeddings = None

        # Строим индексы.
        self._build_indexes()

    def _find_new_categories(self):
        """
        Ищет категории из товаров, которых нет в KNOWN_CATEGORIES.

        Возвращает set.

        Например:
        если в данных появилась категория "Сад",
        а в KNOWN_CATEGORIES её нет,
        функция вернёт {"Сад"}.
        """

        found_categories = set()

        for product in self.products:
            category = product.get("category")

            if category:
                found_categories.add(category)

        return found_categories - KNOWN_CATEGORIES

    def _normalize_scores(self, scores):
        """
        Приводит scores к диапазону 0...1.

        Это нужно, чтобы можно было смешивать разные оценки.

        Например:
        BM25 может дать [0, 5, 20],
        embedding может дать [0.3, 0.5, 0.8].

        После нормализации оба списка будут в диапазоне 0...1.
        """

        scores = np.array(scores, dtype=np.float32)

        min_score = float(np.min(scores))
        max_score = float(np.max(scores))

        # Если все scores одинаковые,
        # то делить на 0 нельзя.
        if max_score - min_score < 1e-9:
            return np.zeros_like(scores)

        return (scores - min_score) / (max_score - min_score)

    def _build_indexes(self):
        """
        Строит три индекса:

        1. BM25 по exact_search_text.
        2. Общие embeddings по general_embedding_text.
        3. Категориальные embeddings по category_embedding_text.
        """

        # ----------------------------------------------------
        # 1. BM25 INDEX
        # ----------------------------------------------------

        # Берём тексты для точного поиска.
        # В этих текстах есть артикул.
        exact_texts = [
            product["exact_search_text"]
            for product in self.products
        ]

        # Разбиваем каждый текст на слова.
        exact_tokens = [
            tokenize(text)
            for text in exact_texts
        ]

        # Строим BM25.
        self.bm25 = BM25Okapi(exact_tokens)

        # ----------------------------------------------------
        # 2. GENERAL EMBEDDING INDEX
        # ----------------------------------------------------

        # Берём общие embedding-тексты.
        # В них нет артикула.
        general_texts = [
            product["general_embedding_text"]
            for product in self.products
        ]

        # Считаем embeddings через BGE-M3.
        general_output = self.embedding_model.encode(
            general_texts,
            batch_size=8,
            max_length=512
        )

        # Достаём dense-векторы.
        self.general_product_embeddings = np.array(
            general_output["dense_vecs"],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # 3. CATEGORY EMBEDDING INDEX
        # ----------------------------------------------------

        # Берём категориальные embedding-тексты.
        category_texts = [
            product["category_embedding_text"]
            for product in self.products
        ]

        # Считаем категориальные embeddings.
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
        Проверяет точное совпадение одного признака.

        Если query_value пустой, значит признак в запросе не указан,
        поэтому возвращаем None и не учитываем его.

        Если значения совпали — 1.0.
        Если не совпали — 0.0.
        """

        if query_value is None:
            return None

        product_value = normalize_text(product_value)
        query_value = normalize_text(query_value)

        if not query_value:
            return None

        if product_value == query_value:
            return 1.0

        return 0.0

    def _features_score(self, product_features, query_features):
        """
        Сравнивает особенности товара.

        Ожидаем, что features — это строка с признаками через запятую.

        Например:
        product_features = "механизм дельфин, с подножкой"
        query_features = "механизм дельфин"

        Совпало 1 из 1 -> score = 1.0.
        """

        if not query_features:
            return None

        product_features = normalize_text(product_features)
        query_features = normalize_text(query_features)

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
        """
        Считает совпадение по цене.

        Поддерживает:
        - price_min
        - price_max

        Если цена внутри диапазона — 1.0.
        Если цена немного вне диапазона — score уменьшается.
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
        Считает совпадение точных признаков товара с запросом.

        Например query_attributes:

        {
            "category": "Гостиная",
            "product_type": "Диван угловой",
            "color": "чёрный",
            "material": "экокожа",
            "features": "механизм дельфин",
            "style_or_purpose": "лофт",
            "price_max": 50000
        }

        Для каждого признака считаем совпадение,
        умножаем на вес признака,
        потом делим на сумму использованных весов.
        """

        product_attributes = product.get("attributes", {})

        total_score = 0.0
        total_weight = 0.0

        # Эти признаки проверяем точным сравнением.
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

            # None значит, что такого признака в запросе не было.
            if field_score is None:
                continue

            weight = ATTRIBUTE_WEIGHTS[field]

            total_score += weight * field_score
            total_weight += weight

        # Особенности считаем отдельно.
        features_score = self._features_score(
            product_attributes.get("features"),
            query_attributes.get("features")
        )

        if features_score is not None:
            weight = ATTRIBUTE_WEIGHTS["features"]

            total_score += weight * features_score
            total_weight += weight

        # Цену считаем отдельно, потому что она может быть диапазоном.
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

        prepared_query уже должен быть подготовлен заранее.

        Важно:
        - text_for_exact может содержать артикул;
        - text_for_embedding НЕ должен содержать артикул;
        - text_for_category_embedding НЕ должен содержать артикул.

        Пример:

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
                "features": "механизм дельфин",
                "style_or_purpose": "лофт",
                "price_max": 50000
            }
        }
        """

        # ----------------------------------------------------
        # 1. ДОСТАЁМ ПОДГОТОВЛЕННЫЕ ЧАСТИ ЗАПРОСА
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 2. BM25 SCORE
        # ----------------------------------------------------
        #
        # Тут артикул учитывается,
        # если он был в text_for_exact.

        exact_tokens = tokenize(text_for_exact)

        bm25_raw_scores = self.bm25.get_scores(exact_tokens)
        text_scores = self._normalize_scores(bm25_raw_scores)

        # ----------------------------------------------------
        # 3. GENERAL EMBEDDING SCORE
        # ----------------------------------------------------
        #
        # Тут артикул не учитывается.

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
        # 4. CATEGORY EMBEDDING SCORE
        # ----------------------------------------------------
        #
        # Тут тоже артикул не учитывается.

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
        # 5. ФИЛЬТР ПО КАТЕГОРИЯМ
        # ----------------------------------------------------
        #
        # Если categories пустой список, ищем по всем товарам.
        # Если categories = ["Гостиная"], ищем только в этой категории.

        allowed_indexes = []

        for index, product in enumerate(self.products):
            product_category = product.get("category")

            if not categories or product_category in categories:
                allowed_indexes.append(index)

        allowed_indexes_set = set(allowed_indexes)

        if not allowed_indexes:
            return []

        # ----------------------------------------------------
        # 6. БЕРЁМ КАНДИДАТОВ ИЗ BM25
        # ----------------------------------------------------

        bm25_sorted_indexes = np.argsort(text_scores)[::-1]
        bm25_top_indexes = []

        for index in bm25_sorted_indexes:
            index = int(index)

            if index in allowed_indexes_set:
                bm25_top_indexes.append(index)

            if len(bm25_top_indexes) >= bm25_candidates_count:
                break

        # ----------------------------------------------------
        # 7. БЕРЁМ КАНДИДАТОВ ИЗ GENERAL EMBEDDING
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # 8. БЕРЁМ КАНДИДАТОВ ИЗ CATEGORY EMBEDDING
        # ----------------------------------------------------

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
        # 9. ОБЪЕДИНЯЕМ КАНДИДАТОВ
        # ----------------------------------------------------
        #
        # Один товар может прийти из разных источников:
        # - BM25;
        # - общий embedding;
        # - категориальный embedding.
        #
        # Используем set, чтобы не было дублей.

        candidate_indexes = set()

        for index in bm25_top_indexes:
            candidate_indexes.add(index)

        for index in general_embedding_top_indexes:
            candidate_indexes.add(index)

        for index in category_embedding_top_indexes:
            candidate_indexes.add(index)

        # ----------------------------------------------------
        # 10. ФИНАЛЬНЫЙ СКОРИНГ
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
# 8. ПРИМЕР ЗАПУСКА
# ============================================================

if __name__ == "__main__":
    # Путь к файлу.
    file_path = "Qwen_csv_20260516_iyqyxkuif.txt"

    # Загружаем товары.
    products = load_products_from_csv(file_path)

    # Создаём поисковик.
    search_engine = HybridFurnitureSearch(products)

    # Пример уже подготовленного запроса.
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
            "features": "механизм дельфин",
            "style_or_purpose": "лофт",
            "price_max": 50000,
        }
    }

    # Запускаем поиск.
    results = search_engine.search(prepared_query, top_k=10)

    # Показываем новые категории, если они есть.
    print("Новые категории:", search_engine.new_categories)

    # Печатаем результаты.
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