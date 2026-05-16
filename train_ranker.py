import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel
from xgboost import XGBRanker


# ============================================================
# 1. КОНФИГ ДОМЕНА ПОИСКА
# ============================================================

@dataclass
class SearchDomainConfig:
    """
    Конфиг одного типа товаров.

    Сейчас у нас один домен:
        furniture

    Потом можно будет добавить:
        food
        electronics
        clothes
        cosmetics
        и т.д.

    searchable_fields:
        список полей, по которым мы делаем embedding-поиск.

    field_embedding_weights:
        веса отдельных embedding-признаков.
        Они используются для ручного объединения embedding-score.
        Даже если потом XGBoost сам учится ранжировать,
        общий field_embedding_score всё равно полезен как отдельный признак.

    missing_field_scores:
        score, который ставится, если пользователь указал признак,
        а у товара этот признак неизвестен и стоит "-".
    """

    searchable_fields: list[str]
    field_embedding_weights: dict[str, float]
    missing_field_scores: dict[str, float]


# ============================================================
# 2. НАСТРОЙКИ ДЛЯ МЕБЕЛИ
# ============================================================

DOMAIN_CONFIGS = {
    "furniture": SearchDomainConfig(
        # Эти поля участвуют в embedding-поиске.
        #
        # Важно:
        # price тут нет, потому что цену мы НЕ отправляем
        # ни в BM25, ни в embedding.
        #
        # Цена будет учитываться только отдельно через price_score.
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

        # Эти веса нужны, чтобы собрать один общий field_embedding_score.
        #
        # Например:
        # description важнее, чем dimensions,
        # product_type важнее, чем style_or_purpose.
        #
        # Потом XGBoost получит:
        # - общий field_embedding_score
        # - отдельные scores по каждому полю
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

        # Если пользователь указал признак, а у товара этот признак "-"
        # или пустой, мы не всегда хотим ставить 0.
        #
        # Например:
        # пользователь ищет "черный диван",
        # а у товара цвет неизвестен.
        #
        # Это не значит, что товар точно не подходит.
        # Это значит "мы не знаем цвет".
        #
        # Поэтому можно поставить 0.5.
        #
        # Для product_type ставим ниже, потому что если тип товара неизвестен,
        # это более серьёзная проблема.
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


# ============================================================
# 3. ПРОСТЫЕ УТИЛИТЫ ДЛЯ ПОЛЕЙ
# ============================================================

def is_empty(value):
    """
    Проверяем, является ли значение пустым.

    По твоей вводной:
        пустая ячейка обозначается "-"

    Поэтому:
        None
        ""
        "-"
    считаем отсутствием значения.
    """

    return value is None or value == "" or value == "-"


def field_value(value):
    """
    Приводим значение поля к строке.

    Если значение пустое, возвращаем пустую строку.

    Пример:
        field_value("-")        -> ""
        field_value(None)       -> ""
        field_value("черный")   -> "черный"
        field_value(44900)      -> "44900"
    """

    if is_empty(value):
        return ""

    return str(value)


def tokenize(text):
    """
    Разбиваем текст на слова для BM25.

    BM25 работает не со строкой целиком,
    а со списком токенов.

    Пример:
        "черный угловой диван"
    станет:
        ["черный", "угловой", "диван"]
    """

    text = field_value(text)

    if not text:
        return []

    return text.split()


def normalize_vector(vector):
    """
    Нормализуем embedding-вектор.

    Зачем:
        чтобы потом dot product был похож на cosine similarity.

    Если вектор нормализован, то:
        vector_a @ vector_b
    показывает смысловую близость.
    """

    vector = np.array(vector, dtype=np.float32)

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def normalize_scores(scores):
    """
    Нормализуем scores в диапазон 0...1.

    Почему это нужно:
        BM25 может выдавать числа типа 0, 5, 20.
        Embedding может выдавать 0.3, 0.7, 0.9.
        Price может выдавать 0...1.

    Чтобы все эти признаки можно было сравнивать,
    приводим их к одному диапазону.
    """

    scores = np.array(scores, dtype=np.float32)

    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    # Если все scores одинаковые, деление на 0 невозможно.
    # Тогда возвращаем нули.
    if max_score - min_score < 1e-9:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


# ============================================================
# 4. РАБОТА С ЦЕНОЙ
# ============================================================

def price_to_float(value):
    """
    Превращаем цену в float.

    На входе может быть:
        44900
        "44900"
        "-"
        None

    Если цены нет, возвращаем None.
    """

    if is_empty(value):
        return None

    try:
        return float(value)
    except ValueError:
        return None


def calculate_price_score(product_price, price_min=None, price_max=None):
    """
    Считаем price_score.

    Цена НЕ участвует:
        - в BM25
        - в embedding

    Цена участвует только как отдельный числовой признак
    для итогового ранжирования.

    Возможные случаи:

    1. price_max = 50000
       товар стоит 44900 -> score = 1.0

    2. price_max = 50000
       товар стоит 70000 -> score меньше 1.0

    3. price_min = 30000
       товар стоит 20000 -> score меньше 1.0

    4. price_min и price_max не указаны
       price_score = 0.0
    """

    # Если пользователь не указал цену,
    # цена не должна влиять.
    if price_min is None and price_max is None:
        return 0.0

    product_price = price_to_float(product_price)

    # Если у товара нет цены, но пользователь цену указал,
    # товар получает 0 по цене.
    if product_price is None:
        return 0.0

    price_min = price_to_float(price_min)
    price_max = price_to_float(price_max)

    # Случай: задан диапазон цены.
    if price_min is not None and price_max is not None:
        if price_min <= product_price <= price_max:
            return 1.0

        if product_price < price_min:
            diff = price_min - product_price
            return max(0.0, 1.0 - diff / price_min)

        diff = product_price - price_max
        return max(0.0, 1.0 - diff / price_max)

    # Случай: указана только верхняя граница.
    if price_max is not None:
        if product_price <= price_max:
            return 1.0

        diff = product_price - price_max
        return max(0.0, 1.0 - diff / price_max)

    # Случай: указана только нижняя граница.
    if price_min is not None:
        if product_price >= price_min:
            return 1.0

        diff = price_min - product_price
        return max(0.0, 1.0 - diff / price_min)

    return 0.0


# ============================================================
# 5. ЗАГРУЗКА И СОХРАНЕНИЕ JSON
# ============================================================

def load_json(path):
    """
    Загружаем JSON-файл.

    products.json:
        база товаров

    train_queries.json:
        обучающие запросы и правильные ответы
    """

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    """
    Сохраняем данные в JSON.

    Используется для сохранения списка feature_names.
    """

    path = Path(path)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# ============================================================
# 6. FEATURE ENGINE
# ============================================================

class FeatureSearchEngine:
    """
    Этот класс НЕ обучает модель.

    Его задача:
        1. построить BM25;
        2. построить embedding-индексы;
        3. для пары query-product посчитать признаки;
        4. подготовить данные для XGBoost Ranker.

    То есть он превращает пару:
        запрос + товар

    в числовой вектор:
        [
            exact_id_score,
            bm25_score,
            field_embedding_score,
            price_score,
            description_embedding_score,
            product_type_embedding_score,
            ...
        ]
    """

    def __init__(self, products, domain_name="furniture"):
        """
        products:
            список товаров.

        Каждый товар должен быть уже очищен и подготовлен:

        {
            "id": "1001",
            "description": "...",
            "characteristics": {
                "product_type": "...",
                "model": "...",
                "dimensions": "-",
                "color": "...",
                "material": "...",
                "features": "...",
                "style_or_purpose": "..."
            },
            "price": 44900
        }
        """

        self.products = products
        self.domain_name = domain_name

        if domain_name not in DOMAIN_CONFIGS:
            raise ValueError(f"Unknown domain: {domain_name}")

        self.config = DOMAIN_CONFIGS[domain_name]

        # Быстрый словарь:
        # product_id -> index в списке products.
        #
        # Он нужен, чтобы при обучении гарантированно добавлять
        # релевантные товары в кандидаты.
        self.product_id_to_index = {}

        for index, product in enumerate(self.products):
            product_id = field_value(product.get("id"))
            self.product_id_to_index[product_id] = index

        # Загружаем embedding-модель.
        #
        # BAAI/bge-m3 — сильная мультиязычная модель.
        # use_fp16=False безопаснее для CPU.
        self.embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False,
        )

        # Здесь будет BM25-индекс.
        self.bm25 = None

        # Здесь будут embedding-векторы по каждому полю.
        #
        # Пример:
        # self.field_embeddings["color"]
        # хранит embedding цветов всех товаров.
        self.field_embeddings = {}

        # Список признаков, которые будут подаваться в XGBoost.
        self.feature_names = self._build_feature_names()

        # Строим BM25 и embeddings.
        self._build_indexes()

    # --------------------------------------------------------
    # 6.1. СПИСОК ПРИЗНАКОВ ДЛЯ XGBOOST
    # --------------------------------------------------------

    def _build_feature_names(self):
        """
        Создаём список признаков в фиксированном порядке.

        Почему фиксированный порядок важен:
            XGBoost обучается на матрице чисел.
            Он не знает названий колонок внутри numpy-массива.
            Поэтому при обучении и предсказании порядок признаков
            должен быть одинаковым.

        Базовые признаки:
            exact_id_score
            bm25_score
            field_embedding_score
            price_score
            has_query_id
            has_price_filter

        Потом добавляем:
            description_embedding_score
            product_type_embedding_score
            ...

        Потом добавляем:
            product_has_description
            product_has_color
            ...

        product_has_* нужен, чтобы модель понимала,
        был ли признак у товара реально указан,
        или score появился из-за missing_field_score.
        """

        feature_names = [
            "exact_id_score",
            "bm25_score",
            "field_embedding_score",
            "price_score",
            "has_query_id",
            "has_price_filter",
        ]

        # Отдельные embedding-score по каждому полю.
        for field_name in self.config.searchable_fields:
            feature_names.append(f"{field_name}_embedding_score")

        # Флаги наличия полей у товара.
        for field_name in self.config.searchable_fields:
            feature_names.append(f"product_has_{field_name}")

        return feature_names

    # --------------------------------------------------------
    # 6.2. ПОСТРОЕНИЕ ИНДЕКСОВ
    # --------------------------------------------------------

    def _build_indexes(self):
        """
        Строим два типа индексов:

        1. BM25 по description.
           Только описание, без характеристик, потому что характеристики
           уже были извлечены из описания и не надо дублировать слова.

        2. Embedding-индексы отдельно по каждому полю.
           description отдельно,
           product_type отдельно,
           color отдельно,
           material отдельно и т.д.
        """

        self._build_bm25_index()
        self._build_field_embedding_indexes()

    def _build_bm25_index(self):
        """
        Строим BM25 по описаниям товаров.

        Важно:
            BM25 ищет только по description.

        Почему не добавляем характеристики:
            ты сказала, что характеристики уже извлечены из описания.
            Если добавить их снова, получится дубль слов.
        """

        descriptions = []

        for product in self.products:
            description = field_value(product.get("description", ""))
            descriptions.append(description)

        tokens = []

        for description in descriptions:
            tokens.append(tokenize(description))

        self.bm25 = BM25Okapi(tokens)

    def _build_field_embedding_indexes(self):
        """
        Строим embedding-индекс отдельно для каждого поля.

        Пример:
            поле color:
                товар 1 -> "черный"
                товар 2 -> "серый"
                товар 3 -> "орех"

            self.field_embeddings["color"] будет матрицей:
                [
                    vector("черный"),
                    vector("серый"),
                    vector("орех")
                ]

        Так мы можем считать похожесть отдельно по каждому признаку.
        """

        for field_name in self.config.searchable_fields:
            texts = []

            for product in self.products:
                value = self._get_product_field_value(product, field_name)
                texts.append(value)

            self.field_embeddings[field_name] = self._encode_texts(texts)

    # --------------------------------------------------------
    # 6.3. ПОЛУЧЕНИЕ ЗНАЧЕНИЙ ПОЛЕЙ
    # --------------------------------------------------------

    def _get_product_field_value(self, product, field_name):
        """
        Достаём значение поля у товара.

        Если field_name == "description",
        берём product["description"].

        Иначе берём:
            product["characteristics"][field_name]

        Если там "-", возвращаем "".
        """

        if field_name == "description":
            return field_value(product.get("description", ""))

        characteristics = product.get("characteristics", {})

        return field_value(characteristics.get(field_name, ""))

    def _get_query_field_value(self, query, field_name):
        """
        Достаём значение поля из запроса.

        Запрос имеет такой же формат:

        {
            "id": "",
            "description": "...",
            "characteristics": {
                "product_type": "...",
                "color": "...",
                ...
            },
            "price_max": 50000
        }
        """

        if field_name == "description":
            return field_value(query.get("description", ""))

        characteristics = query.get("characteristics", {})

        return field_value(characteristics.get(field_name, ""))

    # --------------------------------------------------------
    # 6.4. EMBEDDING
    # --------------------------------------------------------

    def _encode_texts(self, texts):
        """
        Кодируем список текстов в embedding-векторы.

        Если текст пустой, ставим " ".
        Это нужно, чтобы модель не падала на пустых строках.

        Но дальше при подсчёте score мы всё равно отдельно проверяем:
            если у товара поле пустое,
            ставим missing_field_score.
        """

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
        """
        Кодируем один текст в один embedding-вектор.
        """

        vectors = self._encode_texts([text])
        return vectors[0]

    # --------------------------------------------------------
    # 6.5. EXACT ID SCORE
    # --------------------------------------------------------

    def exact_id_scores(self, query):
        """
        Считаем точное совпадение по id.

        Логика:
            если query["id"] == product["id"]:
                score = 1.0
            иначе:
                score = 0.0

        Это строгое совпадение.
        Никакой похожести, никаких частичных совпадений.
        """

        query_id = field_value(query.get("id", ""))

        scores = np.zeros(len(self.products), dtype=np.float32)

        if not query_id:
            return scores

        for index, product in enumerate(self.products):
            product_id = field_value(product.get("id"))

            if product_id == query_id:
                scores[index] = 1.0

        return scores

    # --------------------------------------------------------
    # 6.6. BM25 SCORE
    # --------------------------------------------------------

    def bm25_scores(self, query):
        """
        Считаем BM25 score по description запроса.

        Важно:
            BM25 не использует цену.
            BM25 не использует характеристики отдельно.
            Только description.
        """

        description = field_value(query.get("description", ""))

        if not description:
            return np.zeros(len(self.products), dtype=np.float32)

        query_tokens = tokenize(description)
        raw_scores = self.bm25.get_scores(query_tokens)

        return normalize_scores(raw_scores)

    # --------------------------------------------------------
    # 6.7. EMBEDDING SCORE ПО КАЖДОМУ ПОЛЮ
    # --------------------------------------------------------

    def field_embedding_scores(self, query):
        """
        Считаем embedding-score отдельно по каждому полю.

        Пример запроса:
            color = "черный"
            material = "экокожа"

        Тогда считаем:
            color_embedding_score
            material_embedding_score

        Если в запросе поле "-",
        оно не участвует.

        Если у товара поле "-",
        ставим missing_field_score.
        """

        result = {}

        for field_name in self.config.searchable_fields:
            query_value = self._get_query_field_value(query, field_name)

            # Если пользователь не указал этот признак,
            # он вообще не участвует.
            if not query_value:
                continue

            if field_name not in self.field_embeddings:
                continue

            query_vector = self._encode_one_text(query_value)

            product_vectors = self.field_embeddings[field_name]

            # Так как векторы нормализованы,
            # dot product работает как cosine similarity.
            raw_scores = product_vectors @ query_vector

            scores = normalize_scores(raw_scores)

            missing_score = self.config.missing_field_scores.get(
                field_name,
                0.5,
            )

            # Если у товара поле отсутствует,
            # ставим не 0, а настраиваемый missing_score.
            #
            # Например:
            # color = "-" может получить 0.5,
            # потому что неизвестный цвет не значит "точно не подходит".
            for index, product in enumerate(self.products):
                product_value = self._get_product_field_value(
                    product,
                    field_name,
                )

                if not product_value:
                    scores[index] = missing_score

            result[field_name] = scores

        return result

    def combined_field_embedding_scores(self, field_scores):
        """
        Собираем общий field_embedding_score из отдельных scores.

        Например:
            description_embedding_score = 0.90
            product_type_embedding_score = 0.95
            color_embedding_score = 0.80

        Тогда:
            field_embedding_score =
                0.30 * description
              + 0.20 * product_type
              + 0.10 * color

        Потом делим на сумму использованных весов.
        """

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

    # --------------------------------------------------------
    # 6.8. PRICE SCORE
    # --------------------------------------------------------

    def price_scores(self, query):
        """
        Считаем price_score для всех товаров.

        Если в запросе нет price_min и price_max,
        возвращаем нули.

        Важно:
            даже если price_score = 0,
            XGBoost увидит has_price_filter,
            поэтому поймёт, была ли цена в запросе.
        """

        price_min = query.get("price_min")
        price_max = query.get("price_max")

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

    # --------------------------------------------------------
    # 6.9. ВСЕ SCORES ДЛЯ ОДНОГО ЗАПРОСА
    # --------------------------------------------------------

    def all_scores_for_query(self, query):
        """
        Считаем все базовые score для одного запроса:

        exact_id_score
        bm25_score
        field_embedding_score
        price_score
        field_scores по каждому полю
        """

        exact_id = self.exact_id_scores(query)
        bm25 = self.bm25_scores(query)

        field_scores = self.field_embedding_scores(query)

        combined_field_embedding = self.combined_field_embedding_scores(
            field_scores
        )

        price = self.price_scores(query)

        return {
            "exact_id_score": exact_id,
            "bm25_score": bm25,
            "field_embedding_score": combined_field_embedding,
            "price_score": price,
            "field_scores": field_scores,
        }

    # --------------------------------------------------------
    # 6.10. КАНДИДАТЫ ДЛЯ ОБУЧЕНИЯ И ПОИСКА
    # --------------------------------------------------------

    def get_top_indexes(self, scores, limit):
        """
        Берём индексы товаров с самыми большими scores.

        Например:
            top-200 по BM25
            top-200 по embedding
        """

        sorted_indexes = np.argsort(scores)[::-1]

        result = []

        for index in sorted_indexes:
            index = int(index)
            result.append(index)

            if len(result) >= limit:
                break

        return result

    def generate_candidates(
        self,
        query,
        relevant_ids=None,
        bm25_limit=200,
        embedding_limit=200,
        id_limit=10,
    ):
        """
        Генерируем кандидатов для одного запроса.

        Зачем:
            нельзя обучать XGBoost на всех товарах,
            если товаров очень много.

        Поэтому берём:
            - top по exact_id
            - top по BM25
            - top по embedding
            - обязательно добавляем relevant_ids

        Почему обязательно добавляем relevant_ids:
            если правильный товар не попал в кандидаты,
            модель не сможет на нём учиться.
        """

        if relevant_ids is None:
            relevant_ids = []

        all_scores = self.all_scores_for_query(query)

        candidate_indexes = set()

        exact_id_scores = all_scores["exact_id_score"]
        bm25_scores = all_scores["bm25_score"]
        embedding_scores = all_scores["field_embedding_score"]

        # Добавляем товары, найденные по точному id.
        for index in self.get_top_indexes(exact_id_scores, id_limit):
            if exact_id_scores[index] > 0:
                candidate_indexes.add(index)

        # Добавляем top по BM25.
        for index in self.get_top_indexes(bm25_scores, bm25_limit):
            candidate_indexes.add(index)

        # Добавляем top по embedding.
        for index in self.get_top_indexes(embedding_scores, embedding_limit):
            candidate_indexes.add(index)

        # Обязательно добавляем все правильные ответы.
        for product_id in relevant_ids:
            product_id = field_value(product_id)

            if product_id in self.product_id_to_index:
                candidate_indexes.add(self.product_id_to_index[product_id])

        return sorted(candidate_indexes), all_scores

    # --------------------------------------------------------
    # 6.11. FEATURE VECTOR ДЛЯ ОДНОГО ТОВАРА
    # --------------------------------------------------------

    def build_feature_vector(self, query, product_index, all_scores):
        """
        Строим числовой вектор признаков для пары:

            query + product

        Именно эти признаки пойдут в XGBoost.

        Порядок признаков должен строго совпадать
        с self.feature_names.
        """

        product = self.products[product_index]

        has_query_id = 1.0 if field_value(query.get("id", "")) else 0.0

        has_price_filter = (
            query.get("price_min") is not None
            or query.get("price_max") is not None
        )

        feature_values = {
            "exact_id_score": float(all_scores["exact_id_score"][product_index]),
            "bm25_score": float(all_scores["bm25_score"][product_index]),
            "field_embedding_score": float(
                all_scores["field_embedding_score"][product_index]
            ),
            "price_score": float(all_scores["price_score"][product_index]),
            "has_query_id": float(has_query_id),
            "has_price_filter": float(has_price_filter),
        }

        field_scores = all_scores["field_scores"]

        # Добавляем отдельные embedding-score по каждому полю.
        for field_name in self.config.searchable_fields:
            key = f"{field_name}_embedding_score"

            if field_name in field_scores:
                feature_values[key] = float(field_scores[field_name][product_index])
            else:
                feature_values[key] = 0.0

        # Добавляем флаги наличия признака у товара.
        #
        # Например:
        # product_has_color = 0,
        # если у товара color = "-"
        for field_name in self.config.searchable_fields:
            key = f"product_has_{field_name}"

            product_value = self._get_product_field_value(
                product,
                field_name,
            )

            feature_values[key] = 1.0 if product_value else 0.0

        feature_vector = []

        for feature_name in self.feature_names:
            feature_vector.append(feature_values.get(feature_name, 0.0))

        return feature_vector


# ============================================================
# 7. СОЗДАНИЕ ОБУЧАЮЩЕГО ДАТАСЕТА
# ============================================================

def build_training_dataset(
    search_engine,
    training_items,
    bm25_limit=200,
    embedding_limit=200,
):
    """
    Строим обучающий датасет для XGBoost.

    На входе training_items из train_queries.json:

    [
        {
            "query_id": "q1",
            "query": {...},
            "relevant_ids": ["1001", "1050"]
        }
    ]

    Для каждого запроса:
        1. генерируем кандидатов;
        2. считаем feature_vector для каждого кандидата;
        3. ставим label:
            1, если товар в relevant_ids
            0, если товар не в relevant_ids

    Возвращаем:
        X   — матрица признаков
        y   — метки 0/1
        qid — номер запроса для каждой строки
    """

    X = []
    y = []
    qid = []

    rows_debug = []

    for query_number, item in enumerate(training_items):
        query = item["query"]

        relevant_ids = set(
            field_value(x)
            for x in item["relevant_ids"]
        )

        candidate_indexes, all_scores = search_engine.generate_candidates(
            query=query,
            relevant_ids=relevant_ids,
            bm25_limit=bm25_limit,
            embedding_limit=embedding_limit,
        )

        for product_index in candidate_indexes:
            product = search_engine.products[product_index]
            product_id = field_value(product.get("id"))

            # Если товар есть в правильных ответах — label = 1.
            # Иначе label = 0.
            label = 1 if product_id in relevant_ids else 0

            feature_vector = search_engine.build_feature_vector(
                query=query,
                product_index=product_index,
                all_scores=all_scores,
            )

            X.append(feature_vector)
            y.append(label)

            # qid говорит XGBoost:
            # какие строки относятся к одному запросу.
            qid.append(query_number)

            rows_debug.append({
                "query_number": query_number,
                "query_id": item.get("query_id", str(query_number)),
                "product_id": product_id,
                "label": label,
            })

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    qid = np.array(qid, dtype=np.int32)

    return X, y, qid, rows_debug


# ============================================================
# 8. ОБУЧЕНИЕ XGBOOST RANKER
# ============================================================

def train_xgb_ranker(X, y, qid):
    """
    Обучаем XGBoost Ranker.

    objective="rank:ndcg":
        модель оптимизирует ранжирование,
        а не обычную классификацию.

    eval_metric="ndcg@10":
        качество смотрится по top-10.

    qid:
        показывает, какие товары принадлежат одному запросу.
    """

    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
    )

    model.fit(
        X,
        y,
        qid=qid,
        verbose=False,
    )

    return model


# ============================================================
# 9. ПОИСК С УЖЕ ОБУЧЕННЫМ XGBOOST
# ============================================================

class XGBoostRerankerSearch:
    """
    Этот класс уже использует обученную модель.

    Схема:
        1. Получаем новый запрос.
        2. Генерируем кандидатов через BM25 + embedding.
        3. Для каждого кандидата считаем признаки.
        4. XGBoost предсказывает ranker_score.
        5. Сортируем товары по ranker_score.
    """

    def __init__(self, feature_engine, ranker_model):
        self.feature_engine = feature_engine
        self.ranker_model = ranker_model

    def search(
        self,
        query,
        top_k=100,
        bm25_limit=300,
        embedding_limit=300,
    ):
        """
        Возвращаем top_k товаров.

        На выходе:
            product_id
            description
            characteristics
            price
            ranker_score
        """

        candidate_indexes, all_scores = self.feature_engine.generate_candidates(
            query=query,
            relevant_ids=[],
            bm25_limit=bm25_limit,
            embedding_limit=embedding_limit,
        )

        if not candidate_indexes:
            return []

        X = []

        for product_index in candidate_indexes:
            feature_vector = self.feature_engine.build_feature_vector(
                query=query,
                product_index=product_index,
                all_scores=all_scores,
            )

            X.append(feature_vector)

        X = np.array(X, dtype=np.float32)

        predicted_scores = self.ranker_model.predict(X)

        results = []

        for product_index, predicted_score in zip(candidate_indexes, predicted_scores):
            product = self.feature_engine.products[product_index]

            result = {
                "product_id": product["id"],
                "description": product.get("description", ""),
                "characteristics": product.get("characteristics", {}),
                "price": product.get("price"),
                "ranker_score": round(float(predicted_score), 6),
            }

            results.append(result)

        results.sort(
            key=lambda item: item["ranker_score"],
            reverse=True,
        )

        return results[:top_k]


# ============================================================
# 10. СОХРАНЕНИЕ И ЗАГРУЗКА МОДЕЛИ
# ============================================================

def save_model(model, feature_names, output_dir):
    """
    Сохраняем модель XGBoost и порядок признаков.

    Порядок признаков обязательно надо сохранять,
    потому что при predict порядок должен быть тем же,
    что и при обучении.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgb_ranker.json"
    features_path = output_dir / "feature_names.json"

    model.save_model(str(model_path))
    save_json(features_path, feature_names)


def load_model(model_dir):
    """
    Загружаем модель и feature_names.
    """

    model_dir = Path(model_dir)

    model = XGBRanker()
    model.load_model(str(model_dir / "xgb_ranker.json"))

    feature_names = load_json(model_dir / "feature_names.json")

    return model, feature_names


# ============================================================
# 11. ПРИМЕР ЗАПУСКА
# ============================================================

if __name__ == "__main__":
    # Файл с товарами.
    #
    # Пример products.json:
    #
    # [
    #   {
    #     "id": "1001",
    #     "description": "черный угловой диван из экокожи механизм дельфин стиль лофт",
    #     "characteristics": {
    #       "product_type": "диван угловой",
    #       "model": "лофт",
    #       "dimensions": "-",
    #       "color": "черный",
    #       "material": "экокожа",
    #       "features": "механизм дельфин",
    #       "style_or_purpose": "лофт"
    #     },
    #     "price": 44900
    #   }
    # ]

    products_path = "products.json"

    # Файл с обучающими запросами.
    #
    # Пример train_queries.json:
    #
    # [
    #   {
    #     "query_id": "q1",
    #     "query": {
    #       "id": "",
    #       "description": "черный угловой диван из экокожи",
    #       "characteristics": {
    #         "product_type": "диван угловой",
    #         "color": "черный",
    #         "material": "экокожа"
    #       },
    #       "price_max": 50000
    #     },
    #     "relevant_ids": ["1001", "1050"]
    #   }
    # ]

    train_queries_path = "train_queries.json"

    # Загружаем товары и обучающие запросы.
    products = load_json(products_path)
    train_queries = load_json(train_queries_path)

    # Создаём feature_engine.
    #
    # Он строит:
    # - BM25;
    # - embedding-индексы;
    # - умеет считать признаки для XGBoost.
    feature_engine = FeatureSearchEngine(
        products=products,
        domain_name="furniture",
    )

    # Собираем обучающую матрицу.
    X, y, qid, rows_debug = build_training_dataset(
        search_engine=feature_engine,
        training_items=train_queries,
        bm25_limit=200,
        embedding_limit=200,
    )

    print("Train rows:", len(X))
    print("Feature count:", X.shape[1])
    print("Positive labels:", int(np.sum(y)))
    print("Feature names:", feature_engine.feature_names)

    # Обучаем XGBoost Ranker.
    ranker = train_xgb_ranker(
        X=X,
        y=y,
        qid=qid,
    )

    # Сохраняем модель.
    save_model(
        model=ranker,
        feature_names=feature_engine.feature_names,
        output_dir="trained_ranker",
    )

    # Создаём поисковую систему с обученной моделью.
    search_system = XGBoostRerankerSearch(
        feature_engine=feature_engine,
        ranker_model=ranker,
    )

    # Пример нового запроса.
    example_query = {
        "id": "",
        "description": "черный угловой диван из экокожи",
        "characteristics": {
            "product_type": "диван угловой",
            "color": "черный",
            "material": "экокожа",
            "features": "механизм дельфин",
            "style_or_purpose": "лофт",
        },
        "price_max": 50000,
    }

    # Получаем top-100 результатов.
    results = search_system.search(
        query=example_query,
        top_k=100,
    )

    # Печатаем первые 10 для проверки.
    for item in results[:10]:
        print()
        print("ID:", item["product_id"])
        print("Score:", item["ranker_score"])
        print("Описание:", item["description"])
        print("Характеристики:", item["characteristics"])
        print("Цена:", item["price"])

    # После обучения можно посмотреть важность признаков.
    #
    # Это НЕ прямые веса формулы.
    # Это важность признаков внутри XGBoost.
    print()
    print("Feature importances:")

    importances = ranker.feature_importances_

    pairs = list(zip(feature_engine.feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)

    for name, value in pairs:
        print(name, value)