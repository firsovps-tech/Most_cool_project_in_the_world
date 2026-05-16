import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from xgboost import XGBRanker


def load_json(file_path):
    # Просто читаем json-файл.
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path, data):
    # Сохраняем json красиво, чтобы потом можно было руками посмотреть.
    file_path = Path(file_path)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def make_answer_key(query_id, product_id):
    # По этой паре будем соединять датасет и ответы.
    return str(query_id), str(product_id)


def load_answers_map(answers_path):
    # Делаем быстрый словарь:
    # (query_id, product_id) -> label
    #
    # Так потом не надо каждый раз искать ответ в списке.
    answers = load_json(answers_path)

    answers_map = {}

    for item in answers:
        query_id = item["query_id"]
        product_id = item["product_id"]
        label = item["label"]

        key = make_answer_key(query_id, product_id)

        answers_map[key] = float(label)

    return answers_map


def collect_feature_names(dataset):
    # Собираем все признаки, которые вообще встретились в датасете.
    #
    # Это важно, потому что для разных категорий товаров
    # признаки могут быть разными.
    #
    # Например:
    # furniture -> color, material, dimensions
    # food      -> taste, weight, calories
    feature_names = set()

    for row in dataset:
        features = row.get("features", {})

        for feature_name in features.keys():
            feature_names.add(feature_name)

    # Сортируем, чтобы порядок признаков был стабильным.
    # Иначе при каждом запуске может быть разный порядок колонок.
    return sorted(feature_names)


def collect_feature_names_by_category(dataset):
    # То же самое, но отдельно для каждой категории.
    #
    # На выходе будет примерно так:
    # {
    #   "furniture": ["bm25_score", "color_score", ...],
    #   "food": ["brand_score", "taste_score", ...]
    # }
    feature_names_by_category = defaultdict(set)

    for row in dataset:
        category = row.get("category", "default")
        features = row.get("features", {})

        for feature_name in features.keys():
            feature_names_by_category[category].add(feature_name)

    result = {}

    for category, feature_names in feature_names_by_category.items():
        result[category] = sorted(feature_names)

    return result


def build_train_matrix(dataset, answers_map, feature_names):
    # Превращаем строки из json в то, что понимает XGBoost:
    #
    # X   -> матрица признаков
    # y   -> ответы 0/1
    # qid -> номер запроса для каждой строки
    #
    # qid нужен именно для ranking-задачи.
    # Модель должна понимать, какие товары сравниваются внутри одного запроса.

    X = []
    y = []
    qid = []

    query_id_to_number = {}
    current_query_number = 0

    skipped_without_answer = 0

    for row in dataset:
        query_id = row["query_id"]
        product_id = row["product_id"]
        features = row.get("features", {})

        answer_key = make_answer_key(query_id, product_id)

        # Если для пары query-product нет ответа,
        # мы не можем использовать эту строку для обучения.
        if answer_key not in answers_map:
            skipped_without_answer += 1
            continue

        # XGBoost Ranker хочет числовой id группы.
        # Поэтому каждому query_id выдаём номер: 0, 1, 2, ...
        if query_id not in query_id_to_number:
            query_id_to_number[query_id] = current_query_number
            current_query_number += 1

        query_number = query_id_to_number[query_id]

        feature_vector = []

        # Идём строго по feature_names.
        # Если признака нет в строке — ставим 0.
        #
        # Это нормально для разных категорий:
        # у мебели может не быть calories_score,
        # у еды может не быть dimensions_score.
        for feature_name in feature_names:
            value = features.get(feature_name, 0.0)
            feature_vector.append(float(value))

        label = answers_map[answer_key]

        X.append(feature_vector)
        y.append(label)
        qid.append(query_number)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    qid = np.array(qid, dtype=np.int32)

    info = {
        "rows_used": len(X),
        "rows_skipped_without_answer": skipped_without_answer,
        "features_count": X.shape[1] if len(X) > 0 else 0,
        "queries_count": len(query_id_to_number),
        "positive_labels": int(np.sum(y)) if len(y) > 0 else 0,
    }

    return X, y, qid, info


def build_train_matrix_for_category(dataset, answers_map, category, feature_names):
    # Берём только строки одной категории.
    # Например только furniture или только food.
    filtered_dataset = []

    for row in dataset:
        row_category = row.get("category", "default")

        if row_category == category:
            filtered_dataset.append(row)

    return build_train_matrix(
        dataset=filtered_dataset,
        answers_map=answers_map,
        feature_names=feature_names,
    )


def train_xgb_ranker(X, y, qid):
    # Обучаем XGBoost именно как ranker.
    #
    # objective="rank:ndcg" говорит модели:
    # не просто угадывай 0/1, а учись правильно сортировать товары внутри запроса.

    if len(X) == 0:
        raise ValueError("Training matrix is empty")

    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",

        # Количество деревьев.
        # Больше — потенциально лучше, но дольше и больше риск переобучения.
        n_estimators=300,

        # Скорость обучения.
        # 0.05 — спокойный нормальный старт.
        learning_rate=0.05,

        # Глубина деревьев.
        # 6 — обычно норм для табличных признаков.
        max_depth=6,

        # Эти параметры немного защищают от переобучения.
        subsample=0.9,
        colsample_bytree=0.9,

        random_state=42,

        # Быстрый способ обучения деревьев.
        tree_method="hist",
    )

    model.fit(
        X,
        y,
        qid=qid,
        verbose=False,
    )

    return model


def save_ranker(model, feature_names, output_dir):
    # Сохраняем:
    # 1. саму модель
    # 2. порядок признаков
    #
    # Порядок признаков обязательно нужен.
    # При predict надо подавать признаки ровно в том же порядке.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xgb_ranker.json"
    feature_names_path = output_dir / "feature_names.json"

    model.save_model(str(model_path))
    save_json(feature_names_path, feature_names)


def print_feature_importance(model, feature_names):
    # Показываем, какие признаки модель чаще использовала.
    #
    # Это не прямые веса типа "bm25 = 0.2".
    # Это важность признаков внутри деревьев XGBoost.
    importances = model.feature_importances_

    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda item: item[1], reverse=True)

    print()
    print("Feature importances:")

    for feature_name, importance in pairs:
        print(feature_name, importance)


def train_single_global_model(dataset_path, answers_path, output_dir):
    # Один общий ranker на все категории.
    #
    # Плюс:
    #   проще поддерживать.
    #
    # Минус:
    #   признаки разных категорий смешиваются.
    dataset = load_json(dataset_path)
    answers_map = load_answers_map(answers_path)

    feature_names = collect_feature_names(dataset)

    X, y, qid, info = build_train_matrix(
        dataset=dataset,
        answers_map=answers_map,
        feature_names=feature_names,
    )

    print("Global model")
    print("Rows used:", info["rows_used"])
    print("Rows skipped without answer:", info["rows_skipped_without_answer"])
    print("Features count:", info["features_count"])
    print("Queries count:", info["queries_count"])
    print("Positive labels:", info["positive_labels"])
    print("Feature names:", feature_names)

    model = train_xgb_ranker(X, y, qid)

    save_ranker(
        model=model,
        feature_names=feature_names,
        output_dir=output_dir,
    )

    print_feature_importance(model, feature_names)

    return model, feature_names


def train_models_by_category(dataset_path, answers_path, output_root_dir):
    # Отдельная модель на каждую категорию.
    #
    # Это лучше, если категории реально разные:
    # мебель, еда, электроника и т.д.
    #
    # Потому что у каждой категории будут свои признаки и своя логика ранжирования.
    dataset = load_json(dataset_path)
    answers_map = load_answers_map(answers_path)

    feature_names_by_category = collect_feature_names_by_category(dataset)

    trained = {}

    for category, feature_names in feature_names_by_category.items():
        X, y, qid, info = build_train_matrix_for_category(
            dataset=dataset,
            answers_map=answers_map,
            category=category,
            feature_names=feature_names,
        )

        print()
        print("Category:", category)
        print("Rows used:", info["rows_used"])
        print("Rows skipped without answer:", info["rows_skipped_without_answer"])
        print("Features count:", info["features_count"])
        print("Queries count:", info["queries_count"])
        print("Positive labels:", info["positive_labels"])
        print("Feature names:", feature_names)

        # Если данных нет — просто пропускаем категорию.
        if info["rows_used"] == 0:
            print("Skip category because there are no rows")
            continue

        model = train_xgb_ranker(X, y, qid)

        # В имени папки убираем слэши, чтобы не сломать путь.
        category_dir_name = str(category).replace("/", "_").replace("\\", "_")

        output_dir = Path(output_root_dir) / category_dir_name

        save_ranker(
            model=model,
            feature_names=feature_names,
            output_dir=output_dir,
        )

        print_feature_importance(model, feature_names)

        trained[category] = {
            "model": model,
            "feature_names": feature_names,
        }

    return trained


def build_predict_matrix(rows, feature_names):
    # Эта функция пригодится потом на inference.
    #
    # На вход:
    #   rows — строки с features
    #
    # На выход:
    #   X — матрица для model.predict(X)
    X = []

    for row in rows:
        features = row.get("features", {})

        feature_vector = []

        for feature_name in feature_names:
            value = features.get(feature_name, 0.0)
            feature_vector.append(float(value))

        X.append(feature_vector)

    return np.array(X, dtype=np.float32)


if __name__ == "__main__":
    # Названия файлов пока условные.
    # Потом просто поменяешь на свои.
    dataset_path = "train_dataset.json"
    answers_path = "train_answers.json"

    # Основной вариант для вашей задачи:
    # обучаем отдельный ranker на каждую категорию.
    train_models_by_category(
        dataset_path=dataset_path,
        answers_path=answers_path,
        output_root_dir="trained_rankers_by_category",
    )

    # Если вдруг понадобится одна общая модель на всё,
    # можно раскомментировать:
    #
    # train_single_global_model(
    #     dataset_path=dataset_path,
    #     answers_path=answers_path,
    #     output_dir="trained_global_ranker",
    # )