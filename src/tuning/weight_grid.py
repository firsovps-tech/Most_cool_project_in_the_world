import itertools
import random

MAX_WEIGHT_COMBINATIONS = 10_000
RANDOM_SEED = 42

# У каждого веса ровно 9 кандидатных значений.
WEIGHT_GRID = {
    "w_exact_id": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "w_bm25": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "w_general_embedding": [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
    "w_detail_embedding": [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
    "w_price": [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "w_field_embedding": [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
    "w_product_has_field": [0.0, 0.01, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5],
}

DEFAULT_PARAMS = {
    "w_exact_id": 10.0,
    "w_bm25": 2.0,
    "w_general_embedding": 0.5,
    "w_detail_embedding": 0.5,
    "w_price": 0.0,
    "w_field_embedding": 0.5,
    "w_product_has_field": 0.05,
}

MANUAL_PROFILES = [
    DEFAULT_PARAMS,
    {**DEFAULT_PARAMS, "w_exact_id": 100.0},
    {**DEFAULT_PARAMS, "w_bm25": 4.0},
    {**DEFAULT_PARAMS, "w_general_embedding": 2.0, "w_detail_embedding": 2.0, "w_field_embedding": 2.0},
    {**DEFAULT_PARAMS, "w_bm25": 3.0, "w_general_embedding": 1.0},
    {**DEFAULT_PARAMS, "w_price": 1.0},
    {**DEFAULT_PARAMS, "w_product_has_field": 0.25},
]


def _key(params: dict) -> tuple:
    return tuple((name, params[name]) for name in WEIGHT_GRID)


def generate_weight_candidates(max_combinations: int = MAX_WEIGHT_COMBINATIONS, seed: int = RANDOM_SEED) -> list[dict]:
    """
    Не делаем полный 9^N перебор.
    Вместо этого:
    1. ручные профили;
    2. one-factor sweeps вокруг DEFAULT_PARAMS;
    3. случайные комбинации из 9 значений каждого веса.
    Итого не больше max_combinations.
    """
    candidates = []
    seen = set()

    def add(params):
        params = {name: params[name] for name in WEIGHT_GRID}
        k = _key(params)
        if k not in seen:
            candidates.append(params)
            seen.add(k)

    for profile in MANUAL_PROFILES:
        add(profile)

    # Локальные переборы: меняем один вес, остальные как в DEFAULT_PARAMS.
    for weight_name, values in WEIGHT_GRID.items():
        for value in values:
            params = dict(DEFAULT_PARAMS)
            params[weight_name] = value
            add(params)

    rng = random.Random(seed)
    names = list(WEIGHT_GRID.keys())

    while len(candidates) < max_combinations:
        params = {
            name: rng.choice(WEIGHT_GRID[name])
            for name in names
        }
        add(params)

        # Предохранитель от бесконечного цикла, если max больше полного пространства.
        if len(seen) >= 1:
            full_space = 1
            for values in WEIGHT_GRID.values():
                full_space *= len(values)
            if len(seen) >= min(max_combinations, full_space):
                break

    return candidates[:max_combinations]
