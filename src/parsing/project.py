import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.parsing.spelling_corrector import correct_spelling, looks_like_product_id


load_dotenv()

DATA_DIR = Path("data")
DOMAIN_CHARACTERISTICS_PATH = DATA_DIR / "domain_characteristics.json"


def load_json(path: str | Path) -> Any:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def field_value(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip()

    if not value or value == "-":
        return ""

    return value


def normalize_id(value: Any) -> str:
    """
    Приводит id товара к единому строковому виду.

    Это нужно, чтобы id из golden_standard.json и id из базы товаров
    сравнивались одинаково.

    Примеры:
    123      -> "123"
    "123"    -> "123"
    "123.0"  -> "123"
    " s1_25 " -> "s1_25"
    """
    value = field_value(value)

    if not value:
        return ""

    if re.fullmatch(r"\d+\.0", value):
        return value[:-2]

    return value


def load_domain_characteristics(
    path: str | Path = DOMAIN_CHARACTERISTICS_PATH,
) -> dict[str, list[str]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path}. Сначала загрузи товары через load_products_from_files(), "
            "чтобы построить data/domain_characteristics.json."
        )

    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError("domain_characteristics.json должен быть JSON-объектом")

    result: dict[str, list[str]] = {}

    for domain, fields in data.items():
        if isinstance(fields, list):
            clean_domain = field_value(domain)

            if clean_domain:
                result[clean_domain] = [field_value(field) for field in fields if field_value(field)]

    if not result:
        raise ValueError("domain_characteristics.json пустой или не содержит доменов")

    return result


def create_client() -> OpenAI:
    """
    Создаёт OpenAI-compatible клиент.

    Работает и с обычным OpenAI-compatible API, и с Yandex Cloud AI Studio,
    если в .env заданы:
    OPENAI_BASE_URL
    OPENAI_API_KEY
    OPENAI_MODEL
    OPENAI_PROJECT  # для Yandex Cloud, folder_id
    """
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    project = os.getenv("OPENAI_PROJECT")

    if not base_url:
        raise RuntimeError("Не задан OPENAI_BASE_URL в .env")

    if not api_key:
        raise RuntimeError("Не задан OPENAI_API_KEY в .env")

    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "api_key": api_key,
        "timeout": 30.0,
    }

    if project:
        kwargs["project"] = project
        kwargs["default_headers"] = {"x-folder-id": project}

    return OpenAI(**kwargs)


def get_model_name() -> str:
    model = os.getenv("OPENAI_MODEL")

    if not model:
        raise RuntimeError("Не задан OPENAI_MODEL в .env")

    return model


def extract_json(text: str) -> dict[str, Any]:
    """
    Достаёт JSON из ответа модели.

    Модель иногда возвращает:
    ```json
    {...}
    ```

    А json.loads умеет читать только чистый JSON.
    Поэтому сначала убираем markdown-обёртку.
    """
    text = field_value(text)

    if not text:
        raise ValueError("Модель вернула пустой ответ")

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError(f"Модель не вернула JSON: {text}")

        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError(f"Модель вернула JSON, но не объект: {parsed}")

    return parsed


def ask_llm_json(client: OpenAI, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
    )

    if hasattr(response, "choices"):
        content = response.choices[0].message.content
    elif isinstance(response, str):
        content = response
    elif isinstance(response, dict):
        if "choices" in response:
            content = response["choices"][0]["message"]["content"]
        elif "content" in response:
            content = response["content"]
        else:
            content = json.dumps(response, ensure_ascii=False)
    else:
        content = str(response)

    return extract_json(content)


def build_domain_prompt(domain_characteristics: dict[str, list[str]]) -> str:
    domain_hint = {
        domain: {"allowed_characteristics": fields}
        for domain, fields in domain_characteristics.items()
    }

    return f"""
Ты классифицируешь товарный запрос.

Нужно выбрать ровно один domain из списка ниже.

Домены и их характеристики:
{json.dumps(domain_hint, ensure_ascii=False, indent=2)}

Верни строго JSON:
{{
  "domain": "название_домена"
}}

Правила:
1. Верни только JSON.
2. Не используй markdown.
3. Не извлекай характеристики на этом шаге.
4. Если запрос с ошибками, выбери ближайший домен.
5. domain должен быть ровно одним из ключей списка.
""".strip()


def classify_domain(
    client: OpenAI,
    query_text: str,
    domain_characteristics: dict[str, list[str]],
) -> str:
    result = ask_llm_json(
        client=client,
        system_prompt=build_domain_prompt(domain_characteristics),
        user_prompt=query_text,
    )

    domain = field_value(result.get("domain"))

    if domain not in domain_characteristics:
        raise ValueError(f"Модель вернула неизвестный домен: {domain}")

    return domain


def build_parser_prompt(domain: str, fields: list[str]) -> str:
    template = {field: None for field in fields}

    return f"""
Ты парсишь товарный запрос в характеристики.

Домен уже выбран: {domain}

Разрешенные характеристики для этого домена:
{json.dumps(fields, ensure_ascii=False, indent=2)}

Верни строго JSON:
{{
  "characteristics": {json.dumps(template, ensure_ascii=False)}
}}

Правила:
1. Верни только JSON.
2. Не используй markdown.
3. Не добавляй характеристики вне списка.
4. Если значение отсутствует, ставь null.
5. Исправляй очевидные опечатки.
6. Не выдумывай данные, которых нет в запросе.
7. Сохраняй смысл запроса.
""".strip()


def parse_query_for_domain(
    client: OpenAI,
    query_text: str,
    domain: str,
    domain_characteristics: dict[str, list[str]],
) -> dict[str, Any]:
    fields = domain_characteristics[domain]

    return ask_llm_json(
        client=client,
        system_prompt=build_parser_prompt(domain, fields),
        user_prompt=query_text,
    )


def clean_characteristics(
    domain: str,
    raw_characteristics: dict[str, Any],
    domain_characteristics: dict[str, list[str]],
) -> dict[str, str]:
    fields = domain_characteristics[domain]
    allowed = set(fields)
    result = {field: "-" for field in fields}

    if not isinstance(raw_characteristics, dict):
        return result

    for key, value in raw_characteristics.items():
        key = field_value(key)

        if key not in allowed:
            continue

        value = field_value(value)

        if value:
            result[key] = value

    return result


def build_clean_query(characteristics: dict[str, str]) -> str:
    parts = []

    for value in characteristics.values():
        value = field_value(value)

        if value:
            parts.append(value)

    return " ".join(parts)


def lexical_fallback_parse(
    query_text: str,
    domain: str,
    domain_characteristics: dict[str, list[str]],
) -> dict[str, str]:
    fields = domain_characteristics[domain]
    result = {field: "-" for field in fields}

    if fields:
        result[fields[0]] = query_text

    return result


def choose_domain_without_extra_llm(
    client: OpenAI | None,
    query_text: str,
    known_domain: str | None,
    domain_characteristics: dict[str, list[str]],
    use_llm: bool,
) -> str:
    """
    Выбирает domain.

    Важная оптимизация:
    если домен в domain_characteristics.json только один, не надо делать
    лишний LLM-запрос classify_domain. Просто берём единственный домен.
    """
    if known_domain:
        domain = field_value(known_domain)

        if domain not in domain_characteristics:
            raise ValueError(f"Неизвестный domain из запроса: {domain}")

        return domain

    if len(domain_characteristics) == 1:
        return next(iter(domain_characteristics))

    if not use_llm:
        raise ValueError("Для сырого запроса без domain нужен LLM-парсер")

    if client is None:
        raise ValueError("LLM client is not initialized")

    return classify_domain(
        client=client,
        query_text=query_text,
        domain_characteristics=domain_characteristics,
    )


def normalize_one_query(
    query_id: str,
    query_text: str,
    known_domain: str | None = None,
    domain_characteristics_path: str | Path = DOMAIN_CHARACTERISTICS_PATH,
    use_llm: bool = True,
) -> dict[str, Any]:
    query_id = field_value(query_id)
    query_text = field_value(query_text)

    if not query_id:
        raise ValueError("query_id не должен быть пустым")

    if not query_text:
        raise ValueError(f"Пустой query_text у запроса {query_id}")

    original_query_text = query_text
    is_product_code = looks_like_product_id(query_text)

    # Артикулы/ID не отправляем в исправление орфографии,
    # чтобы LLM случайно не изменила код товара.
    # Но дальше Rita LLM-парсинг оставляем стандартным.
    if not is_product_code:
        query_text = correct_spelling(query_text, use_llm=use_llm)

    domain_characteristics = load_domain_characteristics(domain_characteristics_path)
    client = create_client() if use_llm else None

    domain = choose_domain_without_extra_llm(
        client=client,
        query_text=query_text,
        known_domain=known_domain,
        domain_characteristics=domain_characteristics,
        use_llm=use_llm,
    )

    if use_llm:
        try:
            parsed = parse_query_for_domain(
                client=client,
                query_text=query_text,
                domain=domain,
                domain_characteristics=domain_characteristics,
            )
            raw_characteristics = parsed.get("characteristics", {})
        except Exception as error:
            print(f"LLM parser fallback for {query_id}: {error}", flush=True)
            raw_characteristics = lexical_fallback_parse(
                query_text=query_text,
                domain=domain,
                domain_characteristics=domain_characteristics,
            )
    else:
        raw_characteristics = lexical_fallback_parse(
            query_text=query_text,
            domain=domain,
            domain_characteristics=domain_characteristics,
        )

    characteristics = clean_characteristics(
        domain=domain,
        raw_characteristics=raw_characteristics,
        domain_characteristics=domain_characteristics,
    )

    clean_query = build_clean_query(characteristics)

    return {
        "query_id": query_id,
        "id": original_query_text,
        "raw_query_text": original_query_text,
        "corrected_query_text": query_text,
        "query_text": clean_query,
        "domain": domain,
        "category": domain,
        "characteristics": characteristics,
    }


def get_query_id(item: dict[str, Any], index: int) -> str:
    return (
        field_value(item.get("query_id"))
        or field_value(item.get("id"))
        or f"q{index}"
    )


def get_query_text(item: dict[str, Any]) -> str:
    for key in ["query_text", "text", "description"]:
        value = field_value(item.get(key))

        if value:
            return value

    query_value = item.get("query")

    if isinstance(query_value, str):
        return field_value(query_value)

    if isinstance(query_value, dict):
        for key in ["query_text", "text", "description"]:
            value = field_value(query_value.get(key))

            if value:
                return value

        characteristics = query_value.get("characteristics")

        if isinstance(characteristics, dict):
            return build_clean_query({
                str(key): field_value(value)
                for key, value in characteristics.items()
            })

    return ""


def ids_from_value(value: Any) -> list[str]:
    """
    Универсально достаёт список id.

    Поддерживает:
    ["s1_1", "s1_2"]
    "s1_1"
    {"ids": [...]}
    [{"id": "s1_1"}, {"product_id": "s1_2"}]
    """
    result: list[str] = []

    if value is None:
        return result

    if isinstance(value, list):
        for item in value:
            result.extend(ids_from_value(item))

        return result

    if isinstance(value, tuple) or isinstance(value, set):
        for item in value:
            result.extend(ids_from_value(item))

        return result

    if isinstance(value, dict):
        nested_keys = [
            "id",
            "ID",
            "product_id",
            "productId",
            "productID",
            "Product ID",
            "ids",
            "product_ids",
            "relevant_ids",
            "ideal_product_ids",
        ]

        for key in nested_keys:
            if key in value:
                result.extend(ids_from_value(value.get(key)))

        return result

    value = normalize_id(value)

    if value:
        result.append(value)

    return result


def unique_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []

    for value in values:
        value = normalize_id(value)

        if not value:
            continue

        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


def get_relevant_ids(item: dict[str, Any]) -> list[str]:
    """
    Достаёт правильные id товаров из golden_standard.json.

    Главная исправленная ошибка:
    в твоём golden_standard.json поле называется ideal_product_ids,
    а старая версия кода его не читала. Поэтому в golden_answers.json
    попадали пустые relevant_ids, Positive labels было 0, и XGBoost
    не мог обучиться.
    """
    possible_keys = [
        "relevant_ids",
        "relevant_product_ids",
        "ideal_product_ids",
        "ideal_ids",
        "expected_ids",
        "expected_product_ids",
        "expected_relevant_product_ids",
        "answer_ids",
        "answers",
        "product_ids",
        "top_product_ids",
        "top_20_product_ids",
    ]

    collected: list[str] = []

    for key in possible_keys:
        if key in item:
            collected.extend(ids_from_value(item.get(key)))

    for nested_key in ["answer", "answers_data", "expected", "target"]:
        nested = item.get(nested_key)

        if isinstance(nested, dict):
            for key in possible_keys:
                if key in nested:
                    collected.extend(ids_from_value(nested.get(key)))

    return unique_keep_order(collected)


def normalize_golden_standard(
    input_path: str | Path = DATA_DIR / "golden_standard.json",
    output_queries_path: str | Path = DATA_DIR / "golden_queries_parsed.json",
    output_answers_path: str | Path = DATA_DIR / "golden_answers.json",
    domain_characteristics_path: str | Path = DOMAIN_CHARACTERISTICS_PATH,
    use_llm: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_data = load_json(input_path)

    if isinstance(raw_data, dict):
        raw_data = (
            raw_data.get("items")
            or raw_data.get("data")
            or raw_data.get("queries")
            or [raw_data]
        )

    if not isinstance(raw_data, list):
        raise ValueError("golden_standard.json должен быть списком или объектом с items/data/queries")

    queries = []
    answers = []

    for index, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            print(f"skip item {index}: item is not dict", flush=True)
            continue

        query_id = get_query_id(item, index)
        query_text = get_query_text(item)
        relevant_ids = get_relevant_ids(item)

        print(
            f"Parse golden query {index}/{len(raw_data)}: {query_id} | "
            f"{query_text} | relevant_ids={len(relevant_ids)}",
            flush=True,
        )

        if "domain" in item and isinstance(item.get("characteristics"), dict):
            domain_characteristics = load_domain_characteristics(domain_characteristics_path)
            domain = field_value(item.get("domain"))

            if not domain:
                if len(domain_characteristics) == 1:
                    domain = next(iter(domain_characteristics))
                else:
                    raise ValueError(f"У запроса {query_id} нет domain")

            if domain not in domain_characteristics:
                raise ValueError(f"У запроса {query_id} неизвестный domain: {domain}")

            characteristics = clean_characteristics(
                domain=domain,
                raw_characteristics=item.get("characteristics", {}),
                domain_characteristics=domain_characteristics,
            )

            normalized = {
                "query_id": query_id,
                "raw_query_text": field_value(item.get("raw_query_text")) or query_text,
                "query_text": build_clean_query(characteristics),
                "domain": domain,
                "characteristics": characteristics,
            }

        else:
            if not query_text:
                print(f"skip {query_id}: empty query text", flush=True)
                continue

            normalized = normalize_one_query(
                query_id=query_id,
                query_text=query_text,
                known_domain=field_value(item.get("domain")) or None,
                domain_characteristics_path=domain_characteristics_path,
                use_llm=use_llm,
            )

        queries.append(normalized)
        answers.append({
            "query_id": query_id,
            "relevant_ids": relevant_ids,
        })

    save_json(output_queries_path, queries)
    save_json(output_answers_path, answers)

    print(
        f"Saved parsed queries: {output_queries_path}, count={len(queries)}",
        flush=True,
    )
    print(
        f"Saved answers: {output_answers_path}, count={len(answers)}",
        flush=True,
    )

    return queries, answers


if __name__ == "__main__":
    normalize_golden_standard(
        input_path=DATA_DIR / "golden_standard.json",
        output_queries_path=DATA_DIR / "golden_queries_parsed.json",
        output_answers_path=DATA_DIR / "golden_answers.json",
        use_llm=True,
    )
