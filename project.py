import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


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


def load_domain_characteristics(path: str | Path = DOMAIN_CHARACTERISTICS_PATH) -> dict[str, list[str]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path}. Сначала загрузи товары через load_products_from_files(), "
            "чтобы построить data/domain_characteristics.json."
        )

    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError("domain_characteristics.json должен быть JSON-объектом")

    result = {}

    for domain, fields in data.items():
        if isinstance(fields, list):
            result[str(domain)] = [str(field) for field in fields]

    return result


def create_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:4010/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "unused"),
    )


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", "opencode/deepseek-v4-flash-free")


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError(f"Модель не вернула JSON: {text}")

        return json.loads(match.group(0))


def ask_llm_json(client: OpenAI, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Пустой ответ от модели")

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
    result = {}
    fields = domain_characteristics[domain]
    allowed = set(fields)

    for field in fields:
        result[field] = "-"

    for key, value in raw_characteristics.items():
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


def normalize_one_query(
    query_id: str,
    query_text: str,
    known_domain: str | None = None,
    domain_characteristics_path: str | Path = DOMAIN_CHARACTERISTICS_PATH,
    use_llm: bool = True,
) -> dict[str, Any]:
    domain_characteristics = load_domain_characteristics(domain_characteristics_path)
    client = create_client() if use_llm else None

    if known_domain:
        domain = known_domain

        if domain not in domain_characteristics:
            raise ValueError(f"Неизвестный domain из запроса: {domain}")
    else:
        if not use_llm:
            raise ValueError("Для сырого запроса без domain нужен LLM-парсер")

        domain = classify_domain(
            client=client,
            query_text=query_text,
            domain_characteristics=domain_characteristics,
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
            print(f"LLM parser fallback for {query_id}: {error}")
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
        "raw_query_text": query_text,
        "query_text": clean_query,
        "domain": domain,
        "characteristics": characteristics,
    }


def get_query_id(item: dict[str, Any], index: int) -> str:
    return field_value(item.get("query_id")) or field_value(item.get("id")) or f"q{index}"


def get_query_text(item: dict[str, Any]) -> str:
    for key in ["query_text", "query", "text", "description"]:
        value = field_value(item.get(key))

        if value:
            return value

    nested_query = item.get("query")

    if isinstance(nested_query, dict):
        for key in ["query_text", "text", "description"]:
            value = field_value(nested_query.get(key))

            if value:
                return value

    return ""


def get_relevant_ids(item: dict[str, Any]) -> list[str]:
    possible_keys = [
        "relevant_ids",
        "relevant_product_ids",
        "answer_ids",
        "answers",
        "product_ids",
        "top_product_ids",
        "top_20_product_ids",
    ]

    for key in possible_keys:
        value = item.get(key)

        if isinstance(value, list):
            return [field_value(product_id) for product_id in value if field_value(product_id)]

    answer = item.get("answer")

    if isinstance(answer, dict):
        for key in possible_keys:
            value = answer.get(key)

            if isinstance(value, list):
                return [field_value(product_id) for product_id in value if field_value(product_id)]

    return []


def normalize_golden_standard(
    input_path: str | Path = DATA_DIR / "golden_standard.json",
    output_queries_path: str | Path = DATA_DIR / "golden_queries_parsed.json",
    output_answers_path: str | Path = DATA_DIR / "golden_answers.json",
    domain_characteristics_path: str | Path = DOMAIN_CHARACTERISTICS_PATH,
    use_llm: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_data = load_json(input_path)

    if isinstance(raw_data, dict):
        raw_data = raw_data.get("items") or raw_data.get("data") or raw_data.get("queries") or [raw_data]

    queries = []
    answers = []

    for index, item in enumerate(raw_data, start=1):
        query_id = get_query_id(item, index)
        relevant_ids = get_relevant_ids(item)

        if "domain" in item and isinstance(item.get("characteristics"), dict):
            domain_characteristics = load_domain_characteristics(domain_characteristics_path)
            domain = field_value(item.get("domain"))

            characteristics = clean_characteristics(
                domain=domain,
                raw_characteristics=item.get("characteristics", {}),
                domain_characteristics=domain_characteristics,
            )

            normalized = {
                "query_id": query_id,
                "raw_query_text": field_value(item.get("raw_query_text")) or field_value(item.get("query_text")),
                "query_text": build_clean_query(characteristics),
                "domain": domain,
                "characteristics": characteristics,
            }
        else:
            query_text = get_query_text(item)

            if not query_text:
                print(f"skip {query_id}: empty query text")
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

    return queries, answers


if __name__ == "__main__":
    normalize_golden_standard(
        input_path=DATA_DIR / "golden_standard.json",
        output_queries_path=DATA_DIR / "golden_queries_parsed.json",
        output_answers_path=DATA_DIR / "golden_answers.json",
        use_llm=True,
    )
