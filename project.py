import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


DOMAINS = {
  "furniture": {
    "description": "Мебель: стулья, столы, шкафы, диваны, кровати и другие предметы мебели.",
    "characteristics": {
      "product_type": "Тип товара: стул, стол, шкаф, диван и т.д.",
      "brand": "Бренд или производитель.",
      "color": "Цвет товара.",
      "material": "Материал товара.",
      "size": "Размер, габариты, высота, ширина."
    }
  },
  "food": {
    "description": "Еда и напитки: кофе, чай, крупы, сладости, продукты питания.",
    "characteristics": {
      "product_type": "Тип продукта: кофе, чай, шоколад, крупа и т.д.",
      "brand": "Бренд или производитель.",
      "weight": "Вес или объем.",
      "taste": "Вкус, добавка, аромат.",
      "package_type": "Формат или упаковка: молотый, зерновой, растворимый, пакетированный."
    }
  },
  "electronics": {
    "description": "Электроника: телефоны, ноутбуки, планшеты, наушники, техника.",
    "characteristics": {
      "product_type": "Тип товара: телефон, ноутбук, планшет, наушники.",
      "brand": "Бренд или производитель.",
      "model": "Модель товара.",
      "memory": "Память, объем накопителя, RAM.",
      "color": "Цвет товара."
    }
  }
}


CHARACTERISTIC_ORDER = {
  "furniture": ["product_type", "brand", "color", "material", "size"],
  "food": ["product_type", "brand", "weight", "taste", "package_type"],
  "electronics": ["product_type", "brand", "model", "memory", "color"]
}


def create_client() -> OpenAI:
  return OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:4010/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "unused")
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


def ask_llm_json(
  client: OpenAI,
  system_prompt: str,
  user_prompt: str
) -> dict[str, Any]:
  response = client.chat.completions.create(
    model=get_model_name(),
    messages=[
      {
        "role": "system",
        "content": system_prompt
      },
      {
        "role": "user",
        "content": user_prompt
      }
    ]
  )

  content = response.choices[0].message.content

  if not content:
    raise ValueError("Пустой ответ от модели")

  return extract_json(content)


def load_queries(path: str) -> list[dict[str, str]]:
  with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

  if isinstance(data, dict):
    data = [data]

  queries = []

  for i, item in enumerate(data, start=1):
    query_id = item.get("query_id", f"q{i}")
    query_text = item.get("query_text") or item.get("query")

    if not query_text:
      continue

    queries.append({
      "query_id": query_id,
      "query_text": query_text
    })

  return queries


def build_domain_classification_prompt() -> str:
  domain_descriptions = {
    domain_name: domain_data["description"]
    for domain_name, domain_data in DOMAINS.items()
  }

  return f"""
Ты классифицируешь пользовательский товарный запрос.

Нужно выбрать ровно один домен из списка:

{json.dumps(domain_descriptions, ensure_ascii=False, indent=2)}

Верни строго JSON такого вида:

{{
  "domain": "furniture"
}}

Допустимые значения domain:
{json.dumps(list(DOMAINS.keys()), ensure_ascii=False)}

Правила:
1. Верни только JSON.
2. Не используй markdown.
3. Не исправляй запрос на этом этапе.
4. Не извлекай характеристики на этом этапе.
5. Если запрос с ошибками, выбери ближайший подходящий домен.
""".strip()


def classify_domain(client: OpenAI, query_text: str) -> str:
  result = ask_llm_json(
    client=client,
    system_prompt=build_domain_classification_prompt(),
    user_prompt=query_text
  )

  domain = result.get("domain")

  if domain not in DOMAINS:
    raise ValueError(f"Модель вернула неизвестный домен: {domain}")

  return domain


def build_domain_parser_prompt(domain: str) -> str:
  domain_data = DOMAINS[domain]
  allowed_characteristics = domain_data["characteristics"]

  example_characteristics = {
    key: None
    for key in allowed_characteristics.keys()
  }

  return f"""
Ты нормализуешь товарный запрос.

Домен уже определен: {domain}

Описание домена:
{domain_data["description"]}

Разрешенные характеристики для этого домена:
{json.dumps(allowed_characteristics, ensure_ascii=False, indent=2)}

Верни строго JSON такого вида:

{{
  "characteristics": {json.dumps(example_characteristics, ensure_ascii=False)}
}}

Твоя задача:
1. Исправить орфографические ошибки.
2. Удалить мусорные слова.
3. Оставить только важные товарные признаки.
4. Распределить информацию по характеристикам.
5. Не выдумывать данные, которых нет в запросе.
6. Если значение характеристики отсутствует, ставь null.
7. Не добавляй характеристики вне разрешенного списка.
8. Бренды и модели пиши нормально, если исправление очевидно.

Примеры исправлений:
- "чернй" → "черный"
- "метл" → "металл"
- "лаваза" → "Lavazza"
- "айфон" → "iPhone"
- "икеа" → "IKEA"

Важно:
Верни только JSON.
Не используй markdown.
Не добавляй пояснения.
""".strip()


def parse_query_for_domain(
  client: OpenAI,
  query_text: str,
  domain: str
) -> dict[str, Any]:
  return ask_llm_json(
    client=client,
    system_prompt=build_domain_parser_prompt(domain),
    user_prompt=query_text
  )


def clean_characteristics(
  domain: str,
  characteristics: dict[str, Any]
) -> dict[str, str]:
  allowed = set(DOMAINS[domain]["characteristics"].keys())
  result = {}

  for key, value in characteristics.items():
    if key not in allowed:
      continue

    if value is None:
      continue

    if not isinstance(value, str):
      continue

    value = value.strip()

    if not value:
      continue

    result[key] = value

  return result


def build_clean_query(domain: str, characteristics: dict[str, str]) -> str:
  order = CHARACTERISTIC_ORDER.get(domain, list(characteristics.keys()))

  parts = []

  for key in order:
    value = characteristics.get(key)

    if value:
      parts.append(value)

  return " ".join(parts)


def normalize_one_query(
  client: OpenAI,
  query_id: str,
  query_text: str
) -> dict[str, Any]:
  domain = classify_domain(client, query_text)

  parsed = parse_query_for_domain(
    client=client,
    query_text=query_text,
    domain=domain
  )

  characteristics = clean_characteristics(
    domain=domain,
    characteristics=parsed.get("characteristics", {})
  )

  clean_query = build_clean_query(
    domain=domain,
    characteristics=characteristics
  )

  return {
    "query_id": query_id,
    "query_text": clean_query,
    "domain": domain,
    "characteristics": characteristics
  }


def normalize_queries(input_path: str, output_path: str) -> None:
  client = create_client()
  queries = load_queries(input_path)

  results = []

  for item in queries:
    query_id = item["query_id"]
    query_text = item["query_text"]

    try:
      result = normalize_one_query(
        client=client,
        query_id=query_id,
        query_text=query_text
      )

      results.append(result)

    except Exception as error:
      print(f"Ошибка при обработке {query_id}: {error}")

  with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
  normalize_queries(
    input_path="input.json",
    output_path="output.json"
  )