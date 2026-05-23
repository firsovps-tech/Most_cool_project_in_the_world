import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CACHE_PATH = Path("data") / "spelling_correction_cache.json"


def field_value(value) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if not value or value == "-":
        return ""
    return value


def looks_like_product_id(query: str) -> bool:
    text = field_value(query)
    if not text:
        return False
    has_digit = any(ch.isdigit() for ch in text)
    has_letter = any(ch.isalpha() for ch in text)
    has_code_symbol = "-" in text or "_" in text or "/" in text
    return has_digit and has_letter and has_code_symbol


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)


def _create_client() -> OpenAI:
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    project = os.getenv("OPENAI_PROJECT")

    if not base_url:
        raise RuntimeError("Не задан OPENAI_BASE_URL в .env")
    if not api_key:
        raise RuntimeError("Не задан OPENAI_API_KEY в .env")

    kwargs = {
        "base_url": base_url,
        "api_key": api_key,
        "timeout": 30.0,
    }
    if project:
        # Оставляем Rita-подход через OPENAI_PROJECT и дополнительно передаём x-folder-id,
        # чтобы Yandex Cloud OpenAI-compatible API видел folder_id.
        kwargs["project"] = project
        kwargs["default_headers"] = {"x-folder-id": project}

    return OpenAI(**kwargs)


def _get_model_name() -> str:
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("Не задан OPENAI_MODEL в .env")
    return model


def correct_spelling(query_text: str, use_llm: bool = True, retries: int = 2) -> str:
    query_text = field_value(query_text)

    if not query_text:
        return ""

    # Артикулы и product id не отправляем в LLM, чтобы модель не сломала код товара.
    if looks_like_product_id(query_text):
        return query_text

    if not use_llm:
        return query_text

    cache = _load_cache()
    if query_text in cache:
        return field_value(cache[query_text]) or query_text

    try:
        client = _create_client()
        model = _get_model_name()
    except Exception as error:
        print(f"Spelling correction skipped: {error}", flush=True)
        cache[query_text] = query_text
        _save_cache(cache)
        return query_text

    prompt = f"""
Исправь только орфографические ошибки и опечатки в поисковом запросе.
Не добавляй новые характеристики.
Не удаляй важные характеристики.
Не меняй смысл.
Не объясняй ответ.
Верни только исправленный запрос одной строкой.

Запрос: {query_text}
""".strip()

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты исправляешь только орфографию и опечатки в русских товарных поисковых запросах.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            corrected = field_value(response.choices[0].message.content).strip('"')
            corrected = corrected or query_text
            cache[query_text] = corrected
            _save_cache(cache)
            return corrected
        except Exception as error:
            print(f"Spelling correction error {attempt + 1}/{retries}: {error}", flush=True)
            if attempt + 1 < retries:
                time.sleep(2)

    cache[query_text] = query_text
    _save_cache(cache)
    return query_text
