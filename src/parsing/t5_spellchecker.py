import re
from functools import lru_cache

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


MODEL_NAME = "ai-forever/T5-base-spellchecker"


def normalize_yo(text: str) -> str:
    if text is None:
        return ""
    return str(text).replace("ё", "е").replace("Ё", "Е")


def looks_like_article_or_id_token(token: str) -> bool:
    """
    Артикулы и id товара не исправляем через T5,
    чтобы модель случайно не изменила код.

    Примеры:
    ЯР-PL-B16
    ABC-999
    s1_110
    """
    token = str(token).strip()

    if not token:
        return False

    has_digit = any(ch.isdigit() for ch in token)
    has_letter = any(ch.isalpha() for ch in token)
    has_special = any(ch in token for ch in ["-", "_", "/", "\\"])

    return has_digit and has_letter and has_special


def protect_article_tokens(original_text: str, corrected_text: str) -> str:
    """
    Возвращает артикулы/id из исходного запроса обратно в результат,
    если T5 случайно их удалила или изменила.
    """
    original_tokens = str(original_text).split()
    corrected = str(corrected_text).strip()

    protected_tokens = [
        token for token in original_tokens
        if looks_like_article_or_id_token(token)
    ]

    if not protected_tokens:
        return corrected

    result = corrected

    for token in protected_tokens:
        if token not in result:
            result = token + " " + result

    return result.strip()


@lru_cache(maxsize=1)
def load_t5_spellchecker():
    """
    Загружаем модель один раз и кешируем.
    При первом вызове будет долго, потом быстрее.
    """
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device


def correct_sentence_t5(text: str) -> str:
    """
    Исправляет орфографию в русском тексте через T5 spellchecker.
    """
    text = normalize_yo(text).strip()

    if not text:
        return ""

    tokenizer, model, device = load_t5_spellchecker()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=5,
            early_stopping=True,
        )

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    corrected = normalize_yo(corrected).strip()

    return corrected or text


def correct_query_spelling_t5(query_text: str) -> str:
    """
    Главная функция для проекта.

    Делает исправление запроса перед LLM-парсингом:
    1. заменяет ё на е;
    2. если запрос состоит только из артикула/id, не исправляет его;
    3. исправляет орфографию через T5;
    4. возвращает артикулы/id обратно, если модель их случайно изменила.
    """
    original = normalize_yo(query_text).strip()

    if not original:
        return ""

    # Если весь запрос — один артикул/id, не трогаем вообще.
    if len(original.split()) == 1 and looks_like_article_or_id_token(original):
        return original

    corrected = correct_sentence_t5(original)
    corrected = protect_article_tokens(original, corrected)

    if corrected != original:
        print(f"[T5 SPELLCHECK] {original} -> {corrected}", flush=True)

    return corrected
