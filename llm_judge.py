import os
import json
import time
from typing import Tuple
# pyrefly: ignore [missing-import]
from openai import OpenAI

import config

client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)

def load_cache() -> dict:
    if os.path.exists(config.CACHE_FILE):
        try:
            with open(config.CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_cache(cache: dict):
    with open(config.CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def get_llm_judgment(query_text: str, product_name: str, product_desc: str, retries: int = 3) -> Tuple[int | None, str]:
    prompt = f"""
Запрос: {query_text}
Товар: {product_name}
Описание: {product_desc}

Оцени релевантность товара этому запросу по шкале от 0 до 3, строго используя следующие критерии:
3 - Отличное совпадение: товар отличается не более чем на одну характеристику (модель, цвет, размер и т.д.).
2 - Хорошее совпадение: товар отличается не более чем на две характеристики.
1 - Частичное совпадение: товары имеют общее функциональное применение или принадлежат к смежным категориям (например, диван и кровать — на них можно спать; стул и табурет — на них можно сидеть), даже если базовая сущность и характеристики разные.
0 - Мусор: товар совершенно нерелевантен запросу.

Сначала проанализируй совпадение характеристик и напиши обоснование в поле 'reason', а затем поставь итоговую оценку в поле 'score'.
Отвечай СТРОГО в формате JSON с двумя полями: 'reason' (строка) и 'score' (число).
"""
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Ты — эксперт по качеству поиска мебели. Твоя задача — оценить релевантность товара запросу. Отвечай только в формате JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            raw_content = response.choices[0].message.content
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:-3].strip()
                
            data = json.loads(raw_content)
            
            if 'score' not in data or 'reason' not in data:
                print(f"  [ВНИМАНИЕ] Неожиданный формат JSON от модели, пытаюсь адаптировать:\n{raw_content}")
            
            score = int(data.get('score', data.get('relevance', data.get('оценка', 0))))
            reason = str(data.get('reason', data.get('justification', data.get('причина', 'Без причины'))))
            
            if score not in [0, 1, 2, 3]:
                score = max(0, min(3, score))
                
            return score, reason

        except Exception as e:
            print(f"  [Ошибка API/JSON, попытка {i+1}/{retries}]: {e}")
            if i < retries - 1:
                time.sleep(5)
                continue
            return None, f"Ошибка: {e}"

def is_common_word_article(art: str) -> bool:
    # Если артикул состоит только из букв и его длина <= 5 символов (например, "стол"),
    # считаем его обычным словом, а не специализированным кодом артикула.
    return art.isalpha() and len(art) <= 5

def evaluate_pair(
    query_id: str, 
    query_text: str, 
    product_id: str, 
    product_name: str, 
    product_desc: str, 
    cache: dict, 
    product_article: str = None, 
    article_to_id: dict = None
) -> Tuple[int | None, str]:
    cache_key = f"{query_id}:{product_id}"
    if cache_key in cache:
        return cache[cache_key]['score'], cache[cache_key]['reason']
    
    # 1. Проверяем, не является ли запрос поиском по конкретному артикулу
    if article_to_id:
        query_clean = str(query_text).strip().upper()
        matched_art = None
        
        # Проверяем точное совпадение запроса с артикулом
        if query_clean in article_to_id:
            matched_art = query_clean
        else:
            # Проверяем, содержит ли запрос артикул как отдельное слово/токен
            for w in query_clean.split():
                if w in article_to_id:
                    # Если артикул — это простое слово (например, "стол"), 
                    # мы НЕ триггерим обход, если весь запрос не равен этому слову
                    if is_common_word_article(w):
                        continue
                    matched_art = w
                    break
        
        if matched_art:
            target_pid = article_to_id[matched_art]
            if str(product_id) == str(target_pid):
                score = 3
                reason = f"Товар полностью соответствует запросу по артикулу: {matched_art}."
            else:
                p_art = str(product_article).strip() if product_article else "отсутствует"
                reason = f"Запрос содержит конкретный артикул {matched_art}, но данный товар имеет другой артикул ({p_art})."
                score = 0
            
            # Записываем в кэш
            cache[cache_key] = {'score': score, 'reason': reason}
            save_cache(cache)
            print(f"  [АРТИКУЛ BYPASS] {product_id} | Оценка: {score} | Обоснование: {reason}")
            return score, reason

    print(f"  Запрос к LLM для {product_id}...")
    score, reason = get_llm_judgment(query_text, product_name, product_desc)
    
    if score is not None:
        cache[cache_key] = {'score': score, 'reason': reason}
        save_cache(cache)
        
    time.sleep(1)
    return score, reason

