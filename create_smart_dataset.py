#!/usr/bin/env python3
import json
import os
import time
import pandas as pd
from openai import OpenAI

import config
from main import load_all_products
from generate_golden_standard import compute_relevance_score

# Принудительно используем рабочую умную модель Kimi K2.6
SMART_MODEL = "kimi-k2p6"

# 15 разнообразных запросов для ручной проверки
QUERIES = [
    {"id": "sq1", "text": "белая кровать 160"},
    {"id": "sq2", "text": "угловой диван рогожка"},
    {"id": "sq3", "text": "шкаф купе с зеркалом"},
    {"id": "sq4", "text": "стол обеденный раздвижной"},
    {"id": "sq5", "text": "ЯР-PL-B16"},  # Проверка точного артикула
    {"id": "sq6", "text": "стул деревянный кухонный"},
    {"id": "sq7", "text": "компьютерное кресло черное"},
    {"id": "sq8", "text": "тумба прикроватная дуб сонома"},
    {"id": "sq9", "text": "подвесная тумба под раковину"},
    {"id": "sq10", "text": "детская кровать белая"},
    {"id": "sq11", "text": "стеллаж лофт металл"},
    {"id": "sq12", "text": "кресло реклайнер графит"},
    {"id": "sq13", "text": "Christmas tree slim"}, # Проверка английских товаров
    {"id": "sq14", "text": "пуфик велюровый круглый"},
    {"id": "sq15", "text": "банкетка в прихожую обувница"} # Сложный составной запрос
]

CACHE_FILE = os.path.join(config.BASE_DIR, "smart_gold_standard_cache.json")
OUTPUT_FILE = os.path.join(config.BASE_DIR, "smart_gold_standard.json")

client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL
)

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache_data):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

def main():
    print("=== ЗАПУСК УМНОЙ ГЕНЕРАЦИИ ЗОЛОТОГО СТАНДАРТА ===")
    start_time = time.time()
    
    products_df = load_all_products()
    
    # Строим словарь артикулов для быстрого отбора кандидатов
    article_to_id = {}
    for pid, row in products_df.iterrows():
        art = row.get('Артикул')
        if pd.notna(art) and str(art).strip():
            article_to_id[str(art).strip().upper()] = str(pid)

    # Загружаем кэш промежуточных результатов
    cache = load_cache()
    gold_standard = []
    
    # Восстанавливаем уже готовые запросы из кэша
    for q in QUERIES:
        q_id = q["id"]
        if q_id in cache:
            gold_standard.append({
                "query_id": q_id,
                "query_text": q["text"],
                "ideal_product_ids": cache[q_id]
            })

    total_queries = len(QUERIES)
    already_done = len(gold_standard)
    
    print(f"Модель: {SMART_MODEL}")
    print(f"Всего запросов: {total_queries} (Уже готово из кэша: {already_done})")
    print("-" * 60)
    
    for idx, q in enumerate(QUERIES):
        q_id = q["id"]
        q_text = q["text"]
        
        # Если запрос уже есть в кэше — пропускаем вызов API
        if q_id in cache:
            continue
            
        # 1. Отбираем топ-80 кандидатов через эвристику.
        # Это обеспечивает максимальную полноту при стабильной скорости Kimi
        scores = []
        for pid, row in products_df.iterrows():
            score = compute_relevance_score(q_text, row, article_to_id)
            scores.append((pid, score))
            
        scores.sort(key=lambda x: (-x[1], x[0]))
        top_candidates_ids = [pid for pid, score in scores[:80]]
        
        # 2. Формируем промпт для модели
        prompt = f"Ты — эксперт-оценщик мебели. Задача: выбери и отсортируй предложенные товары по релевантности для поискового запроса: '{q_text}'.\n\nСписок кандидатов:\n"
        
        for pid in top_candidates_ids:
            row = products_df.loc[pid]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            name = row.get('Name', '')
            art = row.get('Артикул', '')
            color = row.get('Цвет', '')
            mat = row.get('Материал', '')
            dims = row.get('Габариты', '')
            prompt += f"- ID: {pid} | Назв: {name} | Арт: {art} | Цвет: {color} | Габариты: {dims} | Мат: {mat}\n"
            
        prompt += "\nОтветь СТРОГО в формате JSON: списком из 20 лучших ID товаров, отсортированных от самого идеального к наименее подходящему.\nПример: [\"s1_12\", \"s5_16\"]"

        # 3. Вызов модели с повторными попытками при ошибках
        ideal_ids = []
        retries = 3
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=SMART_MODEL,
                    messages=[
                        {"role": "system", "content": "Ты эксперт-асессор. Возвращай только валидный JSON-список строк (ID товаров). Без форматирования markdown. Никакого текста, только JSON массив."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                
                content = response.choices[0].message.content.strip()
                # Очистка от markdown
                if content.startswith('```'):
                    content = content.split('\n', 1)[1]
                    if content.endswith('```'):
                        content = content.rsplit('\n', 1)[0]
                elif content.startswith('['):
                    pass
                    
                ideal_ids = json.loads(content)
                if ideal_ids:
                    break # Успешно получили данные
            except Exception as e:
                if attempt == retries - 1:
                    print(f"\n[Ошибка] Не удалось обработать запрос '{q_text}' после {retries} попыток: {e}")
                    ideal_ids = top_candidates_ids[:20]
                else:
                    time.sleep(2)
            
        # Записываем результат в кэш
        cache[q_id] = ideal_ids[:20]
        save_cache(cache)
        
        gold_standard.append({
            "query_id": q_id,
            "query_text": q_text,
            "ideal_product_ids": ideal_ids[:20]
        })
        
        # 4. Расчёт прогресса и времени
        processed_count = len(gold_standard)
        elapsed = time.time() - start_time
        percent = (processed_count / total_queries) * 100
        
        newly_processed = processed_count - already_done
        if newly_processed > 0:
            avg_time = elapsed / newly_processed
            remaining = avg_time * (total_queries - processed_count)
        else:
            remaining = 0
            
        bar_length = 20
        filled = int(bar_length * processed_count // total_queries)
        bar = '█' * filled + '-' * (bar_length - filled)
        
        print(f"\r[{bar}] {percent:5.1f}% | Прошло: {format_time(elapsed)} | Осталось: {format_time(remaining)} | Запрос: {q_text:<30}", end="", flush=True)
        
    # Сохраняем финальный умный датасет
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(gold_standard, f, ensure_ascii=False, indent=2)
        
    # Удаляем временный кэш после успешного завершения всего процесса
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        
    print(f"\n\n=== УСПЕХ! ===")
    print(f"Умный датасет (15 запросов) сохранен в {OUTPUT_FILE}")
    print(f"Общее время работы: {format_time(time.time() - start_time)}")

if __name__ == "__main__":
    main()
