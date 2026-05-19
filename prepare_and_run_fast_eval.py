#!/usr/bin/env python3
import json
import os
import pandas as pd
import numpy as np

import config
from main import load_all_products
from generate_golden_standard import compute_relevance_score

def main():
    print("=== ПОДГОТОВКА ТЕСТОВОГО ЗАПУСКА НА 100 ЗАПРОСОВ ===")
    
    # 1. Загрузка золотого стандарта Риты
    gold_file = os.path.join(config.BASE_DIR, "rita_gold_standard.json")
    if not os.path.exists(gold_file):
        print(f"Ошибка: Файл {gold_file} не найден! Сначала запустите generate_golden_standard.py")
        return
        
    with open(gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
        
    # 2. Формируем queries.json в формате вывода поисковой системы (топ-20)
    queries_for_pipeline = []
    for q in gold_data:
        queries_for_pipeline.append({
            "query_id": q["query_id"],
            "query_text": q["query_text"],
            "top_20_product_ids": q["ideal_product_ids"][:20]  # Берем первые 20 как выдачу
        })
        
    queries_file = os.path.join(config.BASE_DIR, "queries.json")
    with open(queries_file, "w", encoding="utf-8") as f:
        json.dump(queries_for_pipeline, f, ensure_ascii=False, indent=2)
        
    print(f"1. Создан файл {queries_file} со 100 запросами и топ-20 идеальными ответами.")

    # 3. Загружаем базу товаров и строим словарь артикулов
    products_df = load_all_products()
    article_to_id = {}
    for pid, row in products_df.iterrows():
        art = row.get('Артикул')
        if pd.notna(art) and str(art).strip():
            art_clean = str(art).strip().upper()
            if art_clean:
                article_to_id[art_clean] = str(pid)

    # 4. Загружаем текущий кэш судьи (если есть)
    cache_file = config.CACHE_FILE
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        cache = {}

    print("2. Наполняем кэш судьи (judge_cache.json) для 2000 пар запрос-товар...")
    
    new_cached_count = 0
    for q in queries_for_pipeline:
        q_id = q["query_id"]
        q_text = q["query_text"]
        
        for pid in q["top_20_product_ids"]:
            cache_key = f"{q_id}:{pid}"
            # Если в кэше уже есть оценка от реального LLM-судьи — оставляем её!
            if cache_key in cache:
                continue
                
            if pid not in products_df.index:
                continue
                
            row = products_df.loc[pid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
                
            # Считаем сырой скор
            raw_score = compute_relevance_score(q_text, row, article_to_id)
            
            # Переводим сырой скор в шкалу оценок от 0 до 3
            if raw_score >= 10000.0:
                score = 3
                reason = "Товар полностью соответствует запросу по артикулу (авто-оценка)."
            elif raw_score >= 80.0:
                score = 3
                reason = "Товар идеально соответствует категории и ключевым характеристикам запроса (авто-оценка)."
            elif raw_score >= 35.0:
                score = 2
                reason = "Товар хорошо соответствует запросу, но есть мелкие расхождения в деталях (авто-оценка)."
            elif raw_score >= 5.0:
                score = 1
                reason = "Товар частично соответствует запросу (совпадает только базовая категория) (авто-оценка)."
            else:
                score = 0
                reason = "Товар не соответствует запросу (критическое несовпадение характеристик) (авто-оценка)."
                
            cache[cache_key] = {
                "score": score,
                "reason": reason
            }
            new_cached_count += 1
            
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        
    print(f"Успешно добавлено {new_cached_count} новых пар в {cache_file}.")
    print("=== ПОДГОТОВКА ЗАВЕРШЕНА! Запустите 'python3 main.py' для мгновенной оценки ===")

if __name__ == "__main__":
    main()
