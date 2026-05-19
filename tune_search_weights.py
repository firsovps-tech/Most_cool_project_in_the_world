#!/usr/bin/env python3
import json
import os
import time
import pandas as pd
import numpy as np

import config
from main import load_all_products
from metrics import RankingMetrics

GOLD_STANDARD_FILE = os.path.join(config.BASE_DIR, "overnight_gold_standard.json")

def search_products_tuned(query_text, products_df, article_to_id, weights, k=20):
    query_clean = str(query_text).strip().upper()
    
    # 1. Точный артикул
    matched_art = None
    if query_clean in article_to_id:
        matched_art = query_clean
    else:
        for w in query_clean.split():
            if w in article_to_id:
                matched_art = w
                break
                
    if matched_art:
        target_pid = article_to_id[matched_art]
        if target_pid in products_df.index:
            return [target_pid]
        return []
        
    # 2. Поиск с тюнингованными весами
    query_words = [w.lower() for w in query_clean.split() if len(w) >= 2]
    if not query_words:
        return list(products_df.index[:k])
        
    scores = {}
    for pid, row in products_df.iterrows():
        score = 0.0
        
        p_name = str(row.get('Name', '')).lower()
        p_desc = str(row.get('Description', '')).lower()
        p_color = str(row.get('Цвет', '')).lower()
        p_material = str(row.get('Материал', '')).lower()
        
        for word in query_words:
            # Размерные цифры
            if word.isdigit() and len(word) >= 2:
                if word in p_name or word in p_desc:
                    score += weights['w_size']
            else:
                # Поиск по имени
                if word in p_name:
                    score += weights['w_name']
                # Поиск по описанию
                if word in p_desc:
                    score += weights['w_desc']
                # Поиск по цвету
                if word in p_color:
                    score += weights['w_color']
                # Поиск по материалу
                if word in p_material:
                    score += weights['w_material']
                    
        if score > 0:
            scores[pid] = score
            
    sorted_pids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in sorted_pids[:k]]

def evaluate_parameters(products_df, article_to_id, gold_data, weights, evaluator):
    ndcgs = []
    
    for q in gold_data:
        q_text = q["query_text"]
        ideal_ids = q["ideal_product_ids"]
        
        # Создаем словарь идеальных оценок релевантности:
        # Топ-5: 3 (идеально), Топ-6-15: 2 (хорошо), Топ-16-40: 1 (частично), остальные: 0
        relevance_map = {}
        for idx, pid in enumerate(ideal_ids):
            if idx < 5:
                relevance_map[pid] = 3
            elif idx < 15:
                relevance_map[pid] = 2
            else:
                relevance_map[pid] = 1
                
        # Запускаем наш поиск
        retrieved_ids = search_products_tuned(q_text, products_df, article_to_id, weights, k=20)
        
        # Считаем оценки релевантности для полученной выдачи
        scores = [relevance_map.get(pid, 0) for pid in retrieved_ids]
        
        # Добиваем нулями до k=20, если выдача меньше
        if len(scores) < 20:
            scores += [0] * (20 - len(scores))
            
        metrics = evaluator.calculate_llm(scores)
        ndcgs.append(metrics["ndcg"])
        
    return np.mean(ndcgs)

def main():
    print("=== ЗАПУСК АВТО-ТЮНИНГА ПОИСКОВЫХ ВЕСОВ ===")
    
    if not os.path.exists(GOLD_STANDARD_FILE):
        print(f"Ошибка: Файл золотого стандарта {GOLD_STANDARD_FILE} не найден!")
        return
        
    with open(GOLD_STANDARD_FILE, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
        
    products_df = load_all_products()
    
    # Строим словарь артикулов
    article_to_id = {}
    for pid, row in products_df.iterrows():
        art = row.get('Артикул')
        if pd.notna(art) and str(art).strip():
            article_to_id[str(art).strip().upper()] = str(pid)
            
    evaluator = RankingMetrics(k=20)
    
    # 1. Считаем базовый (текущий) скор
    baseline_weights = {
        'w_name': 5.0,
        'w_desc': 1.5,
        'w_size': 15.0,
        'w_color': 0.0,
        'w_material': 0.0
    }
    
    print("\nРасчет базовых метрик...")
    baseline_ndcg = evaluate_parameters(products_df, article_to_id, gold_data, baseline_weights, evaluator)
    print(f"Базовый nDCG@20: {baseline_ndcg:.4f}")
    
    # 2. Сетка поиска параметров (Grid Search)
    print("\nЗапуск поиска по сетке (Grid Search) параметров...")
    
    best_ndcg = baseline_ndcg
    best_weights = baseline_weights.copy()
    
    # Сетки возможных значений весов
    name_grid = [3.0, 5.0, 7.0]
    desc_grid = [0.5, 1.0, 1.5, 2.0]
    size_grid = [10.0, 15.0, 20.0]
    color_grid = [1.0, 2.0, 4.0]
    material_grid = [1.0, 2.0, 4.0]
    
    total_combinations = len(name_grid) * len(desc_grid) * len(size_grid) * len(color_grid) * len(material_grid)
    print(f"Всего будет протестировано комбинаций весов: {total_combinations}")
    
    start_time = time.time()
    count = 0
    
    for w_name in name_grid:
        for w_desc in desc_grid:
            for w_size in size_grid:
                for w_color in color_grid:
                    for w_material in material_grid:
                        weights = {
                            'w_name': w_name,
                            'w_desc': w_desc,
                            'w_size': w_size,
                            'w_color': w_color,
                            'w_material': w_material
                        }
                        
                        ndcg = evaluate_parameters(products_df, article_to_id, gold_data, weights, evaluator)
                        count += 1
                        
                        if ndcg > best_ndcg:
                            best_ndcg = ndcg
                            best_weights = weights.copy()
                            print(f"  [Найдено улучшение!] Шаг {count}/{total_combinations} | nDCG@20: {best_ndcg:.4f} | Веса: {best_weights}")
                            
                        if count % 100 == 0:
                            elapsed = time.time() - start_time
                            eta = (elapsed / count) * (total_combinations - count)
                            print(f"  Прогресс: {count}/{total_combinations} ({count/total_combinations*100:.1f}%) | Прошло: {int(elapsed)}с | Ожидание: {int(eta)}с")
                            
    print("\n" + "=" * 50)
    print("=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ВЕСОВ ===")
    print("=" * 50)
    print(f"Базовый nDCG@20:  {baseline_ndcg:.4f}")
    print(f"Идеальный nDCG@20: {best_ndcg:.4f} (Прирост: {((best_ndcg - baseline_ndcg) / baseline_ndcg * 100):+.2f}%)")
    print("\nРекомендуемые веса для интеграции в main.py:")
    for param, val in best_weights.items():
        print(f"  {param:<12} = {val}")
    print("=" * 50)

if __name__ == "__main__":
    main()
