import json
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import argparse
import os

import config
from llm_judge import evaluate_pair, load_cache
from metrics import RankingMetrics

def load_all_products():
    import glob
    import os
    
    base_dir = config.BASE_DIR
    dfs = []
    
    # Ищем файлы *_таблица.csv в директории проекта
    csv_pattern = os.path.join(base_dir, "*_таблица.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print("Внимание: Файлы *_таблица.csv не найдены! Пробую загрузить из products.csv")
        fallback_path = os.path.join(base_dir, "products.csv")
        if os.path.exists(fallback_path):
            df = pd.read_csv(fallback_path)
            df = df.rename(columns={'Price': 'Цена', 'Description': 'Описание', 'Name': 'Наименование'})
            df['ID'] = df['ID'].astype(str)
            df.set_index('ID', inplace=True)
            return df
        else:
            raise FileNotFoundError("Не найдены ни *_таблица.csv, ни products.csv!")

    for f in csv_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                first_line = file.readline()
            sep = '\t' if '\t' in first_line else ','
            df = pd.read_csv(f, sep=sep)
            
            # Нормализация колонок
            col_map = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in ['id']:
                    col_map[col] = 'ID'
                elif col_lower in ['артикул', 'article']:
                    col_map[col] = 'Артикул'
                elif col_lower in ['тип товара', 'product type']:
                    col_map[col] = 'Тип товара'
                elif col_lower in ['наименование', 'name']:
                    col_map[col] = 'Наименование'
                elif col_lower in ['габариты', 'dimensions']:
                    col_map[col] = 'Габариты'
                elif col_lower in ['цвет', 'color']:
                    col_map[col] = 'Цвет'
                elif col_lower in ['материал', 'material']:
                    col_map[col] = 'Материал'
                elif col_lower in ['особенности', 'features']:
                    col_map[col] = 'Особенности'
                elif col_lower in ['цена (₽)', 'price (rub)']:
                    col_map[col] = 'Цена'
                elif col_lower in ['домен', 'domen']:
                    col_map[col] = 'Домен'
            
            df = df.rename(columns=col_map)
            
            cols_to_keep = ['ID', 'Артикул', 'Тип товара', 'Наименование', 'Габариты', 'Цвет', 'Материал', 'Особенности', 'Цена', 'Домен']
            existing_cols = [c for c in cols_to_keep if c in df.columns]
            df = df[existing_cols]
            
            dfs.append(df)
        except Exception as e:
            print(f"Ошибка при чтении файла {f}: {e}")
            
    if not dfs:
        raise ValueError("Не удалось загрузить ни одну таблицу!")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['ID'] = combined_df['ID'].astype(str)
    
    names = []
    descriptions = []
    
    for idx, row in combined_df.iterrows():
        p_type = str(row.get('Тип товара', '')).strip() if pd.notna(row.get('Тип товара')) else ''
        p_name = str(row.get('Наименование', '')).strip() if pd.notna(row.get('Наименование')) else ''
        
        if p_type and p_name:
            if p_name.lower() in p_type.lower():
                name_str = p_type
            else:
                name_str = f"{p_type} {p_name}"
        elif p_type:
            name_str = p_type
        elif p_name:
            name_str = p_name
        else:
            name_str = 'Без названия'
            
        names.append(name_str)
        
        desc_parts = []
        if p_type:
            desc_parts.append(f"Тип товара: {p_type}")
        if pd.notna(row.get('Габариты')) and str(row.get('Габариты')).strip():
            desc_parts.append(f"Габариты: {row['Габариты']}")
        if pd.notna(row.get('Цвет')) and str(row.get('Цвет')).strip():
            desc_parts.append(f"Цвет: {row['Цвет']}")
        if pd.notna(row.get('Материал')) and str(row.get('Материал')).strip():
            desc_parts.append(f"Материал: {row['Материал']}")
        if pd.notna(row.get('Особенности')) and str(row.get('Особенности')).strip():
            desc_parts.append(f"Особенности: {row['Особенности']}")
        if pd.notna(row.get('Цена')) and str(row.get('Цена')).strip():
            desc_parts.append(f"Цена: {row['Цена']} ₽")
        if pd.notna(row.get('Артикул')) and str(row.get('Артикул')).strip():
            desc_parts.append(f"Артикул: {row['Артикул']}")
            
        desc_str = ". ".join(desc_parts)
        descriptions.append(desc_str)
        
    combined_df['Name'] = names
    combined_df['Description'] = descriptions
    combined_df.set_index('ID', inplace=True)
    return combined_df

def search_products(query_text: str, products_df: pd.DataFrame, article_to_id: dict, k: int = 20) -> list:
    query_clean = str(query_text).strip().upper()
    
    # 1. Если запрос содержит точный артикул, возвращаем только этот товар
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
        
    # 2. Обычный текстовый поиск с весами по совпадению слов
    query_words = [w.lower() for w in query_clean.split() if len(w) >= 2]
    if not query_words:
        return list(products_df.index[:k])
        
    scores = {}
    for pid, row in products_df.iterrows():
        score = 0.0
        
        # Получаем текстовые поля товара
        p_name = str(row.get('Name', '')).lower()
        p_desc = str(row.get('Description', '')).lower()
        
        # Проверяем совпадение каждого слова запроса
        for word in query_words:
            # Если это размерная цифра (например, 160)
            if word.isdigit() and len(word) >= 2:
                if word in p_name or word in p_desc:
                    score += 15.0  # Большой вес за точное совпадение размера!
            else:
                # Обычное совпадение слов
                if word in p_name:
                    score += 5.0  # Вес за имя/тип товара
                if word in p_desc:
                    score += 1.5  # Вес за описание/характеристики
                    
        if score > 0:
            scores[pid] = score
            
    # Сортируем по убыванию скора
    sorted_pids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    retrieved_ids = [pid for pid, _ in sorted_pids]
    
    return retrieved_ids[:k]

def main():
    parser = argparse.ArgumentParser(description="Furniture Search Evaluation Pipeline")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the judge cache before running")
    parser.add_argument("--no-cache", action="store_true", help="Disable using cache")
    args = parser.parse_args()

    print("=== ЗАПУСК EVALUATION PIPELINE (v3.0 Модульный) ===")
    
    if args.clear_cache:
        if os.path.exists(config.CACHE_FILE):
            try:
                os.remove(config.CACHE_FILE)
                print("Кэш судьи успешно очищен!")
            except Exception as e:
                print(f"Ошибка очистки кэша: {e}")
    
    # 1. Загрузка данных
    try:
        products_df = load_all_products()
        print(f"Успешно загружено {len(products_df)} товаров из таблиц данных.")
    except Exception as e:
        print(f"Ошибка загрузки товаров: {e}")
        return

    # Построение словаря артикулов для быстрого поиска
    article_to_id = {}
    for pid, row in products_df.iterrows():
        art = row.get('Артикул')
        if pd.notna(art) and str(art).strip():
            art_clean = str(art).strip().upper()
            if art_clean:
                article_to_id[art_clean] = str(pid)

    try:
        with open(config.QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries = json.load(f)
    except Exception as e:
        print(f"Ошибка чтения {config.QUERIES_FILE}: {e}")
        return

    if args.no_cache:
        cache = {}
        print("Использование кэша отключено.")
    else:
        cache = load_cache()
    evaluator = RankingMetrics(k=20)
    evaluation_report = []

    for q in queries:
        query_id = str(q.get('query_id', 'unknown'))
        query_text = q.get('query_text', '')
        top_ids = q.get('top_20_product_ids', [])
        
        # Если команда еще не прислала ответы или содержит старые невалидные ID (например, "1"),
        # генерируем поисковую выдачу из нашей новой базы данных!
        is_invalid = any(pid not in products_df.index for pid in top_ids) if top_ids else True
        if is_invalid:
            print(f"\n[ПОИСК] Поиск '{query_text}' в базе данных...")
            top_ids = search_products(query_text, products_df, article_to_id, k=20)
            q['top_20_product_ids'] = top_ids
            
        if not top_ids:
            print(f"\n[Пропуск] Запрос: '{query_text}'. Нет данных о выдаче.")
            continue
            
        print(f"\n[Запрос] {query_text} (ID: {query_id})")
        
        scores = []
        detailed_results = []
        
        # 2. Оценка через Судью (LLM)
        for pid in top_ids[:20]:
            pid = str(pid)
            if pid not in products_df.index:
                print(f"  Товар ID={pid} не найден в базе!")
                scores.append(0)
                continue
                
            row = products_df.loc[pid]
            if isinstance(row, pd.DataFrame): 
                row = row.iloc[0]
                
            p_name = str(row.get('Name', 'Без названия'))
            p_desc = str(row.get('Description', 'Без описания'))
            p_art = row.get('Артикул')
            
            score, reason = evaluate_pair(
                query_id, query_text, pid, p_name, p_desc, cache,
                product_article=p_art, article_to_id=article_to_id
            )
            if score is None:
                continue

            scores.append(score)
            print(f"  Товар ID={pid} | Оценка: {score} | Обоснование: {reason}")
            detailed_results.append({
                "product_id": pid, "name": p_name, "score": score, "reason": reason
            })
            
        # 3. Расчет метрик
        metrics_result = evaluator.calculate_llm(scores)
        print(f"-> Метрики (LLM-Judge): nDCG@20 = {metrics_result['ndcg']:.4f}, Precision@5 = {metrics_result['precision@5']:.4f}")
        
        # Опционально: Если у нас есть эталонные ответы (target_product_ids), можем посчитать traditional
        target_ids = q.get('target_product_ids', [])
        traditional_metrics = {}
        if target_ids:
            traditional_metrics = evaluator.calculate_traditional(top_ids, target_ids)
            print(f"-> Метрики (Traditional): nDCG@20 = {traditional_metrics['ndcg']}, Hit = {traditional_metrics['hit']}")
        
        evaluation_report.append({
            "query_id": query_id,
            "query_text": query_text,
            "metrics_llm": metrics_result,
            "metrics_traditional": traditional_metrics,
            "scores": scores,
            "results": detailed_results
        })

    if evaluation_report:
        avg_ndcg = np.mean([r["metrics_llm"]["ndcg"] for r in evaluation_report])
        avg_p5 = np.mean([r["metrics_llm"]["precision@5"] for r in evaluation_report])
    else:
        avg_ndcg, avg_p5 = 0.0, 0.0

    final_output = {
        "summary": {
            "average_llm_ndcg@20": round(avg_ndcg, 4),
            "average_llm_precision@5": round(avg_p5, 4),
            "total_queries_evaluated": len(evaluation_report)
        },
        "queries": evaluation_report
    }

    with open(config.REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n=== ИТОГИ ===")
    print(f"Отчет сохранен в {config.REPORT_FILE}")
    print(f"Средний nDCG@20: {avg_ndcg:.4f}")
    
if __name__ == "__main__":
    main()
