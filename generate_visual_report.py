#!/usr/bin/env python3
import json
import os
import pandas as pd
import config
from main import load_all_products

INPUT_FILE = os.path.join(config.BASE_DIR, "smart_gold_standard.json")
OUTPUT_MD = os.path.join(config.BASE_DIR, "smart_gold_standard_visual.md")

def main():
    print("Загружаем данные для визуализации...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Ошибка: Файл {INPUT_FILE} не найден! Сначала запустите create_smart_dataset.py")
        return
        
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        gold_standard = json.load(f)
        
    products_df = load_all_products()
    
    markdown_content = []
    markdown_content.append("# 👁️ Визуализация «Золотого стандарта» для ручной проверки")
    markdown_content.append("Этот файл содержит результаты ранжирования мебели моделью **Kimi K2.6** для 15 тестовых запросов.")
    markdown_content.append("Позиции отсортированы по релевантности от самой идеальной (№1) до наименее подходящей (№20).\n")
    markdown_content.append("---")
    
    for idx, item in enumerate(gold_standard):
        q_text = item["query_text"]
        q_id = item["query_id"]
        product_ids = item["ideal_product_ids"]
        
        markdown_content.append(f"\n## Запрос №{idx + 1}: «{q_text}» (ID: {q_id})")
        markdown_content.append(f"Модель отобрала {len(product_ids)} лучших кандидатов:\n")
        
        for num, pid in enumerate(product_ids, 1):
            if pid in products_df.index:
                row = products_df.loc[pid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                
                name = row.get('Name', 'Без названия')
                art = row.get('Артикул', '-')
                color = row.get('Цвет', '-')
                mat = row.get('Материал', '-')
                dims = row.get('Габариты', '-')
                price = row.get('Цена', '-')
                
                # Формируем красивую визуальную строку для каждого товара
                product_info = (
                    f"**{num}. ID: `{pid}`** | **{name}** | "
                    f"Арт: `{art}` | "
                    f"Цвет: *{color}* | "
                    f"Габариты: {dims} | "
                    f"Материал: {mat} | "
                    f"**Цена: {price} руб.**"
                )
                markdown_content.append(product_info)
            else:
                markdown_content.append(f"**{num}. ID: `{pid}`** | [Ошибка] Товар не найден в базе данных CSV!")
                
        markdown_content.append("\n---")
        
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_content))
        
    print(f"Успех! Файл визуализации успешно создан: {OUTPUT_MD}")

if __name__ == "__main__":
    main()
