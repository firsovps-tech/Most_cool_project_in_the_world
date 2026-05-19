#!/usr/bin/env python3
import json
import os
import re
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

import config
from main import load_all_products

# Список 100 золотых запросов для Риты
QUERIES = [
    # Категории и общие
    {"id": "g1", "text": "белая кровать 160"},
    {"id": "g2", "text": "стул деревянный кухонный"},
    {"id": "g3", "text": "стол обеденный раздвижной"},
    {"id": "g4", "text": "угловой диван рогожка"},
    {"id": "g5", "text": "журнальный столик мрамор"},
    {"id": "g6", "text": "шкаф купе с зеркалом"},
    {"id": "g7", "text": "комод дуб артизан"},
    {"id": "g8", "text": "кресло мешок велюр"},
    {"id": "g9", "text": "барный стул экокожа"},
    {"id": "g10", "text": "компьютерное кресло черное"},
    {"id": "g11", "text": "стеллаж лофт металл"},
    {"id": "g12", "text": "детский шкаф белый"},
    {"id": "g13", "text": "полутораспальная кровать велюр"},
    {"id": "g14", "text": "стол кухонный стекло"},
    {"id": "g15", "text": "кресло реклайнер графит"},
    {"id": "g16", "text": "подвесная тумба под раковину"},
    {"id": "g17", "text": "диван канапе велюр"},
    {"id": "g18", "text": "вешалка для одежды напольная"},
    {"id": "g19", "text": "круглый обеденный стол"},
    {"id": "g20", "text": "письменный стол лдсп"},
    
    # Кровати и матрасы с размерами
    {"id": "g21", "text": "двуспальная кровать с подъемным механизмом"},
    {"id": "g22", "text": "сервант с подсветкой"},
    {"id": "g23", "text": "зеркало в раме с подсветкой"},
    {"id": "g24", "text": "подвесной книжный шкаф"},
    {"id": "g25", "text": "диван кушетка серый"},
    {"id": "g26", "text": "корзина для белья тканевая"},
    {"id": "g27", "text": "торшер с полкой дерево"},
    {"id": "g28", "text": "покрывало летнее зеленый"},
    {"id": "g29", "text": "шкаф витрина классика"},
    {"id": "g30", "text": "стул из массива бука"},
    {"id": "g31", "text": "подвесное кресло ротанг"},
    {"id": "g32", "text": "скатерть фуршетная белая"},
    {"id": "g33", "text": "двуспальная кровать 180х200"},
    {"id": "g34", "text": "диван прямой раскладной"},
    {"id": "g35", "text": "компьютерный стол венге"},
    {"id": "g36", "text": "тумба прикроватная дуб сонома"},
    {"id": "g37", "text": "полубарный стул серый"},
    {"id": "g38", "text": "стеллаж многоуровневый лофт"},
    {"id": "g39", "text": "кресло реклайнер с подножкой"},
    {"id": "g40", "text": "шкаф пенал прихожая"},
    
    # Специфические цвета, материалы и особенности
    {"id": "g41", "text": "обеденный стол орех"},
    {"id": "g42", "text": "детская кровать белая"},
    {"id": "g43", "text": "кухонный стул серый"},
    {"id": "g44", "text": "диван угловой экокожа"},
    {"id": "g45", "text": "туалетный столик с подсветкой"},
    {"id": "g46", "text": "зеркало настенное прямоугольное"},
    {"id": "g47", "text": "диван прямой 3-местный"},
    {"id": "g48", "text": "кресло реклайнер рогожка"},
    {"id": "g49", "text": "столик журнальный круглый"},
    {"id": "g50", "text": "шкаф для одежды 3-дверный"},
    
    # Точные артикулы из Table 4 (прямой поиск)
    {"id": "g51", "text": "ЯР-PL-B16"},
    {"id": "g52", "text": "СТЖ-906-C58"},
    {"id": "g53", "text": "БАН-1004-C80"},
    {"id": "g54", "text": "ЗЕР-6080-C56"},
    {"id": "g55", "text": "ПР-1206-781"},
    {"id": "g56", "text": "ТМБ-706-B89"},
    {"id": "g57", "text": "МАТ-1620-C12"},
    {"id": "g58", "text": "ШК-КУП-120"},
    {"id": "g59", "text": "ДИВ-УГЛ-300"},
    {"id": "g60", "text": "КРВ-АЛЬФ-160"},
    
    # Английские товары из Table 3
    {"id": "g61", "text": "Christmas tree slim"},
    {"id": "g62", "text": "Guitar stand foldable"},
    {"id": "g63", "text": "ящик для хранения с колесами"},
    {"id": "g64", "text": "готовые шторы блэкаут"},
    {"id": "g65", "text": "подставка под монитор бамбук"},
    {"id": "g66", "text": "пуфик велюровый круглый"},
    {"id": "g67", "text": "диван клик-кляк шенилл"},
    {"id": "g68", "text": "кухонный уголок со столом"},
    {"id": "g69", "text": "офисный стул на колесиках"},
    {"id": "g70", "text": "кровать из массива сосны"},
    
    # Сложные многокритериальные запросы
    {"id": "g71", "text": "шкаф купе трехдверный"},
    {"id": "g72", "text": "стол на металлокаркасе"},
    {"id": "g73", "text": "кресло качалка деревянное"},
    {"id": "g74", "text": "банкетка в прихожую обувница"},
    {"id": "g75", "text": "стеллаж для книг белый"},
    {"id": "g76", "text": "кровать 140х200 с матрасом"},
    {"id": "g77", "text": "раскладушка с матрасом"},
    {"id": "g78", "text": "комод с выдвижными ящиками"},
    {"id": "g79", "text": "вешалка стойка для одежды"},
    {"id": "g80", "text": "туалетный столик с зеркалом"},
    
    # Разное и домашний декор
    {"id": "g81", "text": "навесная полка на стену"},
    {"id": "g82", "text": "кухонный гарнитур угловой"},
    {"id": "g83", "text": "диван еврокнижка велюр"},
    {"id": "g84", "text": "обеденная группа стол стул"},
    {"id": "g85", "text": "кровать трансформер детская"},
    {"id": "g86", "text": "садовая мебель из ротанга"},
    {"id": "g87", "text": "складной туристический стул"},
    {"id": "g88", "text": "зеркало в полный рост"},
    {"id": "g89", "text": "кровать двуспальная венге"},
    {"id": "g90", "text": "журнальный столик лофт"},
    {"id": "g91", "text": "комод белый глянец"},
    {"id": "g92", "text": "стеллаж перегородка для комнаты"},
    {"id": "g93", "text": "стол трансформер обеденный"},
    {"id": "g94", "text": "компьютерный стол белый"},
    {"id": "g95", "text": "кресло для геймера игровое"},
    {"id": "g96", "text": "диван книжка рогожка"},
    {"id": "g97", "text": "кровать 160х200 бежевая"},
    {"id": "g98", "text": "кухонный стол круглый"},
    {"id": "g99", "text": "напольное зеркало в раме"},
    {"id": "g100", "text": "шкаф для обуви обувница"}
]

# Словари синонимов и категорий
CATEGORY_WORDS = {
    "кровать": ["кровать", "кровати", "полутораспальная", "двуспальная", "чердак", "bed", "beds"],
    "диван": ["диван", "кушетка", "канапе", "софа", "sofa", "couch"],
    "стул": ["стул", "кресло", "полубарный", "пуф", "пуфик", "банкетка", "табурет", "chair", "armchair", "stool"],
    "стол": ["стол", "столик", "парта", "table", "desk"],
    "шкаф": ["шкаф", "пенал", "витрина", "гардероб", "гарнитур", "буфет", "сервант", "wardrobe", "cabinet", "closet"],
    "стеллаж": ["стеллаж", "полка", "полки", "стелаж", "rack", "shelf", "shelves"],
    "комод": ["комод", "тумба", "тумбочка", "dresser", "nightstand", "chest"],
    "зеркало": ["зеркало", "зеркала", "трюмо", "mirror"],
    "матрас": ["матрас", "матрац", "mattress"],
    "вешалка": ["вешалка", "стойка напольная", "hanger", "rack"],
    "шторы": ["шторы", "занавески", "портьеры", "curtains"]
}

COLOR_SYNONYMS = {
    "белый": ["бел", "white", "шампань"],
    "черный": ["черн", "чёрн", "black", "темн"],
    "серый": ["сер", "grey", "gray", "графит"],
    "бежевый": ["беж", "beige", "песоч"],
    "орех": ["орех", "walnut"],
    "венге": ["венге", "wenge"],
    "золото": ["золот", "gold"],
    "розовый": ["розов", "rose", "пудр"],
    "зеленый": ["зелен", "зелён", "green", "оливк", "изумруд"],
    "синий": ["син", "blue", "бирюз", "голубой"],
    "латунь": ["латун", "brass"]
}

MATERIAL_KEYWORDS = {
    "рогожка": ["рогож"],
    "велюр": ["велюр"],
    "кожа": ["кожа", "кожан", "leather"],
    "экокожа": ["экокожа", "экокожи", "винилискожа"],
    "массив": ["массив", "дерево", "дуб", "орех", "бук", "сосна", "wood"],
    "стекло": ["стекл", "glass"],
    "металл": ["метал", "желез", "сталь", "metal", "steel"],
    "лдсп": ["лдсп", "ldsp"],
    "мдф": ["мдф", "mdf"],
    "пластик": ["пластик", "plastic"],
    "ротанг": ["ротанг", "rattan"]
}

STOP_WORDS = {"для", "в", "с", "из", "на", "под", "и", "ко", "со", "без", "tm", "тм"}

def compute_relevance_score(query: str, product_row, article_to_id: dict) -> float:
    # 1. Очистка и токенизация
    q_clean = str(query).strip().upper()
    q_words = [w.lower() for w in q_clean.split() if w.lower() not in STOP_WORDS]
    
    p_id = str(product_row.name)
    p_name = str(product_row.get('Name', '')).lower()
    p_desc = str(product_row.get('Description', '')).lower()
    p_art = str(product_row.get('Артикул', '')).strip().upper()
    p_type = str(product_row.get('Тип товара', '')).lower()
    p_color = str(product_row.get('Цвет', '')).lower()
    p_mat = str(product_row.get('Материал', '')).lower()
    p_feat = str(product_row.get('Особенности', '')).lower()
    
    score = 0.0
    
    # 2. ПРЯМОЙ ОБХОД ПО АРТИКУЛУ (Абсолютный топ #1)
    if p_art and q_clean == p_art:
        return 100000.0  # Максимальный приоритет
        
    # Если запрос содержит этот артикул как токен
    if p_art and len(p_art) > 3 and p_art in q_clean:
        score += 50000.0
        
    # 3. ПОИСК И СОВПАДЕНИЕ ПО КАТЕГОРИИ
    category_matched = False
    for cat, aliases in CATEGORY_WORDS.items():
        # Есть ли слово категории в запросе?
        cat_in_query = any(alias in q_words for alias in aliases)
        if cat_in_query:
            # Соответствует ли товар этой категории?
            matches_product = False
            
            # Проверяем тип товара или Name
            matches_product = any(alias in p_type or alias in p_name for alias in aliases)
            
            # Особая проверка для смежных категорий (например, кушетка/диван, стул/кресло)
            if matches_product:
                score += 80.0
                category_matched = True
            elif cat == "стул" and "кресло" in p_type:
                score += 35.0  # Частичное совпадение категории
                category_matched = True
            elif cat == "диван" and "кушетка" in p_type:
                score += 45.0  # Очень близкое совпадение
                category_matched = True
                
    # Если в запросе есть категория, а товар относится к другой категории — даем жесткий штраф!
    if not category_matched:
        # Проверяем, не принадлежит ли товар к другой явной категории
        for cat, aliases in CATEGORY_WORDS.items():
            cat_in_query = any(alias in q_words for alias in aliases)
            if not cat_in_query:
                # Если товар принадлежит к этой категории, а в запросе ее нет, но есть другая
                has_other_cat_in_query = any(any(a in q_words for a in als) for c, als in CATEGORY_WORDS.items() if c != cat)
                if has_other_cat_in_query and any(alias in p_type for alias in aliases):
                    score -= 50.0 # Огромный штраф за несовпадение базовой сущности (например, шкаф по запросу стул)

    # 4. РАЗМЕРЫ (КРИТИЧНО ДЛЯ МЕБЕЛИ)
    # Ищем цифры в запросе (например, 160, 180, 120х60)
    dimensions_in_query = re.findall(r'\b\d{2,4}\b', q_clean)
    if dimensions_in_query:
        for dim in dimensions_in_query:
            dim_str = str(dim)
            # Есть ли размер в габаритах или названии товара?
            p_dims = str(product_row.get('Габариты', '')).lower()
            if dim_str in p_dims or dim_str in p_name or dim_str in p_desc:
                score += 45.0  # Точное попадание по размеру!
            else:
                score -= 15.0  # Штраф за несовпадение размеров, если они указаны в запросе

    # 5. ЦВЕТА
    color_in_query = False
    for col_key, aliases in COLOR_SYNONYMS.items():
        if any(alias in q_words for alias in aliases):
            color_in_query = True
            # Проверяем, совпадает ли цвет
            if any(alias in p_color or alias in p_name or alias in p_desc for alias in aliases):
                score += 35.0
            else:
                # Если у товара другой цвет
                if p_color and p_color.strip() and not any(alias in p_color for alias in aliases):
                    score -= 25.0 # Жесткий штраф за не тот цвет!
                    
    # 6. МАТЕРИАЛЫ
    for mat_key, aliases in MATERIAL_KEYWORDS.items():
        if any(alias in q_words for alias in aliases):
            if any(alias in p_mat or alias in p_desc or alias in p_name for alias in aliases):
                score += 25.0
                
    # 7. ДРУГИЕ ОСОБЕННОСТИ И ОПИСАНИЕ
    # Пересечение оставшихся слов
    for word in q_words:
        # Пропускаем слова категорий, цветов и материалов, которые уже учтены с большими весами
        is_special = False
        for aliases in list(CATEGORY_WORDS.values()) + list(COLOR_SYNONYMS.values()) + list(MATERIAL_KEYWORDS.values()):
            if word in aliases:
                is_special = True
                break
        if is_special:
            continue
            
        # Обычное текстовое совпадение
        if word in p_name:
            score += 15.0
        elif word in p_type:
            score += 10.0
        elif word in p_feat:
            score += 8.0
        elif word in p_desc:
            score += 3.0
            
    # Небольшой штраф за слишком длинный хвост или отсутствие совпадений
    if score <= 0:
        score = -100.0
        
    return score

def main():
    print("=== ГЕНЕРАЦИЯ ЗОЛОТОГО СТАНДАРТА ДЛЯ РИТЫ ===")
    
    # 1. Загружаем все 4894 товаров
    try:
        products_df = load_all_products()
        print(f"Загружено {len(products_df)} товаров для ранжирования.")
    except Exception as e:
        print(f"Ошибка загрузки товаров: {e}")
        return
        
    # Строим словарь артикулов
    article_to_id = {}
    for pid, row in products_df.iterrows():
        art = row.get('Артикул')
        if pd.notna(art) and str(art).strip():
            art_clean = str(art).strip().upper()
            if art_clean:
                article_to_id[art_clean] = str(pid)

    gold_standard = []
    
    # 2. Ранжируем товары для каждого из 100 запросов
    for idx, q in enumerate(QUERIES):
        q_id = q["id"]
        q_text = q["text"]
        
        # Считаем релевантность для всех товаров
        scores = []
        for pid, row in products_df.iterrows():
            score = compute_relevance_score(q_text, row, article_to_id)
            scores.append((pid, score))
            
        # Сортируем по убыванию очков релевантности
        # Если очки равны, сортируем по ID для стабильности
        scores.sort(key=lambda x: (-x[1], x[0]))
        
        # Выбираем ровно 100 лучших ответов
        top_100_ids = [str(pid) for pid, score in scores[:100]]
        
        # Проверяем, что получили ровно 100 идеальных ответов
        if len(top_100_ids) < 100:
            # Дозаполняем оставшимися товарами, если в базе вдруг не хватило (хотя у нас 4894 товара)
            all_ids = [str(x) for x in products_df.index]
            for aid in all_ids:
                if aid not in top_100_ids:
                    top_100_ids.append(aid)
                if len(top_100_ids) == 100:
                    break
                    
        gold_standard.append({
            "query_id": q_id,
            "query_text": q_text,
            "ideal_product_ids": top_100_ids
        })
        
        # Печатаем первые 3 топ-результата для проверки качества v3.0
        print(f"[{idx+1}/100] Запрос: '{q_text}'")
        for rank, pid in enumerate(top_100_ids[:3]):
            row = products_df.loc[pid]
            p_name = row.get('Name', 'Без названия')
            p_art = row.get('Артикул', '')
            p_color = row.get('Цвет', '')
            p_dims = row.get('Габариты', '')
            print(f"  #{rank+1}: ID={pid} | {p_name} | Арт={p_art} | Цв={p_color} | Габ={p_dims}")

    # 3. Сохраняем в rita_gold_standard.json
    output_file = os.path.join(config.BASE_DIR, "rita_gold_standard.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gold_standard, f, ensure_ascii=False, indent=2)
        
    print(f"\n=== УСПЕХ ===")
    print(f"Золотой стандарт для Риты из 100 запросов по 100 идеальных ответов успешно сохранен в {output_file}!")
    print(f"Размер файла: {os.path.getsize(output_file)} байт.")

if __name__ == "__main__":
    main()
