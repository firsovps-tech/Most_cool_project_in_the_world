import pandas as pd
import json
import time
import os

INPUT_JSON = "input.json"
OUTPUT_JSON = "output.json"
SOURCE_FILE = "source.csv"      # исходная таблица
RESULT_FILE = "result.csv"      # итоговая таблица

TIMEOUT = 60  # максимум ожидания ответа (сек)


def wait_for_output(prev_mtime):
    """Ждём обновления output.json"""
    start_time = time.time()
    
    while True:
        if os.path.exists(OUTPUT_JSON):
            current_mtime = os.path.getmtime(OUTPUT_JSON)
            if current_mtime != prev_mtime:
                return True
        
        if time.time() - start_time > TIMEOUT:
            raise TimeoutError("Превышено время ожидания output.json")
        
        time.sleep(0.5)


def main():
    df = pd.read_csv(SOURCE_FILE, sep=';')
    results = []

    # если output.json уже существует — запомним время изменения
    prev_mtime = os.path.getmtime(OUTPUT_JSON) if os.path.exists(OUTPUT_JSON) else 0
    i1 = 0
    for index, row in df.iterrows():
        print(f"Обработка строки {i1}")
        i1 += 1
        # формируем входной json
        input_data = {
            "query_id": 1,
            "query_text": str(row["product_name"]),
        }

        # записываем input.json
        with open(INPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(input_data, f, ensure_ascii=False, indent=2)

        # ждём появления нового output
        wait_for_output(prev_mtime)

        # читаем output
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            output_data = json.load(f)

        output_data["price"] = row["price_rub"]

        # обновляем время изменения
        prev_mtime = os.path.getmtime(OUTPUT_JSON)

        results.append(output_data)

    # сохраняем итог
    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

    print("✅ Готово. Результат сохранён в", RESULT_FILE)


if __name__ == "__main__":
    main()
