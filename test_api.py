import os
import json
from openai import OpenAI

# --- НАСТРОЙКИ ПОДКЛЮЧЕНИЯ ---
# Вставь сюда свой токен из Telegram. Убедись, что нет пробелов в начале или конце.
MY_TOKEN = "hydra_379313e7bb27453684b301dd13e0afeb" # <-- СЮДА ВСТАВЬ ПОЛНЫЙ ТЕКСТ

client = OpenAI(
    api_key=MY_TOKEN,
    base_url="https://hydragpt.ru/v1"
)

def run_relevance_test():
    print("=== ЗАПУСК ТЕСТА СУДЬИ (LLM-JUDGE) ===")
    
    # Пример данных для оценки
    query = "кровать белая 160х200"
    product_name = "Кровать Орхидея, белая патина, 160х200"
    product_desc = "Двуспальная кровать из ЛДСП с подъемным механизмом, цвет белый с патиной."

    print(f"Тестируем запрос: '{query}'")
    print(f"Товар: '{product_name}'")
    
    try:
        # Отправляем запрос к Kimi (она самая умная в твоем списке)
        response = client.chat.completions.create(
            model="kimi-k2p6",
            messages=[
                {
                    "role": "system", 
                    "content": "Ты — эксперт по качеству поиска мебели. Твоя задача — оценить релевантность товара запросу. Отвечай только в формате JSON."
                },
                {
                    "role": "user", 
                    "content": f"Запрос: {query}\nТовар: {product_name}\nОписание: {product_desc}\nОцени релевантность от 0 до 3. Верни JSON с полями 'score' и 'reason'."
                }
            ],
            temperature=0  # Стабильность результата
        )

        # Выводим сырой ответ для проверки
        raw_content = response.choices[0].message.content
        print("\n--- ОТВЕТ ОТ НЕЙРОСЕТИ ---")
        print(raw_content)
        
        # Проверяем, что это валидный JSON
        data = json.loads(raw_content)
        print(f"\nВердикт судьи: {data['score']}/3")
        print(f"Обоснование: {data['reason']}")
        print("\n=== ТЕСТ ПРОЙДЕН УСПЕШНО ===")

    except Exception as e:
        print("\n!!! ОШИБКА !!!")
        print(f"Тип ошибки: {type(e).__name__}")
        print(f"Сообщение: {e}")
        print("\nЕсли ошибка 401 — проверь токен. Если 404 — проверь имя модели.")

if __name__ == "__main__":
    run_relevance_test()