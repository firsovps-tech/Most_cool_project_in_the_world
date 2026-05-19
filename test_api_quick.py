import os
from openai import OpenAI
import json

client = OpenAI(
    api_key="hydra_379313e7bb27453684b301dd13e0afeb",
    base_url="https://hydragpt.ru/v1"
)

prompt = """
Запрос: стул
Товар: Стул деревянный
Описание: Дубовый стул

Оцени релевантность товара этому запросу по шкале от 0 до 3.
Отвечай СТРОГО в формате JSON с полями 'score' (число) и 'reason' (строка).
"""

response = client.chat.completions.create(
    model="deepseek-v3p2",
    messages=[
        {"role": "system", "content": "Ты — эксперт по качеству поиска мебели. Твоя задача — оценить релевантность товара запросу. Отвечай только в формате JSON."},
        {"role": "user", "content": prompt}
    ],
    temperature=0
)

print("RAW CONTENT:", repr(response.choices[0].message.content))
