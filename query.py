import json
import os

FILE_NAME = "input.json"

def load_data():
  if not os.path.exists(FILE_NAME):
    return []

  with open(FILE_NAME, "r", encoding="utf-8") as file:
    try:
      data = json.load(file)
      if isinstance(data, list):
        return data
      return []
    except json.JSONDecodeError:
      return []


def save_data(data):
  with open(FILE_NAME, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)


def main():
  query_text = input("Введите запрос: ").strip()

  if not query_text:
    print("Пустую строку не добавляем.")
    return

  data = load_data()

  query_id = f"q{len(data) + 1}"

  new_query = {
    "query_id": query_id,
    "query_text": query_text
  }

  data.append(new_query)
  save_data(data)

  print(f"Добавлено: {query_id}")


if __name__ == "__main__":
  main()