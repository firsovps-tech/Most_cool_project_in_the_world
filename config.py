import os

# Получаем абсолютный путь к корню проекта (папке, где лежит config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

API_KEY = os.getenv("HYDRAGPT_TOKEN", "hydra_379313e7bb27453684b301dd13e0afeb")
BASE_URL = "https://hydragpt.ru/v1"
MODEL_NAME = "gpt-oss-120b"

CACHE_FILE = os.path.join(BASE_DIR, "judge_cache.json")
PRODUCTS_FILE = os.path.join(BASE_DIR, "products.csv")
QUERIES_FILE = os.path.join(BASE_DIR, "queries.json")
REPORT_FILE = os.path.join(BASE_DIR, "evaluation_results.json")
