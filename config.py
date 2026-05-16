import os

DATA_PATH = "data/products.csv"
TYPE1_QUERIES_PATH = "search_queries_test.json"
TYPE2_QUERIES_PATH = "gold_top20_queries.json"

TOP_K = 20

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-key-here")
LLM_MODEL = "gpt-4o-mini"
