import json
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from src.core.search_core import field_value
from src.core.search_cache import load_search_engine_cache
from src.ranking.boost_ranker import WeightedBoostProductSearch, load_weights_by_domain
from src.parsing.project import normalize_one_query


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "search_cache"
MODEL_DIR = PROJECT_ROOT / "trained_boost_ranker"
BEST_PARAMS_PATH = PROJECT_ROOT / "grid_best_params.json"

TOP_K_DEFAULT = 10
FIRST_STAGE_LIMIT = 1000
FINAL_CANDIDATE_LIMIT = None

load_dotenv(PROJECT_ROOT / ".env")

app = Flask(__name__)
_search_engine = None
_search_system = None


def make_weights_for_domain(feature_names, params):
  """Создает веса для домена из grid_best_params.json."""
  weights = {}

  for name in feature_names:
    if name == "exact_id_score":
      weights[name] = params.get("w_exact_id", 0.0)
    elif name == "bm25_score":
      weights[name] = params.get("w_bm25", 0.0)
    elif name == "general_embedding_score":
      weights[name] = params.get("w_general_embedding", 0.0)
    elif name == "detail_embedding_score":
      weights[name] = params.get("w_detail_embedding", 0.0)
    elif name == "price_score":
      weights[name] = params.get("w_price", 0.0)
    elif name.endswith("_embedding_score"):
      weights[name] = params.get("w_field_embedding", 0.0)
    elif name.startswith("product_has_"):
      weights[name] = params.get("w_product_has_field", 0.0)
    else:
      weights[name] = 0.0

  return weights


def load_robust_weights(search_engine):
  """
  Загружает обученные веса.
  Если папка домена распаковалась с испорченным Unicode-именем,
  недостающие веса пересобираются из grid_best_params.json.
  """
  try:
    weights_by_domain = load_weights_by_domain(MODEL_DIR)
  except Exception:
    weights_by_domain = {}

  missing_domains = [
    domain for domain in search_engine.domain_configs
    if domain not in weights_by_domain
  ]

  if missing_domains and BEST_PARAMS_PATH.exists():
    with BEST_PARAMS_PATH.open("r", encoding="utf-8") as file:
      params = json.load(file)

    for domain in missing_domains:
      feature_names = search_engine.get_feature_names(domain)
      weights_by_domain[domain] = make_weights_for_domain(feature_names, params)

  return weights_by_domain


def get_search_system():
  """Ленивая загрузка: тяжелая модель и индексы поднимаются при первом запросе."""
  global _search_engine, _search_system

  if _search_system is None:
    _search_engine = load_search_engine_cache(CACHE_DIR)
    weights_by_domain = load_robust_weights(_search_engine)
    _search_system = WeightedBoostProductSearch(
      search_engine=_search_engine,
      weights_by_domain=weights_by_domain,
    )

  return _search_system


def guess_domain(query_text, domains):
  forced_domain = field_value(os.getenv("WEB_FORCE_DOMAIN"))
  if forced_domain in domains:
    return forced_domain

  has_cyrillic = any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in query_text)

  if has_cyrillic and "Мебель" in domains:
    return "Мебель"

  if not has_cyrillic and "Furniture" in domains:
    return "Furniture"

  if "Мебель" in domains:
    return "Мебель"

  return next(iter(domains))


def use_llm_parser():
  value = os.getenv("WEB_USE_LLM", "0").strip().lower()
  return value in {"1", "true", "yes", "y", "да"}


def manual_query(query_text, domain, search_engine):
  """Запасной вариант без LLM: кладем весь запрос в первое поле домена."""
  fields = search_engine.get_domain_config(domain).searchable_fields
  characteristics = {field: "-" for field in fields}

  if fields:
    characteristics[fields[0]] = query_text

  return {
    "query_id": "web",
    "id": query_text,
    "raw_query_text": query_text,
    "corrected_query_text": query_text,
    "query_text": query_text,
    "domain": domain,
    "category": domain,
    "characteristics": characteristics,
  }


def prepare_query(query_text):
  search_system = get_search_system()
  search_engine = search_system.search_engine
  domain = guess_domain(query_text, search_engine.domain_configs.keys())

  try:
    return normalize_one_query(
      query_id="web",
      query_text=query_text,
      known_domain=domain,
      use_llm=use_llm_parser(),
    )
  except Exception:
    return manual_query(query_text, domain, search_engine)


def make_result_card(row):
  characteristics = row.get("characteristics", {}) or {}

  name = (
    field_value(characteristics.get("Наименование"))
    or field_value(characteristics.get("Name"))
  )
  product_type = (
    field_value(characteristics.get("Тип товара"))
    or field_value(characteristics.get("Product Type"))
  )
  title = " ".join(part for part in [product_type, name] if part).strip()

  if not title:
    title = field_value(row.get("description")) or f"Результат {row.get('product_id')}"

  visible_characteristics = []
  for key, value in characteristics.items():
    value = field_value(value)
    if value and value != "-":
      visible_characteristics.append({"name": key, "value": value})

  return {
    "id": field_value(row.get("product_id")),
    "article": field_value(row.get("article")),
    "title": title,
    "description": field_value(row.get("description")),
    "domain": field_value(row.get("domain")),
    "price": field_value(row.get("price")),
    "score": row.get("ranker_score"),
    "characteristics": visible_characteristics,
  }


@app.get("/")
def index():
  return render_template("index.html")


@app.get("/health")
def health():
  return jsonify({"status": "ok"})


@app.post("/api/search")
def api_search():
  payload = request.get_json(silent=True) or {}
  query_text = field_value(payload.get("query"))

  if not query_text:
    return jsonify({"error": "Введите запрос"}), 400

  try:
    top_k = int(payload.get("top_k", TOP_K_DEFAULT))
  except (TypeError, ValueError):
    top_k = TOP_K_DEFAULT

  top_k = max(1, min(top_k, 50))

  try:
    search_system = get_search_system()
    prepared_query = prepare_query(query_text)
    results = search_system.search(
      prepared_query=prepared_query,
      top_k=top_k,
      first_stage_limit=FIRST_STAGE_LIMIT,
      final_candidate_limit=FINAL_CANDIDATE_LIMIT,
    )

    return jsonify({
      "query": query_text,
      "prepared_query": {
        "domain": prepared_query.get("domain"),
        "corrected_query_text": prepared_query.get("corrected_query_text"),
        "characteristics": prepared_query.get("characteristics", {}),
      },
      "count": len(results),
      "results": [make_result_card(row) for row in results],
    })
  except Exception as error:
    return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000, debug=True)
