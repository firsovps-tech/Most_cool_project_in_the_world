from data_loader import DataLoader
from search_interface import SearchInterface
from metrics import RankingMetrics, LLMJudge
import config

def main():
    print("=== Запуск системы валидации поиска ===\n")
    loader = DataLoader()
    search = SearchInterface(loader.df)
    metrics = RankingMetrics(config.TOP_K)
    judge = LLMJudge()

    # --- ТЕСТ 1: Точечный поиск ---
    t1_queries = loader.load_type1()
    print(f"Тест 1: Обработка {len(t1_queries)} запросов...")
    
    for q in t1_queries[:40]:
        results = search.get_results(q['query_text'])
        res_ids = [r['id'] for r in results]
        
        m = metrics.calculate_traditional(res_ids, q['target_product_id'])
        
        l_scores = judge.evaluate(q['query_text'], results)
        l_ndcg = judge.calculate_llm_ndcg(l_scores)
        
        print(f"Q: {q['query_text'][:30]} | Hit: {m['hit']} | NDCG: {m['ndcg']} | LLM-NDCG: {l_ndcg}")

    # --- ТЕСТ 2: Золотой стандарт ---
    t2_queries = loader.load_type2()
    if t2_queries:
        print(f"\nТест 2: Сравнение с Золотым Топ-20 ({len(t2_queries)} запросов)...")
        for q in t2_queries:
            results = search.get_results(q['query_text'])
            res_ids = [r['id'] for r in results]
            m = metrics.calculate_traditional(res_ids, q['gold_ids'])
            print(f"Q: {q['query_text']} | Gold-NDCG: {m['ndcg']}")
    else:
        print("\nТест 2 пропущен: создайте файл gold_top20_queries.json")

if __name__ == "__main__":
    main()
