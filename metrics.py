import numpy as np
import json
from openai import OpenAI
import config

class RankingMetrics:
    def __init__(self, k=20):
        self.k = k

    def calculate_traditional(self, retrieved_ids, target_data):
        """Поддерживает и один ID (Тест 1), и список ID (Тест 2)."""
        retrieved = [str(x) for x in retrieved_ids[:self.k]]
        targets = [str(target_data)] if isinstance(target_data, (str, int)) else [str(x) for x in target_data]
        
        hits = [1 if rid in targets else 0 for rid in retrieved]
        
        # NDCG
        dcg = sum([hits[i] / np.log2(i + 2) for i in range(len(hits))])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(retrieved), len(targets)))])
        
        return {
            "ndcg": round(dcg / idcg, 4) if idcg > 0 else 0,
            "hit": 1 if sum(hits) > 0 else 0,
            "rank": (hits.index(1) + 1) if 1 in hits else None
        }

class LLMJudge:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def evaluate(self, query, results):
        """Оценка релевантности через LLM по шкале 0-3."""
        if not results: return [0] * config.TOP_K
        
        formatted_results = "\n".join([f"{i+1}. {r['title']}" for i, r in enumerate(results)])
        prompt = (
            f"Запрос пользователя: '{query}'\n"
            f"Результаты поиска:\n{formatted_results}\n\n"
            "Оцени каждый товар от 0 до 3 (0-неверно, 3-точное попадание).\n"
            "Верни JSON формат: {\"scores\": [число, число, ...]}"
        )
        
        try:
            resp = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            scores = json.loads(resp.choices[0].message.content)["scores"]
            return scores[:len(results)]
        except Exception as e:
            print(f"Ошибка LLM: {e}")
            return [0] * len(results)

    def calculate_llm_ndcg(self, scores):
        if not scores or sum(scores) == 0: return 0.0
        dcg = sum([s / np.log2(i + 2) for i, s in enumerate(scores)])
        ideal_scores = sorted(scores, reverse=True)
        idcg = sum([s / np.log2(i + 2) for i, s in enumerate(ideal_scores)])
        return round(dcg / idcg, 4) if idcg > 0 else 0
