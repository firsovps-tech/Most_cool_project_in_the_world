import numpy as np
from typing import List


class RankingMetrics:
    def __init__(self, k=20):
        self.k = k

    def calculate_traditional(self, retrieved_ids, target_data):
        """Метрика Ильнура: ndcg, hit, rank. Поддерживает один ID и список ID."""
        retrieved = [str(x) for x in retrieved_ids[:self.k]]
        targets = [str(target_data)] if isinstance(target_data, (str, int)) else [str(x) for x in target_data]

        hits = [1 if rid in targets else 0 for rid in retrieved]

        dcg = sum([hits[i] / np.log2(i + 2) for i in range(len(hits))])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(retrieved), len(targets)))])

        return {
            "ndcg": round(dcg / idcg, 4) if idcg > 0 else 0,
            "hit": 1 if sum(hits) > 0 else 0,
            "rank": (hits.index(1) + 1) if 1 in hits else None,
        }

    def calculate_llm(self, scores: List[int]):
        """Метрика Ильнура на основе LLM score 0..3."""
        scores = np.asarray(scores, dtype=float)[:self.k]

        dcg = 0.0
        if scores.size:
            dcg = np.sum(scores / np.log2(np.arange(2, scores.size + 2)))

        sorted_scores = np.sort(scores)[::-1]
        idcg = 0.0
        if sorted_scores.size:
            idcg = np.sum(sorted_scores / np.log2(np.arange(2, sorted_scores.size + 2)))

        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        relevant = sum(1 for s in scores[:5] if s > 0)
        precision_at_5 = relevant / 5.0

        return {
            "ndcg": round(ndcg, 4),
            "precision@5": round(precision_at_5, 4),
        }
