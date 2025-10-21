import numpy as np
from typing import List


def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    topk = retrieved[:k]
    if len(topk) == 0:
        return 0.0
    return float(len([d for d in topk if d in relevant])) / len(topk)


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    topk = retrieved[:k]
    if len(relevant) == 0:
        return 0.0
    return float(len([d for d in topk if d in relevant])) / len(relevant)


def mean_reciprocal_rank(all_relevant: List[List[str]], all_retrieved: List[List[str]]) -> float:
    rr = []
    for relevant, retrieved in zip(all_relevant, all_retrieved):
        rank = 0
        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                rank = 1.0 / i
                break
        rr.append(rank)
    return float(np.mean(rr)) if rr else 0.0


def ndcg_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    def dcg(rel_list):
        return sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(rel_list))

    rels = [1 if d in relevant else 0 for d in retrieved[:k]]
    dcg_v = dcg(rels)
    ideal = sorted(rels, reverse=True)
    idcg_v = dcg(ideal)
    return float(dcg_v / idcg_v) if idcg_v > 0 else 0.0
