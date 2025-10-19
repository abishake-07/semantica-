from semantica.metrics import precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k


def test_precision_recall():
    relevant = ["a", "b", "c"]
    retrieved = ["b", "x", "a", "y"]
    assert precision_at_k(relevant, retrieved, 2) == 0.5
    assert recall_at_k(relevant, retrieved, 3) == 2 / 3


def test_mrr():
    all_rel = [["a"], ["b"], ["c"]]
    all_ret = [["x", "a"], ["b"], ["c"]]
    assert mean_reciprocal_rank(all_rel, all_ret) == (1/2 + 1 + 1) / 3


def test_ndcg():
    relevant = ["a", "b"]
    retrieved = ["b", "a", "x"]
    val = ndcg_at_k(relevant, retrieved, 3)
    assert val > 0
