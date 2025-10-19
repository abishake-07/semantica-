import faiss
import numpy as np
from typing import List, Dict, Tuple


class FaissRetriever:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors
        self._ids = []

    def index_embeddings(self, embeddings: np.ndarray, ids: List[str]):
        # normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vecs = embeddings / norms
        self.index.add(vecs.astype('float32'))
        self._ids.extend(ids)

    def query(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        qe = query_embedding.reshape(1, -1)
        qnorm = np.linalg.norm(qe, axis=1, keepdims=True)
        if qnorm == 0:
            qnorm = 1
        qe = qe / qnorm
        D, I = self.index.search(qe.astype('float32'), k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            results.append((self._ids[idx], float(score)))
        return results
