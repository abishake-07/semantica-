from typing import List, Dict
import numpy as np
from .llm_adapters import get_llm_rerank_scores


class BaseReranker:
    def score(self, query: str, candidates: List[Dict]) -> List[float]:
        """Score candidates for the query and return a list of floats in the same order."""
        raise NotImplementedError()


class CosineReranker(BaseReranker):
    def __init__(self, embedder):
        self.embedder = embedder

    def score(self, query: str, candidates: List[Dict]) -> List[float]:
        texts = [c.get("title", "") + "\n" + c.get("description", "") for c in candidates]
        emb_texts = self.embedder.embed_texts(texts)
        q_emb = self.embedder.embed_texts([query])[0]
        # cosine
        qnorm = np.linalg.norm(q_emb)
        if qnorm == 0:
            qnorm = 1
        scores = (emb_texts @ q_emb) / (np.linalg.norm(emb_texts, axis=1) * qnorm)
        scores = np.nan_to_num(scores).tolist()
        return scores


class LLMReranker(BaseReranker):
    """Reranker that uses external LLM services (HuggingFace, OpenRouter) with cosine fallback."""
    
    def __init__(self, embedder=None, fallback_to_cosine=True):
        self.embedder = embedder
        self.fallback_to_cosine = fallback_to_cosine
        self.cosine_reranker = CosineReranker(embedder) if embedder and fallback_to_cosine else None
    
    def score(self, query: str, candidates: List[Dict]) -> List[float]:
        try:
            return get_llm_rerank_scores(query, candidates)
        except RuntimeError as e:
            if self.cosine_reranker:
                print(f"LLM reranking failed, falling back to cosine similarity: {e}")
                return self.cosine_reranker.score(query, candidates)
            else:
                raise e
