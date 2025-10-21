from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize embedding model with better defaults:
        
        Recommended for e-commerce:
        - "BAAI/bge-small-en-v1.5" (384 dim) - Best balance, search-optimized
        - "BAAI/bge-base-en-v1.5" (768 dim) - Higher quality, more compute
        - "intfloat/e5-small-v2" (384 dim) - Strong retrieval performance
        
        Legacy option:
        - "all-MiniLM-L6-v2" (384 dim) - Fast but basic
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False))
