from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False))
