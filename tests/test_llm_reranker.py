import pytest
import os
from dotenv import load_dotenv
from semantica.reranker import LLMReranker, CosineReranker
from semantica.embeddings import EmbeddingModel

# Load environment variables from .env file
load_dotenv()


def test_llm_reranker_fallback():
    """Test that LLMReranker falls back to cosine when no LLM is configured."""
    # Ensure no HF token is set for this test
    old_token = os.environ.get("HF_API_TOKEN")
    if "HF_API_TOKEN" in os.environ:
        del os.environ["HF_API_TOKEN"]
    
    try:
        embedder = EmbeddingModel()
        reranker = LLMReranker(embedder=embedder, fallback_to_cosine=True)
        
        candidates = [
            {"product_id": "1", "title": "Blue lamp", "description": "A beautiful blue table lamp"},
            {"product_id": "2", "title": "Red chair", "description": "Comfortable red office chair"}
        ]
        
        scores = reranker.score("blue light", candidates)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        
    finally:
        # Restore token if it existed
        if old_token:
            os.environ["HF_API_TOKEN"] = old_token


def test_llm_reranker_no_fallback():
    """Test that LLMReranker raises error when no LLM configured and no fallback."""
    old_token = os.environ.get("HF_API_TOKEN")
    if "HF_API_TOKEN" in os.environ:
        del os.environ["HF_API_TOKEN"]
    
    try:
        reranker = LLMReranker(fallback_to_cosine=False)
        candidates = [{"product_id": "1", "title": "Test", "description": "Test product"}]
        
        with pytest.raises(RuntimeError):
            reranker.score("test", candidates)
            
    finally:
        if old_token:
            os.environ["HF_API_TOKEN"] = old_token