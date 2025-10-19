"""LLM adapters to call external services for reranking.

Environment variables (loaded from .env file):
- HF_API_TOKEN: HuggingFace API token
- OPENROUTER_API_KEY: OpenRouter API key
- OPENROUTER_URL: OpenRouter endpoint URL
"""
import os
import json
from typing import List, Dict, Optional
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class HuggingFaceReranker:
    """Reranker using HuggingFace Inference API with Llama3 or Dolly models."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", api_token: Optional[str] = None):
        self.model_name = model_name
        self.api_token = api_token or os.getenv("HF_API_TOKEN")
        if not self.api_token:
            raise ValueError("HuggingFace API token required. Set HF_API_TOKEN environment variable.")
        self.client = InferenceClient(model=model_name, token=self.api_token)
    
    def score_relevance(self, query: str, product_text: str) -> float:
        """Score a single product's relevance to the query using the LLM."""
        prompt = f"""Rate the relevance of this product to the search query on a scale of 0.0 to 1.0.

Query: {query}
Product: {product_text}

Relevance score (0.0-1.0):"""
        
        try:
            response = self.client.text_generation(
                prompt, 
                max_new_tokens=10,
                temperature=0.1,
                return_full_text=False
            )
            # Extract numeric score from response
            score_text = response.strip()
            # Try to parse the first number found
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
            if numbers:
                return min(max(float(numbers[0]), 0.0), 1.0)
            return 0.5  # fallback
        except Exception as e:
            print(f"HF API error: {e}")
            return 0.5  # fallback score
    
    def score_batch(self, query: str, candidates: List[Dict]) -> List[float]:
        """Score multiple candidates. Returns scores in same order as candidates."""
        scores = []
        for candidate in candidates:
            product_text = f"{candidate.get('title', '')} - {candidate.get('description', '')}"
            score = self.score_relevance(query, product_text)
            scores.append(score)
        return scores


def openrouter_rerank(query: str, candidates: List[Dict], api_key: str, router_url: str):
    """Example: call OpenRouter to get relevance scores. This is a placeholder illustrating payload shape.

    Returns list of floats corresponding to candidates.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "input": {
            "query": query,
            "candidates": [{"id": c.get("product_id"), "text": c.get("title") + "\n" + c.get("description")} for c in candidates]
        }
    }
    resp = requests.post(router_url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # This assumes the response contains scores in data['scores'] matching candidates order
    return data.get("scores", [0.0] * len(candidates))


def get_llm_rerank_scores(query: str, candidates: List[Dict]) -> List[float]:
    """Try configured providers in order; fallback to error if none are configured."""
    
    # Try HuggingFace first
    hf_token = os.getenv("HF_API_TOKEN")
    if hf_token:
        try:
            reranker = HuggingFaceReranker()
            return reranker.score_batch(query, candidates)
        except Exception as e:
            print(f"HuggingFace reranker failed: {e}")
    
    # OpenRouter fallback
    or_key = os.getenv("OPENROUTER_API_KEY")
    or_url = os.getenv("OPENROUTER_URL")
    if or_key and or_url:
        try:
            return openrouter_rerank(query, candidates, or_key, or_url)
        except Exception as e:
            print(f"OpenRouter reranker failed: {e}")

    raise RuntimeError("No LLM provider configured. Set HF_API_TOKEN or (OPENROUTER_API_KEY + OPENROUTER_URL) or use CosineReranker.")
