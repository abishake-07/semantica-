import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from semantica.data_loader import load_wands_csv
from semantica.embeddings import EmbeddingModel
from semantica.retriever import FaissRetriever
from semantica.reranker import CosineReranker, LLMReranker

# Load environment variables from .env file
load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to WANDS-style CSV")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--use-llm", action="store_true", help="Use LLM reranker (requires HF_API_TOKEN)")
    args = parser.parse_args()

    products = load_wands_csv(args.csv)
    texts = [p.get("title", "") + "\n" + p.get("description", "") for p in products]
    ids = [p.get("product_id") for p in products]
    
    embedder = EmbeddingModel()
    emb = embedder.embed_texts(texts)
    retriever = FaissRetriever(dim=emb.shape[1])
    retriever.index_embeddings(emb, ids)
    
    # Choose reranker
    if args.use_llm:
        reranker = LLMReranker(embedder=embedder, fallback_to_cosine=True)
        print("Using LLM reranker with cosine fallback")
    else:
        reranker = CosineReranker(embedder)
        print("Using cosine similarity reranker")
    
    # Search and rerank
    q_emb = embedder.embed_texts([args.query])[0]
    results = retriever.query(q_emb, k=args.k)
    id_to_prod = {p['product_id']: p for p in products}
    candidates = [id_to_prod[_id] for _id, score in results if _id in id_to_prod]
    
    scores = reranker.score(args.query, candidates)
    for c, s in zip(candidates, scores):
        c["score"] = s
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    
    print(f"\nSearch results for: '{args.query}'\n")
    for i, c in enumerate(candidates):
        print(f"{i+1}. {c.get('title', '')} (score: {c.get('score', 0):.4f})")
        print(f"   {c.get('description', '')[:100]}...")
        print()


if __name__ == '__main__':
    main()
