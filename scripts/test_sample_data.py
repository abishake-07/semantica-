"""
Simple test script to verify the Semantica system works with sample data.
Run this to test the pipeline without the web interface.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path so we can import semantica
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from semantica.data_loader import load_wands_csv
from semantica.embeddings import EmbeddingModel
from semantica.retriever import FaissRetriever
from semantica.reranker import CosineReranker, LLMReranker

def test_semantic_search():
    print("🔍 Testing Semantica with Sample Data")
    print("=" * 50)
    
    # Load sample data
    print("📁 Loading sample products...")
    products = load_wands_csv("sample_products.csv")
    print(f"✅ Loaded {len(products)} products")
    
    # Prepare texts and IDs
    texts = [p.get("title", "") + "\n" + p.get("description", "") for p in products]
    ids = [p.get("product_id") for p in products]
    
    # Initialize embedding model
    print("🧠 Initializing embedding model...")
    embedder = EmbeddingModel()
    embeddings = embedder.embed_texts(texts)
    print(f"✅ Generated embeddings: {embeddings.shape}")
    
    # Build FAISS index
    print("🔎 Building search index...")
    retriever = FaissRetriever(dim=embeddings.shape[1])
    retriever.index_embeddings(embeddings, ids)
    print("✅ Search index ready")
    
    # Initialize reranker
    hf_token = os.getenv("HF_API_TOKEN")
    if hf_token:
        print("🤖 Using LLM reranker with HuggingFace")
        reranker = LLMReranker(embedder=embedder, fallback_to_cosine=True)
    else:
        print("📊 Using cosine similarity reranker")
        reranker = CosineReranker(embedder)
    
    # Test queries
    test_queries = [
        "comfortable chair for reading",
        "blue lighting for bedroom",
        "wooden furniture for dining room",
        "kitchen accessories",
        "cozy home decor"
    ]
    
    print("\n🎯 Testing Search Queries")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 30)
        
        # Retrieve candidates
        q_emb = embedder.embed_texts([query])[0]
        results = retriever.query(q_emb, k=5)
        
        # Map to products
        id_to_prod = {p["product_id"]: p for p in products}
        candidates = [id_to_prod[_id] for _id, score in results if _id in id_to_prod]
        
        # Rerank
        scores = reranker.score(query, candidates)
        
        # Sort by reranker score
        for c, s in zip(candidates, scores):
            c["final_score"] = s
        candidates = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
        
        # Show top 3 results
        for i, product in enumerate(candidates[:3], 1):
            print(f"{i}. {product.get('title', 'No title')} (Score: {product.get('final_score', 0):.3f})")
            desc = product.get('description', 'No description')
            print(f"   {desc[:80]}{'...' if len(desc) > 80 else ''}")
    
    print("\n✅ Test completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Run the Gradio app: python app/gradio_app.py")
    print("   2. Upload sample_products.csv in the web interface")
    print("   3. Try the same queries in the web UI")

if __name__ == "__main__":
    test_semantic_search()