"""
Compare different embedding models for e-commerce search
Run this to see which model works best for your use case
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ModelComparison:
    """Compare different embedding models"""
    
    def __init__(self):
        self.models_to_test = {
            # Current default (basic)
            "all-MiniLM-L6-v2": {
                "size": "80MB",
                "dims": 384,
                "description": "Fast, basic model (current default)"
            },
            
            # Better search models
            "BAAI/bge-small-en-v1.5": {
                "size": "130MB", 
                "dims": 384,
                "description": "Search-optimized, better quality"
            },
            
            "intfloat/e5-small-v2": {
                "size": "130MB",
                "dims": 384, 
                "description": "Strong retrieval performance"
            },
            
            # High quality options
            "BAAI/bge-base-en-v1.5": {
                "size": "440MB",
                "dims": 768,
                "description": "Higher quality, more compute"
            },
            
            "sentence-transformers/all-mpnet-base-v2": {
                "size": "440MB",
                "dims": 768,
                "description": "Strong general-purpose model"
            }
        }
        
        self.test_queries = [
            "blue lamp for bedroom",
            "comfortable office chair", 
            "wooden dining table",
            "kitchen cutting tools",
            "modern lighting solutions"
        ]
        
        self.test_products = [
            "Modern Blue Table Lamp. Sleek contemporary table lamp with blue ceramic base",
            "Ergonomic Office Chair Black. Premium ergonomic office chair with lumbar support", 
            "Vintage Wooden Dining Table. Rustic farmhouse style dining table made from reclaimed oak",
            "Bamboo Cutting Board Large. Eco-friendly bamboo cutting board with juice groove",
            "LED Floor Lamp Adjustable. Modern LED floor lamp with adjustable brightness"
        ]
    
    def test_model_performance(self, model_name, limit_models=3):
        """Test a single model's performance"""
        
        print(f"\n🧪 Testing {model_name}")
        print("-" * 50)
        
        try:
            # Load model and measure time
            start_time = time.time()
            model = SentenceTransformer(model_name)
            load_time = time.time() - start_time
            
            # Test embedding generation
            start_time = time.time()
            query_embeddings = model.encode(self.test_queries)
            product_embeddings = model.encode(self.test_products)
            embed_time = time.time() - start_time
            
            # Test semantic similarity
            similarities = cosine_similarity(query_embeddings, product_embeddings)
            
            # Calculate relevance scores for first query
            query_idx = 0  # "blue lamp for bedroom"
            scores = similarities[query_idx]
            ranked_indices = np.argsort(scores)[::-1]
            
            info = self.models_to_test[model_name]
            
            print(f"📊 Model Info:")
            print(f"   Size: {info['size']}")
            print(f"   Dimensions: {info['dims']}")
            print(f"   Description: {info['description']}")
            
            print(f"\n⏱️  Performance:")
            print(f"   Load time: {load_time:.2f}s")
            print(f"   Embedding time: {embed_time:.3f}s")
            
            print(f"\n🎯 Results for '{self.test_queries[query_idx]}':")
            for i, idx in enumerate(ranked_indices[:3]):
                product_name = self.test_products[idx].split('.')[0]
                score = scores[idx]
                print(f"   {i+1}. {product_name} (score: {score:.4f})")
            
            return {
                'model': model_name,
                'load_time': load_time,
                'embed_time': embed_time,
                'top_score': scores[ranked_indices[0]],
                'score_range': scores.max() - scores.min(),
                'dims': info['dims']
            }
            
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return None
    
    def run_comparison(self, limit_models=3):
        """Run comparison of multiple models"""
        
        print("🔍 Embedding Model Comparison for E-commerce Search")
        print("=" * 60)
        
        results = []
        model_names = list(self.models_to_test.keys())[:limit_models]
        
        for model_name in model_names:
            result = self.test_model_performance(model_name)
            if result:
                results.append(result)
        
        # Summary comparison
        if results:
            print(f"\n📋 Summary Comparison")
            print("=" * 60)
            print(f"{'Model':<30} {'Load(s)':<8} {'Embed(s)':<9} {'Top Score':<10} {'Range':<8}")
            print("-" * 60)
            
            for r in results:
                model_short = r['model'].split('/')[-1][:25]
                print(f"{model_short:<30} {r['load_time']:<8.2f} {r['embed_time']:<9.3f} {r['top_score']:<10.4f} {r['score_range']:<8.4f}")
        
        return results
    
    def recommend_model(self, results):
        """Recommend best model based on results"""
        
        if not results:
            return "No successful model tests"
        
        print(f"\n🎯 Recommendations:")
        print("-" * 30)
        
        # Find best performing models
        best_quality = max(results, key=lambda x: x['top_score'])
        fastest = min(results, key=lambda x: x['embed_time'])
        
        print(f"🏆 Best Quality: {best_quality['model']}")
        print(f"   Top score: {best_quality['top_score']:.4f}")
        print(f"   Embedding time: {best_quality['embed_time']:.3f}s")
        
        print(f"\n⚡ Fastest: {fastest['model']}")
        print(f"   Embedding time: {fastest['embed_time']:.3f}s")
        print(f"   Top score: {fastest['top_score']:.4f}")
        
        # Overall recommendation
        print(f"\n💡 Recommendation for e-commerce:")
        if any('bge-small' in r['model'] for r in results):
            print("   Use BAAI/bge-small-en-v1.5 - Best balance of speed and quality")
        else:
            print("   Use the model with highest top score for your use case")

def quick_model_test():
    """Quick test of top 3 models"""
    
    comparison = ModelComparison()
    results = comparison.run_comparison(limit_models=3)
    comparison.recommend_model(results)
    
    print(f"\n🔧 To use a different model in your code:")
    print(f"   embedder = EmbeddingModel('BAAI/bge-small-en-v1.5')")
    print(f"   # or")
    print(f"   embedder = EmbeddingModel('intfloat/e5-small-v2')")

if __name__ == "__main__":
    quick_model_test()