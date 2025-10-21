import gradio as gr
import os
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from semantica.data_loader import load_wands_csv
from semantica.embeddings import EmbeddingModel
from semantica.retriever import FaissRetriever
from semantica.reranker import CosineReranker, LLMReranker

# Load environment variables from .env file
load_dotenv()


class SemanticSearchApp:
    def __init__(self):
        self.products = []
        self.embedder = None
        self.retriever = None
        self.reranker = None
        
    def load_dataset(self, file_path, hf_token, use_llm):
        """Load dataset and initialize components."""
        if file_path is None:
            return "Please upload a CSV file", "", gr.update(interactive=False)
            
        try:
            # Read uploaded file - handle both file path string and file object
            if hasattr(file_path, 'name'):
                file_to_read = file_path.name
            else:
                file_to_read = file_path
                
            df = pd.read_csv(file_to_read)
            self.products = []
            for _, row in df.iterrows():
                self.products.append({
                    "product_id": str(row.get("product_id", "")),
                    "title": str(row.get("title", "")),
                    "description": str(row.get("description", "")),
                })
            
            # Initialize embedder and build index
            self.embedder = EmbeddingModel()
            texts = [p.get("title", "") + "\n" + p.get("description", "") for p in self.products]
            ids = [p.get("product_id") for p in self.products]
            
            embeddings = self.embedder.embed_texts(texts)
            self.retriever = FaissRetriever(dim=embeddings.shape[1])
            self.retriever.index_embeddings(embeddings, ids)
            
            # Configure reranker
            if use_llm and hf_token:
                os.environ["HF_API_TOKEN"] = hf_token
                self.reranker = LLMReranker(embedder=self.embedder, fallback_to_cosine=True)
                reranker_info = "Using LLM reranker with cosine fallback"
            else:
                self.reranker = CosineReranker(self.embedder)
                reranker_info = "Using cosine similarity reranker"
            
            status = f"✅ Loaded {len(self.products)} products successfully"
            return status, reranker_info, gr.update(interactive=True)
            
        except Exception as e:
            return f"❌ Error loading dataset: {str(e)}", "", gr.update(interactive=False)
    
    def search_products(self, query, k):
        """Search for products using the configured pipeline."""
        if not query or not self.retriever or not self.reranker:
            return "Please load a dataset and enter a search query."
        
        try:
            # Retrieve candidates
            q_emb = self.embedder.embed_texts([query])[0]
            results = self.retriever.query(q_emb, k=k)
            
            # Map IDs to products
            id_to_prod = {p["product_id"]: p for p in self.products}
            candidates = [id_to_prod[_id] for _id, score in results if _id in id_to_prod]
            
            if not candidates:
                return "No products found."
            
            # Rerank candidates
            scores = self.reranker.score(query, candidates)
            
            # Combine and sort results
            for c, s in zip(candidates, scores):
                c["score"] = s
            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
            
            # Format results
            results_text = f"## Search Results for: '{query}'\n\n"
            for i, c in enumerate(candidates[:k]):
                results_text += f"**{i+1}. {c.get('title', 'No title')}**\n"
                results_text += f"*Score: {c.get('score', 0):.4f}*\n\n"
                results_text += f"{c.get('description', 'No description')}\n\n"
                results_text += "---\n\n"
            
            return results_text
            
        except Exception as e:
            return f"❌ Search error: {str(e)}"


# Initialize the app
app = SemanticSearchApp()

# Create Gradio interface
with gr.Blocks(title="Semantica - E-commerce Semantic Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Semantica - E-commerce Semantic Search Demo")
    gr.Markdown("Upload a WANDS-style CSV file and search for products using semantic understanding.")
    
    with gr.Row():
        with gr.Column(scale=2):
            # File upload and configuration
            file_input = gr.File(
                label="Upload WANDS CSV (columns: product_id, title, description)",
                file_types=[".csv"]
            )
            
            with gr.Row():
                hf_token = gr.Textbox(
                    label="HuggingFace API Token (loaded from .env)",
                    placeholder="hf_...",
                    type="password",
                    value=os.getenv("HF_API_TOKEN", "")
                )
                use_llm = gr.Checkbox(label="Use LLM Reranker", value=bool(os.getenv("HF_API_TOKEN")))
            
            load_btn = gr.Button("Load Dataset", variant="primary")
            
        with gr.Column(scale=1):
            # Status displays
            status_text = gr.Textbox(label="Status", interactive=False)
            reranker_info = gr.Textbox(label="Reranker", interactive=False)
    
    # Search interface
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter your search query (e.g., 'blue lamp', 'comfortable chair')",
                interactive=False
            )
        with gr.Column(scale=1):
            k_slider = gr.Slider(
                minimum=1, maximum=20, value=10, step=1,
                label="Number of Results"
            )
    
    search_btn = gr.Button("Search", variant="secondary", interactive=False)
    
    # Results display
    results_output = gr.Markdown(label="Search Results")
    
    # Event handlers
    load_btn.click(
        fn=app.load_dataset,
        inputs=[file_input, hf_token, use_llm],
        outputs=[status_text, reranker_info, search_btn]
    ).then(
        fn=lambda: gr.update(interactive=True),
        outputs=[query_input]
    )
    
    search_btn.click(
        fn=app.search_products,
        inputs=[query_input, k_slider],
        outputs=[results_output]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["blue lamp"],
            ["comfortable office chair"],
            ["wooden dining table"],
            ["modern lighting"],
        ],
        inputs=[query_input],
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
