E-Commerce Semantic Search Optimization (Semantica)
===============================================

Project scaffold for building an LLM-augmented semantic search system for e-commerce.

What's included
- minimal Python package layout under `src/semantica`
- FAISS + sentence-transformers embedding wrapper
- LLM reranker interface (pluggable)
- Gradio demo app to run queries and show results
- evaluation metrics utilities

Quick start
1. Create a Python virtual environment and activate it.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

If you plan to run tests locally, also install pytest in your environment (included above). If `pytest` is not available in your PATH, run it via the Python -m entry:

```powershell
python -m pytest -q
```

3. Prepare dataset (WANDS) and point to its CSV in the Gradio interface.
4. Run the Gradio demo:

```powershell
python app/gradio_app.py
```

LLM integration
- **Environment File**: Create a `.env` file in the project root with your API tokens:
  ```
  HF_API_TOKEN=your_hf_token_here
  OPENROUTER_API_KEY=your_openrouter_key_here
  OPENROUTER_URL=your_openrouter_endpoint_here
  ```
- **HuggingFace**: The `HF_API_TOKEN` enables LLM reranking using models like Llama3 or Dolly.
- **OpenRouter**: Configure `OPENROUTER_API_KEY` and `OPENROUTER_URL` for OpenRouter endpoints.
- **Fallback**: If no LLM is configured, the system automatically falls back to cosine similarity reranking using sentence embeddings.

Example setup:
```powershell
# Create .env file with your token
echo "HF_API_TOKEN=your_hf_token_here" > .env
python app/gradio_app.py
```

Dataset
- Place the WANDS CSV file locally and use the Streamlit uploader or the CLI (`cli.py --csv path/to/wands.csv --query "lamp"`).


3. Prepare dataset (WANDS) and point to its CSV in `config.yml` or pass path to loader.
4. Run the Streamlit demo:

```powershell
streamlit run app/streamlit_app.py
```

Notes
- This is a scaffold. Replace LLM and dataset configs with real endpoints and data.
