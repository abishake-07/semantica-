# Semantica - E-commerce Semantic Search

A complete e-commerce semantic search system with LLM integration and evaluation metrics.

## 🏗️ Repository Structure

```
Semantica/
├── semantica/              # Core library modules
│   ├── __init__.py        # Package initialization
│   ├── data_loader.py     # Data loading utilities
│   ├── embeddings.py      # Embedding models
│   ├── retriever.py       # Vector search (FAISS)
│   ├── reranker.py        # LLM reranking
│   ├── llm_adapters.py    # LLM integrations
│   └── metrics.py         # Evaluation metrics
├── scripts/               # Executable scripts
│   ├── cli.py            # Command-line interface
│   ├── finetune_clean.py # Model fine-tuning
│   ├── compare_models.py # Model comparison
│   └── test_*.py         # Testing scripts
├── app/                   # Web applications
│   └── gradio_app.py     # Gradio web interface
├── data/                  # Data files
│   ├── sample_products.csv   # Sample product data
│   └── dataset/          # WANDS dataset
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── models/                # Saved models
├── .venv/                 # Virtual environment
├── requirements.txt       # Dependencies
├── README.md             # This file
└── .gitignore            # Git ignore rules
```
## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Semantic Search
```bash
# From scripts directory
cd scripts
python cli.py --csv ../data/sample_products.csv --query "blue lamp for bedroom" --k 5
```

### 3. Compare Models
```bash
cd scripts
python compare_models.py
```

### 4. Fine-tune Models
```bash
cd scripts
python finetune_clean.py
```

### 5. Web Interface
```bash
cd app
python gradio_app.py
```

## 📊 Features

- **Semantic Search**: Using sentence-transformers for embeddings
- **Vector Retrieval**: FAISS for efficient similarity search
- **LLM Reranking**: Integration with HuggingFace models
- **Model Comparison**: Compare different embedding models
- **Fine-tuning**: Adapt models to your specific domain
- **Evaluation Metrics**: Precision@k, Recall@k, MRR, nDCG
- **Web Interface**: Gradio-based demo application

## 🛠️ Development

This is now structured as a standard Python repository (not a package) for easy development and experimentation.

### Running Tests
```bash
cd tests
python -m pytest
```

### Adding New Models
Add new embedding models in `semantica/embeddings.py` and new LLM adapters in `semantica/llm_adapters.py`.

## 📁 Data Format

Expected CSV format for products:
```csv
product_id,title,description,attributes
1,"Modern Blue Table Lamp","Contemporary table lamp with blue base","color:blue,type:lamp"
```

## 🔧 Configuration

Create a `.env` file for API keys:
```env
HUGGINGFACE_API_TOKEN=your_token_here
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
