# Scripts Directory

## 📁 Essential Scripts (Minimalistic)

### Core Scripts:
- **`finetune.py`** - Single fine-tuning script with CLI flags
- **`cli.py`** - Command-line interface for semantic search
- **`compare_models.py`** - Model comparison and benchmarking
- **`monitor_training.py`** - Real-time training progress monitor

### Test Scripts:
- **`test_sample_data.py`** - Test sample data functionality

---

## 🚀 Quick Start

### 1. Fine-tune Model (Your Use Case):
```bash
# Full dataset, 50 epochs
python finetune.py --full-dataset

# Quick test
python finetune.py --quick-test

# Custom settings
python finetune.py --epochs 20 --max-samples 10000
```

### 2. Search Products:
```bash
python cli.py --csv ../data/sample_products.csv --query "blue lamp" --k 5
```

### 3. Compare Models:
```bash
python compare_models.py
```

### 4. Monitor Training:
```bash
# Run in separate terminal while training
python monitor_training.py
```

---

## 🔧 Configuration

All scripts use the parent directory structure:
- `../semantica/` - Core library
- `../data/` - Datasets and sample data  
- `../models/` - Saved models
- `../app/` - Web applications