# Quick Usage Guide for Minimalistic Fine-tuning

## 🎯 Single Fine-tuning Script: `scripts/finetune.py`

### Basic Usage Examples:

```bash
cd scripts

# Quick test (100 samples, 1 epoch)
python finetune.py --quick-test

# Full dataset training (50 epochs, entire dataset)  
python finetune.py --full-dataset

# Custom training
python finetune.py --epochs 10 --max-samples 5000 --batch-size 32

# Different model
python finetune.py --model "BAAI/bge-small-en-v1.5" --epochs 5

# Large scale training
python finetune.py --epochs 50 --max-samples 50000 --chunk-size 10000
```

### All Available Flags:

**Model Options:**
- `--model` - Model to fine-tune (default: all-MiniLM-L6-v2)

**Dataset Options:**
- `--dataset` - Dataset path (default: ../data/dataset)
- `--max-samples` - Max training samples (default: all)
- `--chunk-size` - Product chunk size (default: 5000)

**Training Options:**
- `--epochs` - Training epochs (default: 1)
- `--batch-size` - Batch size (default: 16) 
- `--lr` - Learning rate (default: 2e-5)

**Output Options:**
- `--output-dir` - Output directory (default: ../models)
- `--no-eval` - Skip evaluation

**Special Modes:**
- `--full-dataset` - Use entire dataset (50 epochs)
- `--quick-test` - Quick test (100 samples, 1 epoch)

### Your Requested 50 Epoch Training:

```bash
cd scripts
python finetune.py --full-dataset
```

This will:
- Use the entire WANDS dataset
- Train for 50 epochs
- Load 50,000 products (chunk-size)
- Save timestamped model in ../models/
- Evaluate performance vs original model

### Monitor Training Progress:

```bash
# In another terminal
cd scripts  
python monitor_training.py
```