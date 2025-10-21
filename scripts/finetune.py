#!/usr/bin/env python3
"""
Minimalistic fine-tuning script with CLI flags
Supports various models, epochs, dataset sizes, and evaluation
"""

import argparse
import pandas as pd
import torch
import os
import sys
import time
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

def load_dataset(dataset_path, max_samples=None, chunk_size=5000):
    """Load WANDS dataset efficiently with better ID matching"""
    
    print(f"📊 Loading dataset from {dataset_path}")
    
    # Load queries and labels first
    queries_df = pd.read_csv(f"{dataset_path}/query.csv", sep='\t')
    labels_df = pd.read_csv(f"{dataset_path}/label.csv", sep='\t')
    
    print(f"📋 Dataset info:")
    print(f"   Queries: {len(queries_df)} (IDs: {queries_df['query_id'].min()}-{queries_df['query_id'].max()})")
    print(f"   Labels: {len(labels_df)} (IDs: {labels_df['query_id'].min()}-{labels_df['query_id'].max()})")
    
    # Sample labels if specified
    if max_samples and max_samples < len(labels_df):
        labels_df = labels_df.sample(n=max_samples, random_state=42)
        print(f"📊 Sampled to {len(labels_df)} labels")
    
    # Get needed product IDs from the labels
    needed_product_ids = set(labels_df['product_id'].unique())
    print(f"📦 Need {len(needed_product_ids)} unique products")
    
    # Load products more intelligently
    try:
        print(f"📦 Loading products...")
        
        if chunk_size >= 50000:  # For larger datasets, load all products
            print("   Loading all products for complete coverage...")
            products_df = pd.read_csv(f"{dataset_path}/product.csv", sep='\t')
        else:
            # For smaller tests, load a reasonable chunk
            products_df = pd.read_csv(f"{dataset_path}/product.csv", sep='\t', nrows=chunk_size)
        
        print(f"   Loaded {len(products_df)} products (IDs: {products_df['product_id'].min()}-{products_df['product_id'].max()})")
        
        # Debug: Check column names and sample data
        print(f"   Product columns: {list(products_df.columns)}")
        print(f"   Sample product data:")
        for col in products_df.columns:
            sample_val = products_df[col].iloc[0] if not products_df.empty else "N/A"
            print(f"      {col}: {str(sample_val)[:50]}...")
        
        # Check overlap
        available_products = set(products_df['product_id'].values)
        overlap = needed_product_ids.intersection(available_products)
        print(f"   Product ID overlap: {len(overlap)}/{len(needed_product_ids)} ({len(overlap)/len(needed_product_ids)*100:.1f}%)")
        
        if len(overlap) < len(needed_product_ids) * 0.1:  # Less than 10% overlap
            print("⚠️  Poor product coverage detected - this may cause training issues")
            
    except Exception as e:
        print(f"⚠️  Error loading products ({e}), using sample products instead")
        products_df = pd.read_csv("../data/sample_products.csv")
        # Rename columns to match WANDS format
        products_df = products_df.rename(columns={
            'product_id': 'product_id',
            'title': 'product_title', 
            'description': 'product_description'
        })
    
    print(f"✅ Final dataset: {len(queries_df)} queries, {len(labels_df)} labels, {len(products_df)} products")
    return queries_df, labels_df, products_df

def create_training_examples(queries_df, labels_df, products_df):
    """Create training examples from dataset"""
    
    print("🔄 Creating training examples...")
    print(f"   Available queries: {len(queries_df)} (IDs: {queries_df['query_id'].min()}-{queries_df['query_id'].max()})")
    print(f"   Available products: {len(products_df)} (IDs: {products_df['product_id'].min()}-{products_df['product_id'].max()})")
    print(f"   Labels to process: {len(labels_df)}")
    
    examples = []
    skipped = 0
    query_not_found = 0
    product_not_found = 0
    empty_text = 0
    invalid_label = 0
    
    # Convert to sets for faster lookup
    available_queries = set(queries_df['query_id'].values)
    available_products = set(products_df['product_id'].values)
    
    print(f"   Checking data overlap...")
    valid_labels = labels_df[
        labels_df['query_id'].isin(available_queries) & 
        labels_df['product_id'].isin(available_products)
    ]
    print(f"   Valid labels (with matching IDs): {len(valid_labels)}")
    
    if len(valid_labels) == 0:
        print("❌ No matching query-product pairs found!")
        print("🔍 Debugging info:")
        print(f"   Sample label query IDs: {labels_df['query_id'].head().tolist()}")
        print(f"   Sample available query IDs: {queries_df['query_id'].head().tolist()}")
        print(f"   Sample label product IDs: {labels_df['product_id'].head().tolist()}")
        print(f"   Sample available product IDs: {products_df['product_id'].head().tolist()}")
        return []
    
    # Process valid labels only
    for _, label_row in valid_labels.iterrows():
        query_id = label_row['query_id']
        product_id = label_row['product_id'] 
        label = label_row['label']
        
        # Get query (should exist since we filtered)
        query_row = queries_df[queries_df['query_id'] == query_id]
        if query_row.empty:
            query_not_found += 1
            continue
        query_text = str(query_row.iloc[0]['query']).strip()
        
        # Get product (should exist since we filtered)
        product_row = products_df[products_df['product_id'] == product_id]
        if product_row.empty:
            product_not_found += 1
            continue
            
        product_data = product_row.iloc[0]
        
        # Handle different possible column names for WANDS dataset
        title = ""
        desc = ""
        
        # WANDS dataset uses 'product_name' and 'product_description'
        title_cols = ['product_name', 'product_title', 'title', 'name']
        desc_cols = ['product_description', 'product_descripti', 'description', 'product_desc', 'desc', 'product_features']
        
        for col in title_cols:
            if col in product_data and pd.notna(product_data[col]):
                title = str(product_data[col]).strip()
                break
                
        for col in desc_cols:
            if col in product_data and pd.notna(product_data[col]):
                desc = str(product_data[col])[:200].strip()  # Limit length
                break
        
        # If no description, try using product features or other columns
        if not desc:
            for col in ['product_features', 'category hierarchy']:
                if col in product_data and pd.notna(product_data[col]):
                    desc = str(product_data[col])[:200].strip()
                    break
        
        # If still no title, try first text column
        if not title:
            for col in product_data.index:
                if col not in ['product_id', 'rating_count', 'average_rating', 'review_count'] and pd.notna(product_data[col]):
                    val = str(product_data[col]).strip()
                    if len(val) > 5:  # Reasonable text length
                        title = val[:100]  # Use as title
                        break
        
        if not query_text or not title:
            empty_text += 1
            if len(examples) < 5:  # Debug first few failures
                print(f"   DEBUG empty text: query='{query_text}', title='{title}', product_cols={list(product_data.index)}")
            continue
        
        product_text = f"{title}. {desc}".strip() if desc else title
        
        # Convert labels to scores
        score_map = {'Exact': 1.0, 'Partial': 0.7, 'Irrelevant': 0.1}
        if label not in score_map:
            invalid_label += 1
            continue
            
        examples.append(InputExample(
            texts=[query_text, product_text],
            label=score_map[label]
        ))
        
        # Progress update
        if len(examples) % 1000 == 0:
            print(f"   Created {len(examples)} examples so far...")
    
    print(f"✅ Created {len(examples)} examples")
    print(f"   Skipped breakdown:")
    print(f"   - Query not found: {query_not_found}")
    print(f"   - Product not found: {product_not_found}") 
    print(f"   - Empty text: {empty_text}")
    print(f"   - Invalid label: {invalid_label}")
    
    return examples

def finetune_model(model_name, examples, epochs, batch_size, learning_rate, output_dir):
    """Fine-tune the model"""
    
    print(f"🔥 Fine-tuning {model_name} for {epochs} epochs...")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Prepare training
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss_fn = losses.CosineSimilarityLoss(model)
    
    # Output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/{model_name.replace('/', '_')}_finetuned_{timestamp}"
    
    start_time = time.time()
    
    try:
        # Use sentence-transformers fit method (now that datasets is available)
        print("✅ Using sentence-transformers fit method with datasets library")
        model.fit(
            train_objectives=[(dataloader, loss_fn)],
            epochs=epochs,
            warmup_steps=int(len(dataloader) * 0.1),
            output_path=output_path,
            show_progress_bar=True,
            save_best_model=True
        )
        
    except Exception as e:
        print(f"⚠️  Fit method failed ({e}), using manual training loop")
        
        # Manual training loop - process InputExamples properly
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_loss = 0
            print(f"📈 Epoch {epoch + 1}/{epochs}")
            
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Process InputExamples into features
                sentence_features = []
                labels = []
                
                for example in batch:
                    # Convert InputExample to features
                    texts = example.texts
                    features = [model.tokenize(text) for text in texts]
                    sentence_features.extend(features)
                    labels.append(example.label)
                
                # Convert labels to tensor
                labels = torch.tensor(labels, dtype=torch.float)
                
                # Compute loss
                loss = loss_fn(sentence_features, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"  Average loss: {avg_loss:.4f}")
        
        model.save(output_path)
    
    training_time = time.time() - start_time
    print(f"✅ Training complete in {training_time/60:.1f}min - {output_path}")
    
    return output_path

def evaluate_model(original_model_name, finetuned_path, test_queries=None):
    """Quick evaluation of fine-tuned vs original model"""
    
    if not test_queries:
        test_queries = [
            "blue lamp for bedroom",
            "office chair comfortable", 
            "wooden dining table",
            "kitchen cutting board"
        ]
    
    print("🔬 Evaluating models...")
    
    # Load models
    original = SentenceTransformer(original_model_name)
    finetuned = SentenceTransformer(finetuned_path)
    
    # Load test products
    try:
        test_df = pd.read_csv("../data/sample_products.csv")
        products = [f"{row['title']}. {row['description']}" for _, row in test_df.iterrows()]
    except:
        products = ["Sample product for testing"]
    
    # Compare models
    improvements = []
    for query in test_queries:
        # Original scores
        orig_emb = original.encode([query])
        prod_embs = original.encode(products)
        orig_scores = cosine_similarity(orig_emb, prod_embs)[0]
        orig_top = orig_scores.max()
        
        # Fine-tuned scores  
        ft_emb = finetuned.encode([query])
        prod_embs_ft = finetuned.encode(products)
        ft_scores = cosine_similarity(ft_emb, prod_embs_ft)[0]
        ft_top = ft_scores.max()
        
        improvement = ft_top - orig_top
        improvements.append(improvement)
        
        print(f"📊 '{query}': {orig_top:.4f} → {ft_top:.4f} ({improvement:+.4f})")
    
    avg_improvement = sum(improvements) / len(improvements)
    print(f"🎯 Average improvement: {avg_improvement:+.4f}")
    
    return avg_improvement

def main():
    parser = argparse.ArgumentParser(description="Fine-tune embedding models with CLI flags")
    
    # Model options
    parser.add_argument("--model", default="all-MiniLM-L6-v2", 
                       help="Model to fine-tune (default: all-MiniLM-L6-v2)")
    
    # Dataset options
    parser.add_argument("--dataset", default="../data/dataset",
                       help="Dataset path (default: ../data/dataset)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max training samples (default: all)")
    parser.add_argument("--chunk-size", type=int, default=5000,
                       help="Product chunk size (default: 5000)")
    
    # Training options  
    parser.add_argument("--epochs", type=int, default=1,
                       help="Training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate (default: 2e-5)")
    
    # Output options
    parser.add_argument("--output-dir", default="../models",
                       help="Output directory (default: ../models)")
    parser.add_argument("--no-eval", action="store_true",
                       help="Skip evaluation")
    
    # Special modes
    parser.add_argument("--full-dataset", action="store_true",
                       help="Use entire dataset (50 epochs)")
    parser.add_argument("--half-dataset", action="store_true",
                       help="Use 50% of dataset (50 epochs)")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Quick test (100 samples, 1 epoch)")
    
    args = parser.parse_args()
    
    # Apply special modes
    if args.full_dataset:
        args.epochs = 50
        args.max_samples = None
        args.chunk_size = 50000
        args.batch_size = 32  # Increase for faster training
        print("🎯 FULL DATASET MODE: 50 epochs, entire dataset")
        print("⚠️  WARNING: This will take 8-20 hours to complete!")
        print("💡 Consider using --max-samples 10000 for faster training")
        
    elif args.half_dataset:
        args.epochs = 50
        args.max_samples = 116724  # ~50% of 233,448 labels
        args.chunk_size = 25000
        args.batch_size = 32
        print("🎯 HALF DATASET MODE: 50 epochs, 50% of dataset (~116K samples)")
        print("⚠️  WARNING: This will take 4-10 hours to complete!")
        
    elif args.quick_test:
        args.epochs = 1
        args.max_samples = 100
        args.chunk_size = 1000
        print("⚡ QUICK TEST MODE: 1 epoch, 100 samples")
    
    print(f"\n🚀 Fine-tuning Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Max samples: {args.max_samples or 'All'}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print()
    
    # Load dataset
    queries_df, labels_df, products_df = load_dataset(
        args.dataset, args.max_samples, args.chunk_size
    )
    
    # Create training examples
    examples = create_training_examples(queries_df, labels_df, products_df)
    
    if not examples:
        print("❌ No training examples created!")
        return
    
    # Fine-tune model
    model_path = finetune_model(
        args.model, examples, args.epochs, 
        args.batch_size, args.lr, args.output_dir
    )
    
    # Evaluate if requested
    if not args.no_eval:
        evaluate_model(args.model, model_path)
    
    print(f"\n🎉 Fine-tuning complete!")
    print(f"📁 Model saved: {model_path}")

if __name__ == "__main__":
    main()