"""
Training monitor for fine-tuning progress
Run this in a separate terminal to monitor training
"""

import os
import time
import json
import glob
from datetime import datetime

def monitor_training():
    """Monitor fine-tuning progress in real-time"""
    
    print("📊 Fine-Tuning Progress Monitor")
    print("=" * 40)
    print("Watching for training logs...")
    
    models_dir = "../models"
    start_time = datetime.now()
    
    while True:
        try:
            # Look for training log files
            log_files = glob.glob(f"{models_dir}/*training_log.json")
            
            if log_files:
                # Get the most recent log file
                latest_log = max(log_files, key=os.path.getmtime)
                
                try:
                    with open(latest_log, 'r') as f:
                        log_data = json.load(f)
                    
                    # Clear screen (Windows/Unix compatible)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    print("📊 Fine-Tuning Progress Monitor")
                    print("=" * 50)
                    print(f"📁 Model: {log_data.get('model', 'Unknown')}")
                    print(f"🕐 Started: {log_data.get('start_time', 'Unknown')}")
                    
                    if 'dataset_stats' in log_data:
                        stats = log_data['dataset_stats']
                        print(f"📊 Training examples: {stats.get('training_examples', 0):,}")
                        print(f"📊 Validation examples: {stats.get('validation_examples', 0):,}")
                    
                    if 'epochs' in log_data and log_data['epochs']:
                        epochs = log_data['epochs']
                        current_epoch = len(epochs)
                        total_epochs = 50  # Assuming 50 epochs
                        
                        print(f"\n📈 Progress: {current_epoch}/{total_epochs} epochs")
                        print(f"🔄 Progress bar: {'█' * (current_epoch * 20 // total_epochs):<20} {current_epoch * 100 // total_epochs}%")
                        
                        if epochs:
                            latest_epoch = epochs[-1]
                            print(f"📉 Latest loss: {latest_epoch.get('avg_loss', 0):.4f}")
                            print(f"⏱️  Latest epoch time: {latest_epoch.get('time_seconds', 0):.1f}s")
                            
                            # Estimate remaining time
                            if current_epoch > 1:
                                avg_epoch_time = sum(e.get('time_seconds', 0) for e in epochs) / len(epochs)
                                remaining_epochs = total_epochs - current_epoch
                                eta_seconds = remaining_epochs * avg_epoch_time
                                eta_minutes = eta_seconds / 60
                                print(f"⏰ ETA: {eta_minutes:.1f} minutes")
                        
                        # Show loss trend (last 5 epochs)
                        if len(epochs) >= 2:
                            print(f"\n📊 Recent Loss Trend:")
                            recent_epochs = epochs[-5:] if len(epochs) >= 5 else epochs
                            for epoch in recent_epochs:
                                loss = epoch.get('avg_loss', 0)
                                epoch_num = epoch.get('epoch', 0)
                                bar_length = int(loss * 100) if loss < 1 else 50
                                print(f"   Epoch {epoch_num:2d}: {'█' * min(bar_length, 50):<50} {loss:.4f}")
                    
                    # Show final results if available
                    if 'final_metrics' in log_data:
                        metrics = log_data['final_metrics']
                        print(f"\n🎯 Training Complete!")
                        print(f"📈 Average improvement: {metrics.get('average_improvement', 0):+.4f}")
                        total_time = log_data.get('training_time_minutes', 0)
                        print(f"⏱️  Total time: {total_time:.1f} minutes")
                        break
                    
                except json.JSONDecodeError:
                    print("⚠️  Log file is being written to, waiting...")
                except Exception as e:
                    print(f"❌ Error reading log: {e}")
            
            else:
                elapsed = (datetime.now() - start_time).seconds
                print(f"⏳ Waiting for training to start... ({elapsed}s)", end='\r')
            
            time.sleep(5)  # Update every 5 seconds
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_training()