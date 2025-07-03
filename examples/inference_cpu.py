# examples/inference_cpu.py - Run this on your CPU server for inference

import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from distributed_training import CPULoader
from distributed_training.config import get_config

# Your inference dataset (only features, no labels needed)
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, size=5000, input_dim=100):
        print(f"🔮 Creating inference dataset: {size:,} samples")
        
        # Create inference data - similar patterns to training/validation
        self.X = np.random.randn(size, input_dim).astype(np.float32)
        
        # Add some patterns to make inference meaningful
        # This simulates real-world data with some structure
        for i in range(5):
            start_idx = i * (size // 5)
            end_idx = (i + 1) * (size // 5)
            self.X[start_idx:end_idx, i*2:(i*2)+10] += np.random.normal(2.0, 0.5, (end_idx - start_idx, 10))
        
        print(f"✅ Inference dataset created: {len(self.X):,} samples")
        print(f"📊 Memory usage: ~{self.X.nbytes / 1e6:.1f} MB")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # For inference, we only return the features (no labels)
        return self.X[idx]

def save_predictions(results, output_path="predictions.json"):
    """Save inference results to file"""
    import json
    
    # Create a summary with key information
    summary = {
        'total_samples': results['total_samples'],
        'total_batches': results['total_batches'],
        'predictions': results['predictions'],
        'prediction_counts': {}
    }
    
    # Count predictions for each class
    unique_preds, counts = np.unique(results['predictions'], return_counts=True)
    for pred, count in zip(unique_preds, counts):
        summary['prediction_counts'][int(pred)] = int(count)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Predictions saved to {output_path}")
    print("📊 Prediction distribution:")
    for pred, count in summary['prediction_counts'].items():
        percentage = (count / results['total_samples']) * 100
        print(f"   Class {pred}: {count:,} samples ({percentage:.1f}%)")

if __name__ == "__main__":
    print("🔮 Starting CPU Inference Client")
    
    # 1. Create inference dataset
    inference_dataset = InferenceDataset(
        size=5000,      # Dataset for inference
        input_dim=100
    )
    
    # Alternative: Load inference data from file
    # For CSV: inference_dataset = "/path/to/your/inference_data.csv" 
    # For numpy: inference_dataset = np.load("/path/to/data.npy")
    
    # 2. Setup data loader
    config = get_config()
    loader = CPULoader(num_workers=config.get('inference.num_workers', 4))
    
    # 3. Configure inference dataset
    loader.setup_dataset(
        dataset=inference_dataset,
        batch_size=config.get('inference.batch_size', 256),  # Larger batch for inference
        shuffle=False,  # Don't shuffle inference data
        preprocessing_fn=None  # Usually no preprocessing for inference
    )
    
    # 4. Connect to GPU server and run inference
    print("🌐 Connecting to GPU inference server...")
    print("   This will show inference progress with:")
    print("   - 🔮 Inference progress across all samples")
    print("   - 📊 Real-time processing statistics")
    print("   - 🎯 Prediction results")
    
    # Get network configuration
    network_config = config.get_network_config()
    
    print(f"🎯 Connecting to GPU server: {network_config['gpu_host']}:{network_config['port']}")
    print("💡 You can change these settings in config.yaml or .env file")
    
    try:
        # Run inference with configurable options
        results = loader.run_inference(
            gpu_host=network_config['gpu_host'],
            inference_dataset=inference_dataset,
            batch_size=config.get('inference.batch_size', 256),
            top_k=3,  # Get top-3 predictions
            gpu_port=network_config['port'],
            return_details=False,  # Set to True for detailed batch results
            return_full_probabilities=False,  # CAUTION: Memory intensive for large datasets!
            return_raw_outputs=False,  # CAUTION: Memory intensive for large datasets!
            confidence_threshold=0.5  # Only consider predictions with >50% confidence
        )
        
        # Print summary results
        print("\n" + "="*50)
        print("🔮 INFERENCE RESULTS")
        print("="*50)
        print(f"🎯 Total samples: {results['total_samples']:,}")
        print(f"🔢 Total batches: {results['total_batches']:,}")
        print(f"📊 Predictions shape: {len(results['predictions'])}")
        print(f"🔢 Probabilities shape: {len(results['probabilities'])}")
        
        # Show sample predictions
        print(f"\n📋 Sample predictions (first 10):")
        for i in range(min(10, len(results['predictions']))):
            pred = results['predictions'][i]
            prob = max(results['probabilities'][i])
            print(f"   Sample {i}: Class {pred} (confidence: {prob:.3f})")
        
        print("="*50)
        
        # Save results to file
        save_predictions(results, "inference_results.json")
        
        # Optionally save detailed results with probabilities
        # import json
        # with open('detailed_inference_results.json', 'w') as f:
        #     json.dump(results, f, indent=2)
        # print("💾 Detailed results saved to detailed_inference_results.json")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print("💡 Make sure inference_gpu.py is running on the GPU server")
        print("💡 Check network connectivity and firewall settings")
        print("💡 Ensure the model is properly trained and loaded")