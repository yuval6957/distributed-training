# examples/validate_cpu.py - Run this on your CPU server for validation

import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from distributed_training import CPULoader
from distributed_training.config import get_config

# Your validation dataset
class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000, input_dim=100, num_classes=10):
        print(f"🔍 Creating validation dataset: {size:,} samples")
        
        # Create validation data with same pattern as training
        self.X = np.random.randn(size, input_dim).astype(np.float32)
        self.y = np.random.randint(0, num_classes, size).astype(np.int64)
        
        # Add same patterns as training data for meaningful validation
        for i in range(min(num_classes//2, 5)):
            mask = self.y == i
            self.X[mask, i*2:(i*2)+10] += np.random.normal(2.0, 0.5, (mask.sum(), 10))
        
        print(f"✅ Validation dataset created: {len(self.X):,} samples")
        print(f"📊 Memory usage: ~{(self.X.nbytes + self.y.nbytes) / 1e6:.1f} MB")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    print("🔍 Starting CPU Validation Client")
    
    # 1. Create validation dataset
    val_dataset = ValidationDataset(
        size=10000,     # Smaller dataset for validation
        input_dim=100,
        num_classes=10
    )
    
    # Alternative: Load validation data from file
    # val_dataset = "/path/to/your/validation_data.csv"
    
    # 2. Setup data loader
    config = get_config()
    loader = CPULoader(num_workers=config.get('validation.num_workers', 4))
    
    # 3. Configure validation dataset
    loader.setup_dataset(
        dataset=val_dataset,
        batch_size=config.get('validation.batch_size', 128),  # Larger batch for validation
        shuffle=False,  # Don't shuffle validation data
        preprocessing_fn=None  # Usually no preprocessing for validation
    )
    
    # 4. Connect to GPU server and run validation
    print("🌐 Connecting to GPU validation server...")
    print("   This will show validation progress with:")
    print("   - 🔍 Validation progress across all samples")
    print("   - 📊 Real-time loss and accuracy metrics")
    print("   - ⚡ Batch processing statistics")
    
    # Get network configuration
    network_config = config.get_network_config()
    
    print(f"🎯 Connecting to GPU server: {network_config['gpu_host']}:{network_config['port']}")
    print("💡 You can change these settings in config.yaml or .env file")
    
    try:
        # Run validation
        results = loader.run_validation(
            gpu_host=network_config['gpu_host'],
            val_dataset=val_dataset,
            batch_size=config.get('validation.batch_size', 128),
            gpu_port=network_config['port']
        )
        
        # Print detailed results
        print("\n" + "="*50)
        print("📊 VALIDATION RESULTS")
        print("="*50)
        print(f"🎯 Total samples: {results['total_samples']:,}")
        print(f"🔢 Total batches: {results['total_batches']:,}")
        print(f"📉 Average loss: {results['average_loss']:.4f}")
        print(f"🎯 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print("="*50)
        
        # Optionally save results
        # import json
        # with open('validation_results.json', 'w') as f:
        #     json.dump(results, f, indent=2)
        # print("💾 Results saved to validation_results.json")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        print("💡 Make sure validate_gpu.py is running on the GPU server")
        print("💡 Check network connectivity and firewall settings")