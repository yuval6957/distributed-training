# examples/train_cpu.py - Run this on your CPU server (e.g., 192.168.1.200)

import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from distributed_training import CPULoader, create_simple_dataset
from distributed_training.config import get_config

# Your data pipeline development - focus on this!
class YourDataset(torch.utils.data.Dataset):
    def __init__(self, size=100000, input_dim=100, num_classes=10):
        print(f"🗃️  Creating large dataset: {size:,} samples")
        print("💾 This simulates loading a dataset that requires high memory...")
        
        # Simulate loading large dataset that requires high memory
        self.X = np.random.randn(size, input_dim).astype(np.float32)
        self.y = np.random.randint(0, num_classes, size).astype(np.int64)
        
        # Add some pattern to make training meaningful
        # Features 0-10 have stronger signal for classes 0-4
        for i in range(min(num_classes//2, 5)):
            mask = self.y == i
            self.X[mask, i*2:(i*2)+10] += np.random.normal(2.0, 0.5, (mask.sum(), 10))
        
        print(f"✅ Dataset created: {len(self.X):,} samples")
        print(f"📊 Memory usage: ~{(self.X.nbytes + self.y.nbytes) / 1e6:.1f} MB")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Your custom preprocessing - develop your pipeline here!
def your_preprocessing(batch_data, batch_targets):
    """
    Custom preprocessing function - modify this for your needs!
    
    Args:
        batch_data: Tensor of shape (batch_size, features)
        batch_targets: Tensor of shape (batch_size,)
    
    Returns:
        Processed batch_data, batch_targets
    """
    # Example preprocessing steps:
    
    # 1. Normalization
    batch_data = (batch_data - batch_data.mean(dim=0)) / (batch_data.std(dim=0) + 1e-8)
    
    # 2. Data augmentation - add noise
    if np.random.random() > 0.5:  # 50% chance
        noise = torch.randn_like(batch_data) * 0.1
        batch_data = batch_data + noise
    
    # 3. Feature scaling
    batch_data = batch_data * 1.2
    
    # 4. You can add more preprocessing steps here:
    # - Feature selection
    # - Dimensionality reduction  
    # - Custom augmentations
    # - etc.
    
    return batch_data, batch_targets

if __name__ == "__main__":
    print("🚀 Starting CPU Data Loading Server with Beautiful Progress Bars")
    
    # 1. Create your dataset
    dataset = YourDataset(
        size=100000,    # Large dataset for high-memory server
        input_dim=100,
        num_classes=10
    )
    
    # Alternative: Load from file
    # dataset = "/path/to/your/data.csv"
    
    # 2. Setup data loader with config
    config = get_config()
    loader = CPULoader(num_workers=config.get('training.num_workers', 6))
    
    # 3. Configure data pipeline
    loader.setup_dataset(
        dataset=dataset,
        batch_size=config.get('training.batch_size', 64),
        shuffle=config.get('data.shuffle', True),
        preprocessing_fn=your_preprocessing,  # Your custom preprocessing
        val_split=0.2  # Use 20% for validation (optional)
    )
    
    # 4. Start distributed data loading with beautiful progress tracking
    print("🎯 Connecting to GPU server...")
    print("   You'll see beautiful progress bars showing:")
    print("   - 📊 Epoch progress across all training")
    print("   - 📤 Data transmission progress to GPU")
    print("   - ⚡ GPU processing statistics") 
    print("   - 🔄 Queue sizes and flow control")
    
    # Get network configuration
    network_config = config.get_network_config()
    
    print(f"🌐 Connecting to GPU server: {network_config['gpu_host']}:{network_config['port']}")
    print("💡 You can change these settings in config.yaml or .env file")
    
    # Optional: Configure early stopping
    early_stopping = {
        'monitor': 'val_loss',    # or 'val_accuracy'
        'patience': 5,            # stop after 5 epochs without improvement
        'min_delta': 0.001        # minimum change to count as improvement
    }
    
    loader.start_loading(
        gpu_host=network_config['gpu_host'],
        epochs=config.get('training.epochs', 5),
        gpu_port=network_config['port'],
        early_stopping=early_stopping  # Enable early stopping (optional)
    )