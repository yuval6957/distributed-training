# simple_test.py - Minimal test with small dataset and beautiful progress bars

import torch
import torch.nn as nn
from distributed_training import GPUTrainer, CPULoader
from distributed_training.config import get_config

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 5)
        )
    
    def forward(self, x):
        return self.layers(x)

# Simple dataset
import numpy as np
from torch.utils.data import Dataset

class SimpleDataset(Dataset):  # Inherit from PyTorch Dataset
    def __init__(self):
        print("🎲 Creating test dataset with progress tracking...")
        self.X = np.random.randn(1000, 10).astype(np.float32)
        self.y = np.random.randint(0, 5, 1000).astype(np.int64)
        print("✅ Test dataset ready")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx:idx+1])[0]

def test_gpu_server():
    """Run this on GPU server"""
    print("🚀 Testing GPU Server with Progress Bars")
    
    config = get_config()
    network_config = config.get_network_config()
    
    model = SimpleModel()
    trainer = GPUTrainer(port=network_config['port'])
    trainer.setup_model(model)
    
    print(f"🌐 Using configuration: GPU port {network_config['port']}, expecting CPU from {network_config['cpu_host']}")
    trainer.start_training(cpu_host=network_config['cpu_host'])

def test_cpu_server():
    """Run this on CPU server"""
    print("🚀 Testing CPU Server with Progress Bars")
    
    config = get_config()
    network_config = config.get_network_config()
    
    dataset = SimpleDataset()
    loader = CPULoader(num_workers=2)
    loader.setup_dataset(dataset, batch_size=32)
    
    print(f"🌐 Using configuration: Connecting to GPU at {network_config['gpu_host']}:{network_config['port']}")
    loader.start_loading(gpu_host=network_config['gpu_host'], gpu_port=network_config['port'], epochs=3)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
        test_gpu_server()
    elif len(sys.argv) > 1 and sys.argv[1] == 'cpu':
        test_cpu_server()
    else:
        print("🎯 Beautiful Progress Bar Test")
        print("Usage:")
        print("  python simple_test.py cpu    # Run on CPU server")
        print("  python simple_test.py gpu    # Run on GPU server")
        print("\n💡 Configuration loaded from config.yaml and environment variables")