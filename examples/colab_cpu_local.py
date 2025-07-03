# examples/colab_cpu_local.py
"""
Local CPU Script for Google Colab GPU Training

This script runs on your LOCAL computer to send data to Colab GPU.
It connects to your Colab GPU instance through ngrok tunnel.

Usage:
1. Run the colab_gpu_example.py in Google Colab first
2. Copy the tunnel host and port from Colab output
3. Update COLAB_GPU_HOST and COLAB_GPU_PORT below
4. Run this script on your local machine
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import sys
import os

# Add the distributed_training package to path
# Adjust this path to where your project is located
sys.path.append('/path/to/your/distributed-training-project')  # UPDATE THIS PATH

from distributed_training.cpu_loader import CPULoader

# =============================================================================
# COLAB CONNECTION SETTINGS - UPDATE THESE FROM COLAB OUTPUT
# =============================================================================

# Copy these values from your Colab notebook output
COLAB_GPU_HOST = "2.tcp.ngrok.io"   # TCP tunnel host
COLAB_GPU_PORT = 15262              # TCP tunnel port

# =============================================================================
# YOUR LOCAL DATASET
# =============================================================================

class LocalDataset(Dataset):
    """
    Your local dataset - replace this with your actual data
    This example creates synthetic data, but you can load from files
    """
    
    def __init__(self, data_size=50000, input_size=100, num_classes=10):
        print(f"📊 Creating local dataset with {data_size:,} samples...")
        
        # Generate synthetic data - REPLACE THIS WITH YOUR DATA
        np.random.seed(42)  # For reproducibility
        self.X = np.random.randn(data_size, input_size).astype(np.float32)
        self.y = np.random.randint(0, num_classes, data_size).astype(np.int64)
        
        print(f"✅ Dataset created: {self.X.shape} features, {len(np.unique(self.y))} classes")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx:idx+1])[0]


# Alternative: Load from files
class FileDataset(Dataset):
    """
    Example of loading data from files
    """
    
    def __init__(self, csv_path):
        import pandas as pd
        
        print(f"📄 Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Assume last column is target, rest are features
        self.X = df.iloc[:, :-1].values.astype(np.float32)
        self.y = df.iloc[:, -1].values.astype(np.int64)
        
        print(f"✅ Loaded {len(self.X):,} samples from CSV")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx:idx+1])[0]


def custom_preprocessing(batch_data, batch_targets):
    """
    Optional preprocessing function
    Apply any custom transformations to your data here
    """
    # Example: Add noise for data augmentation
    if np.random.random() < 0.3:  # 30% chance
        noise = torch.randn_like(batch_data) * 0.01
        batch_data = batch_data + noise
    
    # Example: Normalize data
    batch_data = (batch_data - batch_data.mean()) / (batch_data.std() + 1e-8)
    
    return batch_data, batch_targets


def main():
    """Main function to run local CPU training"""
    
    print("🏠 Starting Local CPU Training for Colab GPU")
    print("=" * 60)
    
    # Check connection settings
    if COLAB_GPU_HOST == "abc123.ngrok.io":
        print("❌ ERROR: You need to update COLAB_GPU_HOST and COLAB_GPU_PORT!")
        print("   1. Run colab_gpu_example.py in Google Colab")
        print("   2. Copy the tunnel host and port from Colab output")
        print("   3. Update the values at the top of this file")
        return
    
    print(f"🌐 Connecting to Colab GPU at: {COLAB_GPU_HOST}:{COLAB_GPU_PORT}")
    
    # Create CPU loader
    loader = CPULoader(num_workers=4)  # Adjust based on your CPU cores
    
    # Setup dataset
    print("\n📊 Setting up dataset...")
    
    # Option 1: Use synthetic data (for testing)
    dataset = LocalDataset(data_size=50000, input_size=100, num_classes=10)
    
    # Option 2: Load from CSV file
    # dataset = FileDataset('/path/to/your/data.csv')
    
    # Option 3: Use numpy arrays directly
    # X = np.random.randn(10000, 100).astype(np.float32)
    # y = np.random.randint(0, 10, 10000).astype(np.int64)
    # dataset = (X, y)
    
    # Setup the loader
    loader.setup_dataset(
        dataset=dataset,
        batch_size=64,          # Adjust based on your GPU memory
        shuffle=True,
        preprocessing_fn=custom_preprocessing  # Optional
    )
    
    print("\n🚀 Starting distributed training...")
    print("Make sure your Colab notebook is running and showing the tunnel URL!")
    
    # Start training
    try:
        loader.start_loading(
            gpu_host=COLAB_GPU_HOST,
            gpu_port=COLAB_GPU_PORT,
            epochs=5  # Adjust number of epochs
        )
        
        print("\n✅ Training completed successfully!")
        print("Check your Colab notebook for training results.")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n💡 Common troubleshooting:")
        print("   - Make sure Colab notebook is running")
        print("   - Check that ngrok tunnel is active")
        print("   - Verify COLAB_GPU_HOST and COLAB_GPU_PORT are correct")
        print("   - Ensure both scripts use the same port (default: 29500)")
        print("   - Check your internet connection")
        
        # Additional debugging info
        print(f"\n🔍 Connection details:")
        print(f"   Host: {COLAB_GPU_HOST}")
        print(f"   Port: {COLAB_GPU_PORT}")
        print(f"   Dataset size: {len(dataset):,} samples")


if __name__ == "__main__":
    main()