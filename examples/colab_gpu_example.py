# examples/colab_gpu_example.py
"""
Google Colab GPU Training Example

This script runs on Google Colab with GPU enabled.
It creates an ngrok tunnel to receive data from your local CPU.

Usage in Colab:
1. Upload this file to Colab
2. Run: !pip install -r requirements-colab.txt
3. Add ngrok token to Colab secrets (key: 'ngrok')
4. Run this script
5. Copy the TCP tunnel info and use it in your local CPU script
"""

import torch
import torch.nn as nn
import sys
import os

# Add the distributed_training package to path
# Adjust this path based on where you place the package in Colab
sys.path.append('/content/distributed-training')  # If you clone the repo to /content/distributed-training

from distributed_training.colab_gpu_trainer import ColabGPUTrainer, quick_colab_setup

class ColabDemoModel(nn.Module):
    """
    Demo model for Colab training
    Matches the expected input/output dimensions
    """
    def __init__(self, input_size=100, hidden_size=512, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 4, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


def main():
    """Main training function for Colab"""
    
    print("🚀 Starting Colab GPU Training Setup")
    print("="*50)
    
    # Setup ngrok authentication from Colab secrets
    try:
        from google.colab import userdata
        import pyngrok
        from pyngrok import ngrok
        
        ngrok_token = userdata.get('ngrok')
        ngrok.set_auth_token(ngrok_token)
        print("✅ ngrok authentication configured")
    except Exception as e:
        print(f"⚠️  ngrok setup warning: {e}")
        print("💡 Add your ngrok token to Colab secrets with key 'ngrok'")
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  WARNING: GPU not detected. Make sure GPU is enabled in Colab:")
        print("   Runtime -> Change runtime type -> Hardware accelerator -> GPU")
    
    # Option 1: Quick setup (recommended for beginners)
    print("\n🎯 Using quick setup...")
    trainer = quick_colab_setup(
        model_class=ColabDemoModel,
        model_kwargs={'input_size': 100, 'hidden_size': 512, 'output_size': 10},
        learning_rate=0.001
    )
    
    # Option 2: Manual setup (for more control)
    # trainer = ColabGPUTrainer(port=29500)
    # model = ColabDemoModel(input_size=100, hidden_size=512, output_size=10)
    # trainer.setup_model(model, learning_rate=0.001)
    
    print("\n🌐 Starting training with ngrok tunnel...")
    print("Keep this notebook running and follow the instructions!")
    
    # Start training with tunnel
    try:
        host, port = trainer.start_training_with_tunnel(show_local_instructions=True)
        print(f"✅ Training completed! Tunnel was: {host}:{port}")
    except KeyboardInterrupt:
        print("🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Common issues:")
        print("   - Make sure requirements are installed: !pip install -r requirements-colab.txt")
        print("   - Add ngrok token to Colab secrets (key: 'ngrok')")
        print("   - Check if GPU is enabled in Colab")
        print("   - Verify your local CPU script is running")


if __name__ == "__main__":
    main()