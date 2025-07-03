# examples/validate_gpu.py - Run this on your GPU server for validation

import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from distributed_training import GPUTrainer
from distributed_training.config import get_config

# Your model development - focus on this!
class YourModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=512, output_size=10):
        super(YourModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 4, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Custom loss function for validation
def create_criterion():
    return nn.CrossEntropyLoss(label_smoothing=0.1)

def load_trained_model(model, checkpoint_path=None):
    """Load a trained model from checkpoint"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📂 Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully")
    else:
        print("⚠️  No checkpoint provided, using randomly initialized model")
        print("   For real validation, train a model first or provide checkpoint_path")
    return model

if __name__ == "__main__":
    print("🔍 Starting GPU Validation Server")
    
    # 1. Create your model
    model = YourModel(
        input_size=100,
        hidden_size=512, 
        output_size=10
    )
    
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 2. Load trained weights (optional)
    # Uncomment the next line and provide path to your trained model
    # model = load_trained_model(model, "/path/to/your/checkpoint.pth")
    
    # 3. Setup trainer for validation
    config = get_config()
    network_config = config.get_network_config()
    trainer = GPUTrainer(port=network_config['port'], device=config.get('gpu.device', 'auto'))
    
    # 4. Configure model for validation (no optimizer needed)
    trainer.setup_model(
        model=model,
        optimizer=None,  # No optimizer needed for validation
        criterion=create_criterion()
    )
    
    # 5. Start validation service
    print(f"🔍 Starting validation service...")
    print(f"📡 Listening on port: {network_config['port']}")
    print(f"🎯 Waiting for CPU server to connect from: {network_config['cpu_host']}")
    print("💡 You can change these settings in config.yaml or .env file")
    print("   Run validate_cpu.py on your CPU server to start validation!")
    
    trainer.start_service(mode='validation')