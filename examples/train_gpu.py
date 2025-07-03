# examples/train_gpu.py - Run this on your GPU server (e.g., 192.168.1.100)

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

# Custom optimizer setup
def create_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

# Custom loss function  
def create_criterion():
    return nn.CrossEntropyLoss(label_smoothing=0.1)

if __name__ == "__main__":
    print("🚀 Starting GPU Training Server with Beautiful Progress Bars")
    
    # 1. Create your model
    model = YourModel(
        input_size=100,
        hidden_size=512, 
        output_size=10
    )
    
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 2. Setup trainer with config
    config = get_config()
    network_config = config.get_network_config()
    trainer = GPUTrainer(port=network_config['port'], device=config.get('gpu.device', 'auto'))
    
    # 3. Configure training
    trainer.setup_model(
        model=model,
        optimizer=create_optimizer(model),
        criterion=create_criterion(),
        learning_rate=config.get('training.learning_rate', 0.001)
    )
    
    # 4. Start distributed training with progress bars
    print(f"🎯 Waiting for CPU server to connect from: {network_config['cpu_host']}")
    print(f"📡 Listening on port: {network_config['port']}")
    print("💡 You can change these settings in config.yaml or .env file")
    print("   You'll see beautiful progress bars once training starts!")
    
    trainer.start_training(
        cpu_host=network_config['cpu_host'],
        wait_for_cpu=True
    )