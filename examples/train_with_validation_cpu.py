# examples/train_with_validation_cpu.py - Training with integrated validation

import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from distributed_training import CPULoader
from distributed_training.config import get_config

# Enhanced dataset with better patterns for demonstrating validation
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, size=50000, input_dim=100, num_classes=10, noise_level=0.1):
        print(f"🗃️  Creating training dataset: {size:,} samples")
        
        # Create structured data with clear patterns
        self.X = np.random.randn(size, input_dim).astype(np.float32)
        self.y = np.random.randint(0, num_classes, size).astype(np.int64)
        
        # Add strong patterns that the model can learn
        for class_idx in range(num_classes):
            mask = self.y == class_idx
            # Each class has distinctive features in specific dimensions
            feature_start = class_idx * (input_dim // num_classes)
            feature_end = feature_start + (input_dim // num_classes)
            
            # Add strong signal for this class
            self.X[mask, feature_start:feature_end] += np.random.normal(3.0, noise_level, 
                                                                       (mask.sum(), feature_end - feature_start))
        
        print(f"✅ Training dataset created with clear patterns for {num_classes} classes")
        print(f"📊 Memory usage: ~{(self.X.nbytes + self.y.nbytes) / 1e6:.1f} MB")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def custom_preprocessing(batch_data, batch_targets):
    """
    Custom preprocessing with light augmentation for training
    """
    # Normalize data
    batch_data = (batch_data - batch_data.mean(dim=0)) / (batch_data.std(dim=0) + 1e-8)
    
    # Add slight noise for regularization (only during training)
    if np.random.random() > 0.3:  # 70% chance
        noise = torch.randn_like(batch_data) * 0.05
        batch_data = batch_data + noise
    
    return batch_data, batch_targets

if __name__ == "__main__":
    print("🚀 Training with Integrated Validation Example")
    print("="*60)
    
    # 1. Create training dataset
    dataset = TrainingDataset(
        size=10000,    # Smaller for demo
        input_dim=100,
        num_classes=10,
        noise_level=0.1
    )
    
    # 2. Setup CPU loader with validation split
    config = get_config()
    loader = CPULoader(num_workers=config.get('training.num_workers', 4))
    
    # 3. Configure dataset with 20% validation split
    loader.setup_dataset(
        dataset=dataset,
        batch_size=config.get('training.batch_size', 64),
        shuffle=True,
        preprocessing_fn=custom_preprocessing,
        val_split=0.2  # 20% for validation - this is the key!
    )
    
    # 4. Configure early stopping
    early_stopping = {
        'monitor': 'val_loss',     # Monitor validation loss
        'patience': 5,             # Stop if no improvement for 5 epochs
        'min_delta': 0.001         # Minimum change to qualify as improvement
    }
    
    # Get network configuration
    network_config = config.get_network_config()
    
    print(f"\n🌐 Connecting to GPU server: {network_config['gpu_host']}:{network_config['port']}")
    print("💡 You can change these settings in config.yaml or .env file")
    print("\n🎯 Training Features:")
    print("   - 📊 80% training / 20% validation split")
    print("   - 📈 Validation after every epoch")
    print("   - ⏹️  Early stopping to prevent overfitting")
    print("   - 📉 Training/validation loss tracking")
    print("   - 🎯 Validation accuracy monitoring")
    
    try:
        # Start training with integrated validation
        results = loader.start_loading(
            gpu_host=network_config['gpu_host'],
            epochs=config.get('training.epochs', 20),  # More epochs to see early stopping
            gpu_port=network_config['port'],
            early_stopping=early_stopping
        )
        
        # Print final results
        if results:
            print("\n" + "="*60)
            print("📊 FINAL TRAINING RESULTS")
            print("="*60)
            
            epochs_completed = len(results['epochs'])
            print(f"🎯 Epochs completed: {epochs_completed}")
            
            if results['val_loss']:
                print(f"📉 Training curve:")
                for i, epoch in enumerate(results['epochs']):
                    train_loss = results['train_loss'][i]
                    val_loss = results['val_loss'][i] 
                    val_acc = results['val_accuracy'][i]
                    print(f"   Epoch {epoch:2d}: Train={train_loss:.4f}, Val={val_loss:.4f}, Acc={val_acc:.4f}")
                
                best_epoch = np.argmin(results['val_loss']) + 1
                best_val_loss = min(results['val_loss'])
                best_val_acc = max(results['val_accuracy'])
                
                print(f"\n🏆 Best Results:")
                print(f"   - Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
                print(f"   - Best validation accuracy: {best_val_acc:.4f}")
                
                # Check if early stopping was triggered
                total_requested = config.get('training.epochs', 20)
                if epochs_completed < total_requested:
                    print(f"⏹️  Early stopping saved {total_requested - epochs_completed} epochs!")
            
            # Save results (optional)
            # import json
            # with open('training_results.json', 'w') as f:
            #     json.dump(results, f, indent=2)
            # print("💾 Results saved to training_results.json")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Make sure train_gpu.py is running on the GPU server")
        print("💡 Check network connectivity and firewall settings")
    
    print("\n" + "="*60)
    print("💡 KEY BENEFITS OF INTEGRATED VALIDATION:")
    print("="*60)
    print("1. 📊 Automatic train/validation split")
    print("2. 📈 Per-epoch validation metrics")
    print("3. ⏹️  Early stopping prevents overfitting")
    print("4. 📉 Training curves for analysis")
    print("5. 🎯 No separate validation step needed")
    print("6. 🚀 Optimal training time selection")