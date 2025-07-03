# distributed_training/__init__.py

from .gpu_trainer import GPUTrainer
from .cpu_loader import CPULoader
from .colab_gpu_trainer import ColabGPUTrainer
from . import config

__version__ = "1.0.0"
__all__ = ["GPUTrainer", "CPULoader", "ColabGPUTrainer", "config"]

# Convenience function for quick setup
def create_simple_dataset(size=10000, input_dim=100, num_classes=10):
    """Create a simple synthetic dataset for testing"""
    import numpy as np
    
    X = np.random.randn(size, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, size).astype(np.int64)
    
    return X, y

print("🚀 Distributed Training Framework loaded")
print("   - GPUTrainer: For GPU server")  
print("   - CPULoader: For CPU server")
print("   - ColabGPUTrainer: For Google Colab GPU")
print("   - config: Configuration management")
print("   - Use create_simple_dataset() for testing")
print("💡 Configure IPs in config.yaml or .env file")