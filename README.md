# Distributed Training Framework

A **Distributed Training Framework** for machine learning that enables CPU/GPU split architectures. The system separates data processing (CPU-intensive) and model training (GPU-intensive) across different servers using network communication.

## 🚀 Features

- **CPU/GPU Split Architecture**: Separate data loading and model training across different machines
- **Network-based Communication**: Robust networking with adaptive flow control
- **Google Colab Support**: Train using Colab's free GPU with your local data
- **Automatic Load Balancing**: Smart queue management prevents data loss
- **Beautiful Progress Tracking**: Real-time progress bars and statistics
- **Fault Tolerance**: Connection recovery and error handling

## 🏗️ Architecture

### Core Components
- **CPU Master**: Handles data loading and distribution
- **GPU Worker**: Performs model training
- **User Wrappers**: `GPUTrainer` and `CPULoader` provide simplified interfaces
- **Colab Integration**: `ColabGPUTrainer` for Google Colab support

### Key Design Patterns
- **Network-based communication** between CPU and GPU servers
- **Adaptive flow control** to prevent data loss during transmission
- **Queue-based batch processing** with smart memory management
- **Fault-tolerant networking** with retry logic and connection recovery

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/distributed-training.git
cd distributed-training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

The framework supports flexible configuration through multiple methods:

### Quick Configuration Setup

```bash
# Copy configuration templates
cp config.yaml.example config.yaml
cp .env.example .env

# Edit with your network settings
vim config.yaml  # Set your GPU and CPU server IPs
```

### Configuration Files

1. **`config.yaml`** - Structured configuration (recommended)
2. **`.env`** - Simple key-value pairs
3. **Environment variables** - For deployment

Example `config.yaml`:
```yaml
network:
  gpu_host: "192.168.1.100"    # Your GPU server IP
  cpu_host: "192.168.1.200"    # Your CPU server IP
  port: 29500                  # Communication port

training:
  batch_size: 64
  num_workers: 4
  epochs: 10
  learning_rate: 0.001
```

Example `.env`:
```bash
GPU_HOST=192.168.1.100
CPU_HOST=192.168.1.200
TRAINING_PORT=29500
BATCH_SIZE=64
```

**📖 See [CONFIGURATION.md](CONFIGURATION.md) for complete configuration guide.**

## 🚀 Quick Start

### Basic Usage

**On GPU Server:**
```python
# examples/train_gpu.py
from distributed_training import GPUTrainer
import torch.nn as nn

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Setup trainer with configuration
from distributed_training.config import get_config
config = get_config()
network_config = config.get_network_config()

trainer = GPUTrainer(port=network_config['port'])
model = MyModel()
trainer.setup_model(model, learning_rate=config.get('training.learning_rate'))

# Start training (waits for CPU server)
trainer.start_training(cpu_host=network_config['cpu_host'])
```

**On CPU Server:**
```python
# examples/train_cpu.py
from distributed_training import CPULoader
import torch
import numpy as np

# Define your dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.X = np.random.randn(50000, 100).astype(np.float32)
        self.y = np.random.randint(0, 10, 50000).astype(np.int64)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# Setup loader with configuration
from distributed_training.config import get_config
config = get_config()
network_config = config.get_network_config()

loader = CPULoader(num_workers=config.get('training.num_workers'))
dataset = MyDataset()
loader.setup_dataset(dataset, batch_size=config.get('training.batch_size'))

# Start loading (connects to GPU server)
loader.start_loading(
    gpu_host=network_config['gpu_host'],
    gpu_port=network_config['port'],
    epochs=config.get('training.epochs')
)
```

### Google Colab Support

Train using Google Colab's free GPU while keeping your data local:

**In Google Colab:**
```python
!pip install pyngrok torch torchvision tqdm
!git clone https://github.com/yourusername/distributed-training.git
%cd distributed-training

from distributed_training import ColabGPUTrainer
import torch.nn as nn

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Setup trainer with tunnel
trainer = ColabGPUTrainer()
trainer.setup_model(MyModel())

# Start training with automatic tunnel creation
trainer.start_training_with_tunnel()
```

**On Your Local Machine:**
```python
# Copy the tunnel URL from Colab output
COLAB_GPU_HOST = "abc123.ngrok.io"  # From Colab
COLAB_GPU_PORT = 443

# Run your local dataset
from distributed_training import CPULoader
# ... setup your dataset ...
loader.start_loading(
    gpu_host=COLAB_GPU_HOST,
    gpu_port=COLAB_GPU_PORT,
    epochs=10
)
```

## 📖 Documentation

- **[Google Colab Setup Guide](GOOGLE_COLAB_SETUP.md)**: Detailed instructions for using Colab
- **[API Reference](docs/api.md)**: Complete API documentation
- **[Examples](examples/)**: Complete example scripts

## 🎯 Use Cases

- **Large Dataset Training**: Process huge datasets on CPU while training on GPU
- **Resource Optimization**: Utilize different machines for their strengths
- **Cloud GPU Training**: Use cloud GPUs with local data
- **Distributed ML**: Scale training across multiple machines

## 🔧 Configuration

### Network Configuration
```python
# Default configuration
GPU_HOST = '192.168.1.100'  # Your GPU server IP
CPU_HOST = '192.168.1.200'  # Your CPU server IP
PORT = 29500                # Communication port
```

### Performance Tuning
```python
# CPU Server
loader = CPULoader(num_workers=8)  # Adjust based on CPU cores
loader.setup_dataset(dataset, batch_size=64)  # Adjust batch size

# GPU Server
trainer = GPUTrainer(port=29500, device='cuda')
```

## 🧪 Testing

```bash
# Test network connectivity
python distributed_training/tests/test_connection.py

# Run simple test
python distributed_training/tests/simple_test.py cpu  # On CPU server
python distributed_training/tests/simple_test.py gpu  # On GPU server

# Run all tests
python -m pytest distributed_training/tests/
```

## 📊 Features

### Progress Tracking
- Real-time epoch progress bars
- Data transmission statistics
- GPU processing metrics
- Queue size monitoring

### Fault Tolerance
- Automatic connection recovery
- Graceful error handling
- Flow control to prevent data loss

### Flexibility
- Support for any PyTorch model
- Custom preprocessing functions
- Configurable batch sizes and workers
- Multiple dataset formats (CSV, NumPy, PyTorch)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs and request features on GitHub Issues
- **Documentation**: Check the [docs](docs/) folder for detailed guides
- **Examples**: See [examples](examples/) for complete working examples

## 🚀 Performance Tips

1. **Network**: Use gigabit ethernet for best performance
2. **Batch Size**: Start with 64 and adjust based on GPU memory
3. **Workers**: Use 4-8 CPU workers for optimal throughput
4. **Preprocessing**: Keep preprocessing lightweight to avoid bottlenecks

---

**Happy Distributed Training!** 🎉