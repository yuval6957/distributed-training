# Configuration System Migration

This document describes the new configuration system added to the Distributed Training Framework.

## ✨ What's New

### Before (Hardcoded)
```python
# Old way - hardcoded IPs in each file
trainer = GPUTrainer(port=29500)
trainer.start_training(cpu_host='192.168.1.200')

loader = CPULoader(num_workers=4)
loader.start_loading(gpu_host='192.168.1.100', epochs=10)
```

### After (Configurable)
```python
# New way - configuration loaded automatically
from distributed_training.config import get_config

config = get_config()
network_config = config.get_network_config()

trainer = GPUTrainer(port=network_config['port'])
trainer.start_training(cpu_host=network_config['cpu_host'])

loader = CPULoader(num_workers=config.get('training.num_workers'))
loader.start_loading(gpu_host=network_config['gpu_host'], epochs=config.get('training.epochs'))
```

## 🆕 New Files Added

### Configuration Files
1. **`config.yaml`** - Main structured configuration file
2. **`.env.example`** - Environment variable template
3. **`distributed_training/config.py`** - Configuration management system
4. **`CONFIGURATION.md`** - Complete configuration documentation

### Updated Files
- All example scripts (`examples/*.py`) now use configuration
- All test scripts (`distributed_training/tests/*.py`) now use configuration
- Updated `README.md` with configuration examples
- Updated `__init__.py` to expose config module
- Updated `init_repo.py` with configuration instructions

## 🔧 Configuration Options

### Multiple Configuration Methods
1. **YAML file** (`config.yaml`) - Structured, human-readable
2. **Environment file** (`.env`) - Simple key-value pairs
3. **Environment variables** - Direct system environment
4. **Default values** - Built-in fallbacks

### Priority Order (highest to lowest)
1. Environment variables (e.g., `export GPU_HOST=192.168.1.100`)
2. `.env` file values
3. `config.yaml` file values
4. Default values

## 📋 Configuration Categories

### Network Settings
- `network.gpu_host` - IP address of GPU server
- `network.cpu_host` - IP address of CPU server
- `network.port` - Communication port
- `network.timeout` - Connection timeout

### Training Settings
- `training.batch_size` - Default batch size
- `training.num_workers` - Number of CPU workers
- `training.epochs` - Number of training epochs
- `training.learning_rate` - Learning rate

### GPU Settings
- `gpu.device` - Device selection (auto/cuda/cpu)
- `gpu.mixed_precision` - Mixed precision training

### Google Colab Settings
- `colab.ngrok_auth_token` - ngrok authentication token
- `colab.tunnel_region` - ngrok tunnel region

## 🚀 Quick Setup

### Method 1: Copy Templates
```bash
cp config.yaml.example config.yaml
cp .env.example .env
# Edit files with your settings
```

### Method 2: Environment Variables
```bash
export GPU_HOST=192.168.1.100
export CPU_HOST=192.168.1.200
export TRAINING_PORT=29500
```

### Method 3: Direct Config Usage
```python
from distributed_training.config import get_config, reload_config

# Load configuration
config = get_config()

# Get specific values
gpu_host = config.get('network.gpu_host')
batch_size = config.get('training.batch_size', 64)  # with default

# Get entire sections
network_config = config.get_network_config()
training_config = config.get_training_config()

# Convenience functions
from distributed_training.config import get_gpu_host, get_cpu_host, get_port
gpu_host = get_gpu_host()
cpu_host = get_cpu_host()
port = get_port()
```

## 🔄 Migration Guide

### For Existing Users

1. **Install new dependency**:
   ```bash
   pip install PyYAML>=6.0
   ```

2. **Copy configuration templates**:
   ```bash
   cp config.yaml.example config.yaml
   cp .env.example .env
   ```

3. **Update your custom scripts** (optional):
   ```python
   # Replace hardcoded values with config
   from distributed_training.config import get_config
   config = get_config()
   
   # Old: trainer = GPUTrainer(port=29500)
   # New: 
   trainer = GPUTrainer(port=config.get('network.port'))
   ```

4. **Set your network configuration**:
   Edit `config.yaml` or set environment variables with your actual IP addresses.

### Backward Compatibility

All existing code continues to work without changes. The configuration system is additive and doesn't break existing functionality.

## 🎯 Benefits

1. **No more hardcoded IPs** - Easy to change network settings
2. **Environment-specific configs** - Different settings for dev/prod
3. **Version control friendly** - Keep secrets out of code
4. **Flexible deployment** - Use environment variables in containers
5. **Better documentation** - Clear configuration options
6. **Type conversion** - Automatic string to int/bool/float conversion

## 🔍 Troubleshooting

### Configuration Not Loading
```python
from distributed_training.config import get_config
config = get_config()
print("Current configuration:")
print(config.to_dict())
```

### Check File Locations
- `config.yaml` should be in working directory
- `.env` should be in working directory
- Check file permissions and YAML syntax

### Debug Environment Variables
```python
import os
print(f"GPU_HOST: {os.getenv('GPU_HOST')}")
print(f"CPU_HOST: {os.getenv('CPU_HOST')}")
```

## 🎉 Result

The distributed training framework is now much more flexible and deployment-friendly:

- ✅ No hardcoded IP addresses
- ✅ Multiple configuration methods
- ✅ Environment-specific settings
- ✅ Better documentation
- ✅ Backward compatible
- ✅ Production ready