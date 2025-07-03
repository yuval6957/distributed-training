# Configuration Guide

The Distributed Training Framework supports flexible configuration through multiple methods. You can easily customize network settings, training parameters, and other options without modifying code.

## 🔧 Configuration Methods

### Priority Order (highest to lowest):
1. **Environment Variables** (highest priority)
2. **`.env` file**
3. **`config.yaml` file**
4. **Default values** (lowest priority)

## 📝 Configuration Files

### 1. YAML Configuration File (`config.yaml`)

Create a `config.yaml` file in your project root:

```yaml
# Network Configuration
network:
  gpu_host: "192.168.1.100"    # IP address of GPU server
  cpu_host: "192.168.1.200"    # IP address of CPU server
  port: 29500                  # Communication port
  timeout: 30                  # Connection timeout in seconds

# Training Configuration
training:
  batch_size: 64               # Default batch size
  num_workers: 4               # Number of CPU worker processes
  epochs: 10                   # Default number of training epochs
  learning_rate: 0.001         # Default learning rate

# GPU Configuration
gpu:
  device: "auto"               # "auto", "cuda", "cpu"
  mixed_precision: false       # Use automatic mixed precision

# Google Colab Configuration
colab:
  ngrok_auth_token: null       # ngrok auth token for stable tunnels
  tunnel_region: "us"          # ngrok tunnel region
```

### 2. Environment File (`.env`)

Create a `.env` file for simple key-value configuration:

```bash
# Network Settings
GPU_HOST=192.168.1.100
CPU_HOST=192.168.1.200
TRAINING_PORT=29500

# Training Settings
BATCH_SIZE=64
NUM_WORKERS=4
EPOCHS=10
LEARNING_RATE=0.001

# GPU Settings
GPU_DEVICE=auto
MIXED_PRECISION=false

# Colab Settings
NGROK_AUTH_TOKEN=your_token_here
NGROK_REGION=us
```

### 3. Environment Variables

Set environment variables directly:

```bash
export GPU_HOST=192.168.1.100
export CPU_HOST=192.168.1.200
export TRAINING_PORT=29500
export BATCH_SIZE=128
```

## 🚀 Quick Setup

### Method 1: Copy Templates

```bash
# Copy configuration templates
cp config.yaml.example config.yaml
cp .env.example .env

# Edit the files with your settings
vim config.yaml  # or your preferred editor
vim .env
```

### Method 2: Use Environment Variables

```bash
# Set for current session
export GPU_HOST=your_gpu_ip
export CPU_HOST=your_cpu_ip
export TRAINING_PORT=29500

# Or create a setup script
echo "export GPU_HOST=192.168.1.100" >> setup_env.sh
echo "export CPU_HOST=192.168.1.200" >> setup_env.sh
source setup_env.sh
```

## 📋 Configuration Options

### Network Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `network.gpu_host` | `GPU_HOST` | `192.168.1.100` | IP address of GPU server |
| `network.cpu_host` | `CPU_HOST` | `192.168.1.200` | IP address of CPU server |
| `network.port` | `TRAINING_PORT` | `29500` | Communication port |
| `network.timeout` | - | `30` | Connection timeout (seconds) |

### Training Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `training.batch_size` | `BATCH_SIZE` | `64` | Training batch size |
| `training.num_workers` | `NUM_WORKERS` | `4` | Number of CPU workers |
| `training.epochs` | `EPOCHS` | `10` | Number of training epochs |
| `training.learning_rate` | `LEARNING_RATE` | `0.001` | Learning rate |

### GPU Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `gpu.device` | `GPU_DEVICE` | `auto` | GPU device selection |
| `gpu.mixed_precision` | `MIXED_PRECISION` | `false` | Use mixed precision |

### Google Colab Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `colab.ngrok_auth_token` | `NGROK_AUTH_TOKEN` | `null` | ngrok authentication token |
| `colab.tunnel_region` | `NGROK_REGION` | `us` | ngrok tunnel region |

## 💻 Usage in Code

### Basic Usage

```python
from distributed_training.config import get_config

# Load configuration
config = get_config()

# Get specific values
gpu_host = config.get('network.gpu_host')
batch_size = config.get('training.batch_size')

# Get entire sections
network_config = config.get_network_config()
training_config = config.get_training_config()
```

### Using Convenience Functions

```python
from distributed_training.config import get_gpu_host, get_cpu_host, get_port

# Get network settings directly
gpu_host = get_gpu_host()
cpu_host = get_cpu_host()
port = get_port()
```

### In Your Training Scripts

```python
from distributed_training import GPUTrainer
from distributed_training.config import get_config

# Load configuration
config = get_config()
network_config = config.get_network_config()

# Use configuration
trainer = GPUTrainer(port=network_config['port'])
trainer.start_training(cpu_host=network_config['cpu_host'])
```

## 🔄 Dynamic Configuration

### Reloading Configuration

```python
from distributed_training.config import reload_config

# Reload configuration (e.g., after changing files)
reload_config()
```

### Using Different Config Files

```python
from distributed_training.config import get_config

# Use custom config files
config = get_config(
    config_file='custom_config.yaml',
    env_file='custom.env'
)
```

## 🎯 Common Scenarios

### Scenario 1: Local Development

```yaml
# config.yaml
network:
  gpu_host: "localhost"
  cpu_host: "localhost"
  port: 29500

training:
  batch_size: 32
  num_workers: 2
  epochs: 5
```

### Scenario 2: Multi-Machine Setup

```bash
# .env file
GPU_HOST=10.0.1.100
CPU_HOST=10.0.1.200
TRAINING_PORT=29500
BATCH_SIZE=128
NUM_WORKERS=8
```

### Scenario 3: Google Colab

```yaml
# config.yaml
colab:
  ngrok_auth_token: "your_ngrok_token"
  tunnel_region: "us"

training:
  batch_size: 64
  epochs: 20
```

### Scenario 4: Production Deployment

```bash
# Environment variables (set by deployment system)
export GPU_HOST=${GPU_SERVER_IP}
export CPU_HOST=${CPU_SERVER_IP}
export TRAINING_PORT=29500
export BATCH_SIZE=256
export NUM_WORKERS=16
export MIXED_PRECISION=true
```

## 🐛 Troubleshooting

### Configuration Not Loading

1. **Check file locations**: Ensure `config.yaml` and `.env` are in the working directory
2. **Check file permissions**: Make sure files are readable
3. **Check YAML syntax**: Validate YAML file syntax
4. **Check environment variables**: Use `echo $GPU_HOST` to verify

### Invalid Configuration Values

```python
from distributed_training.config import get_config

config = get_config()
print("Current configuration:")
print(config.to_dict())
```

### Configuration Priority Issues

```python
import os
from distributed_training.config import get_config

# Check what's being loaded
print("Environment variables:")
print(f"GPU_HOST: {os.getenv('GPU_HOST')}")
print(f"CPU_HOST: {os.getenv('CPU_HOST')}")

config = get_config()
print(f"Final config - GPU Host: {config.get('network.gpu_host')}")
```

## 📚 Best Practices

1. **Use `config.yaml` for structured settings**: Complex configurations
2. **Use `.env` for simple overrides**: Development and testing
3. **Use environment variables for deployment**: Production and CI/CD
4. **Keep sensitive data in environment variables**: Tokens and credentials
5. **Document your configuration**: Comment your config files
6. **Version control templates**: Include `.example` files in git

## 🔒 Security Notes

- **Never commit `.env` files with secrets to version control**
- **Use environment variables for sensitive data**
- **Keep ngrok tokens and API keys in environment variables**
- **Add `.env` to your `.gitignore` file**

Example `.gitignore`:
```
.env
config.local.yaml
*.secret
```