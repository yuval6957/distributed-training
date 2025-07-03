# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Distributed Training Framework** for machine learning that enables CPU/GPU split architectures. The system separates data processing (CPU-intensive) and model training (GPU-intensive) across different servers using network communication.

## Architecture

### Core Components
- **CPU Master** (`distributed_training/backends/cpu_master.py`): Handles data loading and distribution
- **GPU Worker** (`distributed_training/backends/gpu_worker.py`): Performs model training
- **User Wrappers**: `gpu_trainer.py` and `cpu_loader.py` provide simplified interfaces
- **Colab Integration**: `colab_gpu_trainer.py` provides Google Colab support

### Key Design Patterns
- **Network-based communication** between CPU and GPU servers
- **Adaptive flow control** to prevent data loss during transmission
- **Queue-based batch processing** with smart memory management
- **Fault-tolerant networking** with retry logic and connection recovery

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the System
```bash
# On GPU server (example: 192.168.1.100)
python examples/train_gpu.py

# On CPU server (example: 192.168.1.200)
python examples/train_cpu.py

# Google Colab example
python examples/colab_gpu_example.py  # In Colab
python examples/colab_cpu_local.py    # On local machine
```

### Testing
```bash
# Test network connection
python distributed_training/tests/test_connection.py

# Run simple test
python distributed_training/tests/simple_test.py cpu  # On CPU server
python distributed_training/tests/simple_test.py gpu  # On GPU server

# Run all tests
python -m pytest distributed_training/tests/
```

## Network Configuration

The system uses configurable IP addresses for communication:
- **GPU Server**: Default examples use `192.168.1.100` (port 29500)
- **CPU Server**: Default examples use `192.168.1.200`
- **Port**: 29500 (configurable)

These addresses should be adjusted based on your network setup.

## Key Dependencies

- **PyTorch**: Deep learning framework
- **Ray**: Distributed computing (core networking)
- **NumPy/Pandas**: Data manipulation
- **tqdm**: Progress tracking
- **psutil**: System monitoring
- **pyngrok**: Google Colab tunnel support

## File Structure Context

- `distributed_training/`: Main package with user-facing wrappers
- `distributed_training/backends/`: Core implementation with networking logic
- `examples/`: Demonstration scripts for both CPU and GPU sides
- `concept/`: Original prototype implementations
- `requirements-worker.txt`: Minimal dependencies for worker-only deployments
- `GOOGLE_COLAB_SETUP.md`: Detailed Google Colab setup instructions

## Google Colab Support

This framework includes special support for Google Colab:
- `ColabGPUTrainer`: Automatic ngrok tunnel creation
- `colab_gpu_example.py`: Complete Colab example
- `colab_cpu_local.py`: Local machine connection script

## Important Notes

- This is a generic distributed training framework suitable for various ML projects
- Network addresses in examples are generic and should be adapted to specific environments
- The framework is designed to be environment-agnostic and publicly usable
- Focus on the distributed training capabilities rather than specific use cases