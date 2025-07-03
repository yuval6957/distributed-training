# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Distributed Training Framework** for machine learning that enables CPU/GPU split architectures. The system separates data processing (CPU-intensive) and model operations (GPU-intensive) across different servers using network communication.

**Complete ML Lifecycle Support**: The framework now supports training, validation, and inference with memory-efficient configurations for large datasets including images.

## Architecture

### Core Components
- **CPU Master** (`distributed_training/backends/cpu_master.py`): Handles data loading, validation, and inference coordination
- **GPU Worker** (`distributed_training/backends/gpu_worker.py`): Performs model training, validation, and inference
- **User Wrappers**: `gpu_trainer.py` and `cpu_loader.py` provide simplified interfaces for all operations
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

#### Training
```bash
# On GPU server (example: 192.168.1.100)
python examples/train_gpu.py

# On CPU server (example: 192.168.1.200)
python examples/train_cpu.py

# Google Colab example
python examples/colab_gpu_example.py  # In Colab
python examples/colab_cpu_local.py    # On local machine
```

#### Validation
```bash
# On GPU server
python examples/validate_gpu.py

# On CPU server
python examples/validate_cpu.py
```

#### Inference
```bash
# On GPU server
python examples/inference_gpu.py

# On CPU server (memory-efficient)
python examples/inference_cpu.py

# Image processing example
python examples/image_inference_example.py
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

- This is a comprehensive distributed ML framework supporting the complete lifecycle: training → validation → inference
- Memory-efficient inference makes it suitable for large datasets including images and high-dimensional data
- Network addresses in examples are generic and should be adapted to specific environments
- The framework is designed to be environment-agnostic and publicly usable
- Focus on the distributed capabilities for training, validation, and inference rather than specific use cases
- For inference with large datasets (images, etc.), use memory-efficient options to prevent network/memory issues