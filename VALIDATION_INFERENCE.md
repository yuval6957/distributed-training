# Validation and Inference Guide

This guide covers the validation and inference capabilities of the Distributed Training Framework, including memory-efficient configurations for large datasets.

## 📋 Overview

The framework supports the complete ML lifecycle:
- **Training**: Distributed model training across CPU/GPU servers
- **Integrated Training+Validation**: Standard ML workflow with automatic train/validation split and early stopping
- **Standalone Validation**: Model evaluation with detailed metrics
- **Inference**: Batch prediction with memory optimization

## 🔄 Integrated Training with Validation

The framework supports the standard ML workflow where training and validation happen together, preventing overfitting through early stopping.

### Setup with Train/Validation Split

```python
# train_with_validation_cpu.py
from distributed_training import CPULoader

loader = CPULoader()

# Automatic train/validation split
loader.setup_dataset(
    dataset=your_dataset,
    batch_size=64,
    val_split=0.2,  # 80% training, 20% validation
    shuffle=True
)

# Configure early stopping
early_stopping = {
    'monitor': 'val_loss',      # Monitor validation loss
    'patience': 5,              # Stop after 5 epochs without improvement  
    'min_delta': 0.001          # Minimum improvement threshold
}

# Start training with integrated validation
results = loader.start_loading(
    gpu_host='192.168.1.100',
    epochs=50,  # Will stop early if needed
    early_stopping=early_stopping
)
```

### Training Process

For each epoch, the framework automatically:

1. **🔥 Training Phase**: Train on 80% of data
2. **📊 Validation Phase**: Evaluate on 20% of data
3. **📈 Metrics Tracking**: Record train/validation metrics
4. **⏹️ Early Stopping Check**: Stop if validation plateaus
5. **🔄 Continue or Stop**: Proceed to next epoch or terminate

### Early Stopping Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `monitor` | `'val_loss'` | Metric to monitor (`'val_loss'` or `'val_accuracy'`) |
| `patience` | `5` | Epochs to wait before stopping |
| `min_delta` | `0.001` | Minimum change to qualify as improvement |

### Training Results

```python
{
    'train_loss': [0.8, 0.6, 0.4, 0.35, 0.34],      # Training loss per epoch
    'val_loss': [0.7, 0.5, 0.45, 0.46, 0.47],       # Validation loss per epoch  
    'val_accuracy': [0.6, 0.75, 0.78, 0.77, 0.76],  # Validation accuracy per epoch
    'epochs': [1, 2, 3, 4, 5]                        # Completed epochs
}

# Early stopping triggered after epoch 5 due to increasing validation loss
```

### Benefits

- **🚫 Prevents Overfitting**: Stops training when validation performance degrades
- **⏰ Saves Time**: No need to run full epochs if model has converged
- **📈 Rich Metrics**: Complete training curves for analysis
- **🎯 Optimal Model**: Automatically finds best training point
- **🔧 Standard Practice**: Follows conventional ML workflow

## 🔍 Standalone Validation

For validating pre-trained models, you can run standalone validation.

### GPU Server Setup

```python
# validate_gpu.py
from distributed_training import GPUTrainer
import torch.nn as nn

# Load your trained model
model = YourModel()
# Optionally load weights: model.load_state_dict(torch.load('checkpoint.pth'))

trainer = GPUTrainer()
trainer.setup_model(
    model=model,
    optimizer=None,  # No optimizer needed for validation
    criterion=nn.CrossEntropyLoss()
)

# Start validation service
trainer.start_service(mode='validation')
```

### CPU Server Validation

```python
# validate_cpu.py
from distributed_training import CPULoader

loader = CPULoader()
loader.setup_dataset(
    dataset=validation_dataset,
    batch_size=128,  # Larger batches for validation
    shuffle=False    # Don't shuffle validation data
)

# Run validation
results = loader.run_validation(
    gpu_host='192.168.1.100',
    val_dataset=validation_dataset,
    batch_size=128
)

# Access results
print(f"Validation Accuracy: {results['accuracy']:.4f}")
print(f"Validation Loss: {results['average_loss']:.4f}")
print(f"Total Samples: {results['total_samples']:,}")
```

### Validation Results

The validation returns a comprehensive results dictionary:

```python
{
    'total_samples': 10000,
    'total_batches': 78,
    'average_loss': 0.2543,
    'accuracy': 0.9234,
    'detailed_results': [...]  # Per-batch results
}
```

## 🔮 Inference

Inference runs predictions on new data with configurable memory efficiency.

### GPU Server Setup

```python
# inference_gpu.py
from distributed_training import GPUTrainer

# Load your trained model
model = YourModel()
model.load_state_dict(torch.load('trained_model.pth'))

trainer = GPUTrainer()
trainer.setup_model(
    model=model,
    optimizer=None,  # No optimizer needed
    criterion=None   # No criterion needed
)

# Start inference service
trainer.start_service(mode='inference')
```

### CPU Server Inference

```python
# inference_cpu.py
from distributed_training import CPULoader

loader = CPULoader()
loader.setup_dataset(
    dataset=inference_dataset,
    batch_size=256,  # Larger batches for inference
    shuffle=False
)

# Memory-efficient inference (recommended)
results = loader.run_inference(
    gpu_host='192.168.1.100',
    inference_dataset=inference_dataset,
    top_k=3,  # Get top-3 predictions
    return_full_probabilities=False,  # Save memory!
    return_raw_outputs=False,         # Save memory!
    confidence_threshold=0.8          # Filter low-confidence
)

# Access predictions
predictions = results['predictions']      # List of predicted classes
confidences = results['confidences']      # Confidence scores
top_k_preds = results.get('top_k_predictions')  # Top-k if requested
```

## 🎛️ Memory Configuration Options

### Configuration Parameters

| Parameter | Default | Description | Memory Impact |
|-----------|---------|-------------|---------------|
| `top_k` | 1 | Return top-k predictions | Medium |
| `return_full_probabilities` | False | Return complete probability matrix | **HIGH** |
| `return_raw_outputs` | False | Return raw model outputs | **HIGH** |
| `confidence_threshold` | 0.0 | Filter by confidence | Low |

### Memory Usage Examples

For **100,000 samples** with **1,000 classes**:

```python
# Memory-efficient (recommended)
results = loader.run_inference(
    gpu_host='192.168.1.100',
    top_k=1,
    return_full_probabilities=False,
    return_raw_outputs=False
)
# Memory usage: ~0.8 MB

# Top-5 predictions
results = loader.run_inference(
    gpu_host='192.168.1.100',
    top_k=5,
    return_full_probabilities=False
)
# Memory usage: ~4 MB

# Full probabilities (research/debugging)
results = loader.run_inference(
    gpu_host='192.168.1.100',
    return_full_probabilities=True  # ⚠️ MEMORY INTENSIVE
)
# Memory usage: ~400 MB (100x larger!)
```

## 🖼️ Image Processing Example

For large image datasets, use memory-efficient configurations:

```python
# Image inference example
from distributed_training import CPULoader
import torch

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image = load_image(self.image_paths[idx])
        return preprocess(image)

# Setup
dataset = ImageDataset(image_paths)
loader = CPULoader()
loader.setup_dataset(dataset, batch_size=32)

# Memory-efficient image inference
results = loader.run_inference(
    gpu_host='192.168.1.100',
    inference_dataset=dataset,
    top_k=5,  # Top-5 for classification
    return_full_probabilities=False,  # Critical for memory efficiency
    confidence_threshold=0.9  # High confidence only
)

# Process results
for i, (pred, conf) in enumerate(zip(results['predictions'], results['confidences'])):
    if conf > 0.9:  # High confidence predictions
        print(f"Image {i}: Class {pred} (confidence: {conf:.3f})")
```

## 📊 Inference Results Structure

```python
{
    'total_samples': 100000,
    'total_batches': 3125,
    'predictions': [2, 5, 1, ...],           # Predicted classes
    'confidences': [0.95, 0.87, 0.92, ...], # Confidence scores
    'estimated_memory_mb': 4.2,             # Memory usage estimate
    'config': {                              # Configuration used
        'top_k': 3,
        'return_full_probabilities': False,
        'confidence_threshold': 0.8
    },
    # Optional fields (if requested):
    'probabilities': [[0.1, 0.9, ...], ...], # Full probability matrix
    'top_k_predictions': {                    # Top-k results
        'indices': [[2, 5, 1], ...],
        'probabilities': [[0.95, 0.87, 0.82], ...]
    }
}
```

## ⚡ Performance Tips

### For Large Datasets (Images, etc.)

1. **Use Memory-Efficient Settings**:
   ```python
   return_full_probabilities=False  # Critical!
   return_raw_outputs=False        # Critical!
   ```

2. **Optimize Batch Size**:
   ```python
   batch_size=256  # Larger batches = fewer network calls
   ```

3. **Filter by Confidence**:
   ```python
   confidence_threshold=0.8  # Only confident predictions
   ```

4. **Use Top-K Instead of Full Probabilities**:
   ```python
   top_k=5  # Much more memory efficient than full probabilities
   ```

### Network Optimization

```python
# Use larger batches to reduce network overhead
loader.setup_dataset(dataset, batch_size=512)

# Process in chunks for very large datasets
def process_large_dataset(dataset, chunk_size=10000):
    for i in range(0, len(dataset), chunk_size):
        chunk = torch.utils.data.Subset(dataset, range(i, min(i+chunk_size, len(dataset))))
        results = loader.run_inference(gpu_host='192.168.1.100', inference_dataset=chunk)
        yield results
```

## 🚨 Memory Warnings

The framework provides automatic warnings for memory-intensive operations:

```python
# This will show a warning
results = loader.run_inference(
    gpu_host='192.168.1.100',
    return_full_probabilities=True  # ⚠️ WARNING: Memory intensive!
)
```

Output:
```
⚠️  WARNING: Requesting full probabilities/raw outputs may use significant memory!
   Consider using top_k predictions instead for large datasets
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**: Use `return_full_probabilities=False`
2. **Slow Inference**: Increase `batch_size`
3. **Network Timeout**: Reduce batch size or check network
4. **Low Accuracy**: Verify model is properly trained and loaded

### Error Messages

```python
# Connection error
ConnectionError: Cannot connect to GPU server: {'status': 'error'}
# Solution: Check GPU server is running and network connectivity

# Memory error (if using full probabilities with large data)
MemoryError: Unable to allocate array
# Solution: Use memory-efficient settings
```

## 📖 Complete Example

See [`examples/image_inference_example.py`](examples/image_inference_example.py) for a comprehensive example demonstrating:

- Memory-efficient configurations
- Performance comparisons
- Real-world image processing
- Error handling
- Results processing

This example shows how to achieve **50x memory savings** for large-scale inference tasks.