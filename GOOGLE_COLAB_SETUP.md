# Google Colab Setup Guide

This guide shows how to use **Google Colab as your GPU server** while keeping your data processing on your local machine. This is perfect for training models with large datasets without uploading them to the cloud.

## 🎯 Overview

- **Google Colab**: Provides free GPU for model training
- **Your Local Computer**: Handles data loading and preprocessing
- **ngrok**: Creates a tunnel for communication between Colab and your local machine

## 🚀 Quick Start

### Step 1: Setup Google Colab (GPU Server)

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Enable GPU**: 
   - Runtime → Change runtime type → Hardware accelerator → **GPU**
   - Runtime → Restart runtime

3. **Upload Your Code**:
   ```python
   # Option A: Clone from GitHub
   !git clone https://github.com/yuval6957/distributed-training.git
   %cd distributed-training
   
   # Option B: Upload files manually
   # Use the file browser on the left to upload your distributed_training folder
   ```

4. **Install Dependencies**:
   ```python
   !pip install pyngrok torch torchvision tqdm
   ```

5. **Run the GPU Script**:
   ```python
   # Copy and run the code from examples/colab_gpu_example.py
   exec(open('examples/colab_gpu_example.py').read())
   ```

6. **Copy the Tunnel URL**: The output will show something like:
   ```
   ====================================================================
   🏠 INSTRUCTIONS FOR YOUR LOCAL COMPUTER:
   ====================================================================
   1. Copy and paste this into your local CPU script:
   
      COLAB_GPU_HOST = 'abc123.ngrok.io'
      COLAB_GPU_PORT = 443
   ```

### Step 2: Setup Your Local Computer (CPU Server)

1. **Update Connection Settings**: Edit `examples/colab_cpu_local.py`:
   ```python
   # Update these lines with values from Colab
   COLAB_GPU_HOST = "abc123.ngrok.io"  # From Colab output
   COLAB_GPU_PORT = 443                # From Colab output
   ```

2. **Setup Your Dataset**: Replace the synthetic data with your actual dataset:
   ```python
   # Option 1: Load from CSV
   dataset = FileDataset('/path/to/your/data.csv')
   
   # Option 2: Load from numpy arrays
   X = np.load('your_features.npy')
   y = np.load('your_labels.npy')
   dataset = (X, y)
   ```

3. **Run the Local Script**:
   ```bash
   python examples/colab_cpu_local.py
   ```

## 📋 Detailed Setup Instructions

### Google Colab Detailed Setup

#### 1. Create a New Colab Notebook

```python
# Cell 1: Setup environment
!pip install pyngrok torch torchvision tqdm

# Cell 2: Clone your repository
!git clone https://github.com/yuval6957/distributed-training.git
%cd distributed-training

# Cell 3: Import and setup
import sys
sys.path.append('/content/distributed-training')

from distributed_training.colab_gpu_trainer import ColabGPUTrainer, quick_colab_setup
import torch
import torch.nn as nn

# Cell 4: Define your model
class MyModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=512, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Cell 5: Start training
trainer = quick_colab_setup(
    model_class=MyModel,
    model_kwargs={'input_size': 100, 'hidden_size': 512, 'output_size': 10},
    learning_rate=0.001
)

# This will create the ngrok tunnel and show connection info
trainer.start_training_with_tunnel()
```

#### 2. ngrok Setup (Optional - for persistent tunnels)

For more stable connections, you can get a free ngrok account:

1. **Sign up**: Go to [ngrok.com](https://ngrok.com) and create a free account
2. **Get your auth token**: Dashboard → Your Authtoken
3. **Use in Colab**:
   ```python
   trainer = ColabGPUTrainer(ngrok_auth_token="your_auth_token_here")
   ```

### Local Computer Detailed Setup

#### 1. Install Dependencies

```bash
# Make sure you have the required packages
pip install torch torchvision tqdm numpy pandas
```

#### 2. Prepare Your Dataset

```python
# examples/colab_cpu_local.py

# For CSV files
class FileDataset(Dataset):
    def __init__(self, csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.X = df.iloc[:, :-1].values.astype(np.float32)
        self.y = df.iloc[:, -1].values.astype(np.int64)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx:idx+1])[0]

# For numpy files
class NumpyDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = torch.from_numpy(np.load(X_path).astype(np.float32))
        self.y = torch.from_numpy(np.load(y_path).astype(np.int64))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

#### 3. Custom Preprocessing

```python
def custom_preprocessing(batch_data, batch_targets):
    # Data augmentation
    if np.random.random() < 0.3:
        noise = torch.randn_like(batch_data) * 0.01
        batch_data = batch_data + noise
    
    # Normalization
    batch_data = (batch_data - batch_data.mean()) / (batch_data.std() + 1e-8)
    
    return batch_data, batch_targets
```

## 🔧 Configuration Options

### GPU Training Options

```python
# Basic setup
trainer = ColabGPUTrainer(
    port=29500,                    # Port for communication
    device='auto',                 # 'auto', 'cuda', or 'cpu'
    ngrok_auth_token=None          # Optional ngrok auth token
)

# Model setup
trainer.setup_model(
    model=your_model,
    optimizer=None,                # None for default AdamW
    criterion=None,                # None for default CrossEntropyLoss
    learning_rate=0.001
)

# Start training
trainer.start_training_with_tunnel(
    show_local_instructions=True,  # Show connection instructions
    tunnel_region='us'             # ngrok region
)
```

### CPU Loading Options

```python
# CPU loader setup
loader = CPULoader(num_workers=4)  # Adjust based on your CPU cores

# Dataset setup
loader.setup_dataset(
    dataset=your_dataset,
    batch_size=64,                 # Adjust based on GPU memory
    shuffle=True,
    preprocessing_fn=custom_preprocessing  # Optional
)

# Start loading
loader.start_loading(
    gpu_host='abc123.ngrok.io',    # From Colab output
    gpu_port=443,                  # From Colab output
    epochs=10                      # Number of training epochs
)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Connection Failed
```
❌ Failed to connect to GPU server
```
**Solutions:**
- Make sure Colab notebook is running
- Check that ngrok tunnel is active in Colab
- Verify `COLAB_GPU_HOST` and `COLAB_GPU_PORT` are correct
- Try refreshing the Colab notebook

#### 2. ngrok Not Found
```
❌ pyngrok not found
```
**Solution:**
```python
!pip install pyngrok
```

#### 3. GPU Not Detected
```
⚠️ WARNING: GPU not detected
```
**Solution:**
- Runtime → Change runtime type → Hardware accelerator → **GPU**
- Runtime → Restart runtime

#### 4. Tunnel Disconnected
```
❌ ngrok tunnel disconnected
```
**Solutions:**
- Restart the Colab cell creating the tunnel
- Get a free ngrok account for more stable tunnels
- Check your internet connection

### Performance Tips

1. **Batch Size**: Start with 64 and adjust based on GPU memory
2. **Workers**: Use 4-8 workers on your local CPU
3. **Preprocessing**: Keep preprocessing light to avoid CPU bottlenecks
4. **Network**: Use a stable internet connection

### Monitoring

```python
# Check tunnel status
info = trainer.get_tunnel_info()
print(f"Tunnel active: {info['active']}")
print(f"URL: {info['url']}")

# Monitor GPU usage in Colab
!nvidia-smi
```

## 📊 Example Training Session

Here's what a successful training session looks like:

### Colab Output:
```
🚀 GPU Trainer initialized on device: cuda
✅ Model setup complete:
   - Parameters: 326,922
   - Trainable: 326,922
   - Optimizer: AdamW
   - Loss: CrossEntropyLoss
🌐 Creating ngrok tunnel for Colab...
✅ ngrok tunnel created: https://abc123.ngrok.io

====================================================================
🏠 INSTRUCTIONS FOR YOUR LOCAL COMPUTER:
====================================================================
1. Copy and paste this into your local CPU script:

   COLAB_GPU_HOST = 'abc123.ngrok.io'
   COLAB_GPU_PORT = 443

2. Run your local CPU script with these values
3. Keep this Colab notebook running!
====================================================================

🚀 Starting GPU worker on cuda
📡 Waiting for local CPU server to connect...
```

### Local Computer Output:
```
🔧 CPU Loader initialized with 4 workers
✅ Dataset setup complete:
   - Samples: 50,000
   - Batch size: 64
   - Batches per epoch: 782
   - Workers: 4
🌐 Starting CPU data master, connecting to GPU at abc123.ngrok.io:443
🔍 Testing connection to GPU server...
✅ Successfully connected to GPU server
🚀 Starting distributed training with your dataset

🎯 Training Epochs: 100%|██████████| 5/5 [02:30<00:00, 30.12s/it]
📤 Data Sent: 100%|██████████| 3910/3910 [02:30<00:00, 26.03it/s]
```

## 🎉 Success!

You've successfully set up distributed training with Google Colab! Your model is training on Colab's GPU while your data stays on your local machine.

## 📝 Next Steps

1. **Customize the model**: Replace the example model with your own architecture
2. **Optimize performance**: Adjust batch sizes and worker counts
3. **Add monitoring**: Integrate with TensorBoard or Weights & Biases
4. **Scale up**: Try with larger datasets and longer training runs

## 🆘 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review the example scripts in `examples/`
3. Make sure all dependencies are installed
4. Verify your internet connection is stable