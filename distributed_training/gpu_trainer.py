# distributed_training/gpu_trainer.py
import torch
import torch.nn as nn
from .backends.gpu_worker import GPUTrainingWorker

class GPUTrainer:
    """
    Simple wrapper for GPU-side distributed training.
    Hides all the networking complexity - just focus on your model!
    """
    
    def __init__(self, port=29500, device='auto'):
        """
        Initialize GPU trainer
        
        Args:
            port: Port to listen on for CPU server connections
            device: 'auto', 'cuda', or 'cpu'
        """
        self.port = port
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 GPU Trainer initialized on device: {self.device}")
        
        self.worker = None
        self.model = None
        self.optimizer = None
        self.criterion = None
    
    def setup_model(self, model, optimizer=None, criterion=None, learning_rate=0.001):
        """
        Setup your model, optimizer, and loss function
        
        Args:
            model: Your PyTorch model
            optimizer: Optional optimizer (default: AdamW)
            criterion: Optional loss function (default: CrossEntropyLoss)
            learning_rate: Learning rate for default optimizer
        """
        self.model = model.to(self.device)
        
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
        print(f"✅ Model setup complete:")
        print(f"   - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   - Trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"   - Optimizer: {type(self.optimizer).__name__}")
        print(f"   - Loss: {type(self.criterion).__name__}")
    
    def start_training(self, cpu_host, model_config=None, wait_for_cpu=True):
        """
        Start distributed training - waits for CPU server to connect
        
        Args:
            cpu_host: IP address of the CPU server
            model_config: Optional model configuration dict
            wait_for_cpu: Whether to wait for CPU server connection
        """
        if self.model is None:
            raise ValueError("Must call setup_model() first!")
        
        # Prepare model config from the actual model
        if model_config is None:
            # Try to infer from model
            model_config = self._infer_model_config()
        
        print(f"🌐 Starting GPU worker, waiting for CPU server at {cpu_host}")
        print(f"📡 Listening on port {self.port}")
        
        # Create and configure the worker with our model
        self.worker = CustomGPUWorker(
            host='0.0.0.0',
            port=self.port,
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device
        )
        
        try:
            self.worker.start()
        except KeyboardInterrupt:
            print("🛑 Training interrupted by user")
        finally:
            if self.worker:
                self.worker.stop()
            print("✅ GPU Trainer shut down")
    
    def _infer_model_config(self):
        """Try to infer model configuration"""
        if self.model is None or self.optimizer is None:
            return {
                'input_size': 100,
                'output_size': 10,
                'learning_rate': 0.001
            }
        
        try:
            # Get input size from first layer
            first_layer = next(self.model.modules())
            if hasattr(first_layer, 'in_features'):
                input_size = first_layer.in_features
            else:
                input_size = 100  # Default
            
            # Get output size from last layer
            *_, last_layer = self.model.modules()
            if hasattr(last_layer, 'out_features'):
                output_size = last_layer.out_features
            else:
                output_size = 10  # Default
            
            return {
                'input_size': input_size,
                'output_size': output_size,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
        except:
            return {
                'input_size': 100,
                'output_size': 10,
                'learning_rate': 0.001
            }


class CustomGPUWorker(GPUTrainingWorker):
    """
    Custom GPU worker that uses your provided model instead of creating its own
    """
    
    def __init__(self, host, port, model, optimizer, criterion, device):
        self.provided_model = model
        self.provided_optimizer = optimizer  
        self.provided_criterion = criterion
        self.provided_device = device
        
        # Initialize parent with the provided components
        super().__init__(host, port, model, optimizer, criterion, device)
    
    # The rest of the methods are inherited from GPUTrainingWorker
    # No need to override _process_batch since the parent already uses
    # the model, optimizer, and criterion passed to __init__