# distributed_training/cpu_loader.py
import torch
import time  # Add this import
import queue  # Add this import
from torch.utils.data import DataLoader, Dataset
from .backends.cpu_master import CPUDataMaster
import numpy as np
from tqdm import tqdm  # Add this import

class CPULoader:
    """
    Simple wrapper for CPU-side distributed training.
    Hides all the networking complexity - just focus on your data!
    """
    
    def __init__(self, num_workers='auto'):
        """
        Initialize CPU data loader
        
        Args:
            num_workers: Number of worker processes ('auto' for CPU count)
        """
        if num_workers == 'auto':
            import multiprocessing as mp
            self.num_workers = min(mp.cpu_count(), 8)
        else:
            self.num_workers = int(num_workers)
        
        print(f"🔧 CPU Loader initialized with {self.num_workers} workers")
        
        self.master = None
        self.dataset = None
        self.dataloader = None
        self.preprocessor = None
    
    def setup_dataset(self, dataset, batch_size=64, shuffle=True, preprocessing_fn=None):
        """
        Setup your dataset and preprocessing
        
        Args:
            dataset: Your PyTorch dataset or data path
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            preprocessing_fn: Optional preprocessing function
        """
        self.dataset = dataset
        self.preprocessor = preprocessing_fn
        
        # Handle different dataset types
        if isinstance(dataset, str):
            # Assume it's a file path
            self.dataset = self._load_from_path(dataset)
        elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
            # Assume it's (X, y) data
            self.dataset = SimpleDataset(dataset[0], dataset[1])
        elif not isinstance(dataset, Dataset):
            raise ValueError("Dataset must be a PyTorch Dataset, file path, or (X, y) tuple")
        
        # Create DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else 2
        )
        
        print(f"✅ Dataset setup complete:")
        print(f"   - Samples: {len(self.dataset):,}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Batches per epoch: {len(self.dataloader):,}")
        print(f"   - Workers: {self.num_workers}")
        if preprocessing_fn:
            print(f"   - Preprocessing: {preprocessing_fn.__name__}")
    
    def start_loading(self, gpu_host, epochs=10, gpu_port=29500):
        """
        Start distributed data loading - sends data to GPU server
        
        Args:
            gpu_host: IP address of the GPU server
            epochs: Number of training epochs
            gpu_port: Port of the GPU server
        """
        if self.dataset is None:
            raise ValueError("Must call setup_dataset() first!")
        
        print(f"🌐 Starting CPU data master, connecting to GPU at {gpu_host}:{gpu_port}")
        print(f"🔄 Training for {epochs} epochs")
        
        # Create the master with our dataset
        self.master = CustomCPUMaster(
            gpu_host=gpu_host,
            gpu_port=gpu_port,
            num_workers=self.num_workers,
            dataloader=self.dataloader,
            preprocessor=self.preprocessor
        )
        
        try:
            # Test connection first
            print("🔍 Testing connection to GPU server...")
            test_response = self.master.send_message_to_gpu({'command': 'get_stats'})
            
            if test_response.get('status') == 'success':
                print("✅ Successfully connected to GPU server")
                
                # Start training
                self.master.train_epochs(epochs)
                
            else:
                print(f"❌ Failed to connect to GPU server: {test_response}")
                print("Make sure the GPU worker is running and accessible")
                
        except KeyboardInterrupt:
            print("🛑 Training interrupted by user")
        except Exception as e:
            print(f"❌ Training failed: {e}")
        finally:
            if self.master:
                self.master.cleanup()
            print("✅ CPU Loader shut down")
    
    def run_validation(self, gpu_host, val_dataset=None, batch_size=64, gpu_port=29500):
        """
        Run validation on the GPU server
        
        Args:
            gpu_host: IP address of the GPU server
            val_dataset: Validation dataset (if None, uses training dataset)
            batch_size: Batch size for validation
            gpu_port: Port of the GPU server
        """
        if val_dataset is None:
            if self.dataset is None:
                raise ValueError("Must call setup_dataset() first or provide val_dataset!")
            val_dataset = self.dataset
        
        # Create validation dataloader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"🔍 Starting validation on GPU at {gpu_host}:{gpu_port}")
        print(f"📊 Validation samples: {len(val_dataset):,}")
        
        # Create master for validation
        master = CPUDataMaster(gpu_host, gpu_port, self.num_workers)
        
        try:
            # Test connection
            test_response = master.send_message_to_gpu({'command': 'get_stats'})
            if test_response.get('status') != 'success':
                raise ConnectionError(f"Cannot connect to GPU server: {test_response}")
            
            # Run validation
            results = master.validate_dataset(val_dataloader)
            return results
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            raise
        finally:
            print("✅ Validation complete")
    
    def run_inference(self, gpu_host, inference_dataset=None, batch_size=64, top_k=1, gpu_port=29500, 
                     return_details=False, return_full_probabilities=False, return_raw_outputs=False, 
                     confidence_threshold=0.0):
        """
        Run inference on the GPU server with configurable output options
        
        Args:
            gpu_host: IP address of the GPU server
            inference_dataset: Dataset for inference (if None, uses training dataset)
            batch_size: Batch size for inference
            top_k: Return top-k predictions
            gpu_port: Port of the GPU server
            return_details: Whether to return detailed batch results
            return_full_probabilities: Whether to return full probability matrices (memory intensive!)
            return_raw_outputs: Whether to return raw model outputs (memory intensive!)
            confidence_threshold: Filter predictions by confidence threshold
        """
        if inference_dataset is None:
            if self.dataset is None:
                raise ValueError("Must call setup_dataset() first or provide inference_dataset!")
            inference_dataset = self.dataset
        
        # Create inference dataloader
        inference_dataloader = DataLoader(
            inference_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"🔮 Starting inference on GPU at {gpu_host}:{gpu_port}")
        print(f"📊 Inference samples: {len(inference_dataset):,}")
        
        # Show memory warning for large data requests
        if return_full_probabilities or return_raw_outputs:
            print("⚠️  WARNING: Requesting full probabilities/raw outputs may use significant memory!")
            print("   Consider using top_k predictions instead for large datasets")
        
        # Create master for inference
        master = CPUDataMaster(gpu_host, gpu_port, self.num_workers)
        
        try:
            # Test connection
            test_response = master.send_message_to_gpu({'command': 'get_stats'})
            if test_response.get('status') != 'success':
                raise ConnectionError(f"Cannot connect to GPU server: {test_response}")
            
            # Run inference with configurable options
            results = master.run_inference(
                inference_dataloader, 
                top_k=top_k, 
                return_details=return_details,
                return_full_probabilities=return_full_probabilities,
                return_raw_outputs=return_raw_outputs,
                confidence_threshold=confidence_threshold
            )
            return results
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            raise
        finally:
            print("✅ Inference complete")
    
    def _load_from_path(self, data_path):
        """Load dataset from file path"""
        from pathlib import Path
        
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        if path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(data_path)
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(np.int64)
            return SimpleDataset(X, y)
        
        elif path.suffix == '.npy':
            data = np.load(data_path, allow_pickle=True)
            if isinstance(data, dict):
                return SimpleDataset(data['X'], data['y'])
            else:
                # Assume first array is X, second is y
                return SimpleDataset(data[0], data[1])
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


class SimpleDataset(Dataset):
    """Simple dataset wrapper for numpy arrays"""
    
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.array(X, dtype=np.float32))
        self.y = torch.from_numpy(np.array(y, dtype=np.int64))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CustomCPUMaster(CPUDataMaster):
    """
    Custom CPU master that uses your provided dataset
    """
    
    def __init__(self, gpu_host, gpu_port, num_workers, dataloader, preprocessor=None):
        self.provided_dataloader = dataloader
        self.provided_preprocessor = preprocessor
        
        # Initialize parent
        super().__init__(gpu_host, gpu_port, num_workers)
    
    def train_epochs(self, epochs):
        """Train for specified number of epochs using provided dataloader"""
        print("🚀 Starting distributed training with your dataset")
        
        try:
            from tqdm import tqdm  # type: ignore # Import here as fallback
        except ImportError:
            print("⚠️  tqdm not available, falling back to basic progress")
            # Create a simple fallback class
            class tqdm:
                def __init__(self, *args, **kwargs):
                    self.total = kwargs.get('total', 0)
                    self.n = 0
                    self.desc = kwargs.get('desc', '')
                def update(self, n=1):
                    self.n += n
                    print(f"{self.desc}: {self.n}/{self.total}")
                def set_postfix(self, *args, **kwargs):
                    pass
                def set_postfix_str(self, s):
                    pass
                def refresh(self):
                    pass
                def close(self):
                    pass
                @staticmethod
                def write(msg):
                    print(msg)
        
        # Setup epoch progress bar
        self.epoch_pbar = tqdm(
            total=epochs,
            desc="🎯 Training Epochs",
            position=0,
            leave=True
        )
        
        # Setup data sending progress bar
        total_batches = len(self.provided_dataloader) * epochs
        self.data_sent_pbar = tqdm(
            total=total_batches,
            desc="📤 Data Sent",
            position=1,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}] GPU:{postfix}"
        )
        
        # Start background threads
        import threading
        processor_thread = threading.Thread(target=self.batch_processor, daemon=True)
        sender_thread = threading.Thread(target=self.gpu_sender, daemon=True)
        
        processor_thread.start()
        sender_thread.start()
        
        try:
            for epoch in range(epochs):
                epoch_start = time.time()
                batch_count = 0
                
                for batch_data, batch_targets in self.provided_dataloader:
                    # Apply custom preprocessing if provided
                    if self.provided_preprocessor:
                        batch_data, batch_targets = self.provided_preprocessor(batch_data, batch_targets)
                    
                    # CHECK: Should we pause data loading?
                    while self.flow_controller.should_pause_loading() and self.running:
                        tqdm.write("⏸️  Data loading PAUSED - waiting for GPU to catch up...")
                        time.sleep(2.0)
                        gpu_stats = self.get_gpu_stats()
                        self.flow_controller.update_gpu_stats(gpu_stats)
                    
                    if not self.running:
                        break
                    
                    metadata = {
                        'epoch': epoch,
                        'batch_id': batch_count,
                        'augment': True,
                        'normalize': True
                    }
                    
                    # Put with reasonable timeout
                    try:
                        self.batch_queue.put((batch_data, batch_targets, metadata), timeout=5.0)
                    except queue.Full:
                        tqdm.write("⏳ Batch queue full, waiting for processing to catch up...")
                        self.batch_queue.put((batch_data, batch_targets, metadata))
                    
                    batch_count += 1
                    
                    # Update progress bars every few batches
                    if batch_count % 5 == 0:
                        gpu_stats = self.get_gpu_stats()
                        self.flow_controller.update_gpu_stats(gpu_stats)
                        
                        # Update data sent progress bar
                        gpu_processed = self.stats['batches_processed_by_gpu']
                        processing_queue = self.batch_queue.qsize()
                        processed_queue = self.processed_queue.qsize()
                        
                        postfix = f"Processed:{gpu_processed} | ProcQ:{processing_queue}/200 | SendQ:{processed_queue}/100"
                        self.data_sent_pbar.set_postfix_str(postfix)
                        self.data_sent_pbar.n = self.stats['batches_sent']
                        self.data_sent_pbar.refresh()
                
                epoch_time = time.time() - epoch_start
                self.stats['total_epochs'] += 1
                
                # Update epoch progress bar
                self.epoch_pbar.update(1)
                self.epoch_pbar.set_postfix({
                    'Time': f"{epoch_time:.1f}s",
                    'Sent': self.stats['batches_sent'],
                    'GPU': self.stats['batches_processed_by_gpu']
                })
                
        except Exception as e:
            tqdm.write(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Give GPU a moment to process any remaining batches
            tqdm.write("⏳ Allowing GPU time to process remaining batches...")
            time.sleep(3)
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown"""
        import time  # Import time here
        
        # Signal shutdown to background threads
        self.batch_queue.put(None)
        self.processed_queue.put(None)
        
        # Wait for GPU to finish processing all sent batches
        print("⏳ Waiting for GPU to finish processing all batches...")
        
        import time
        while self.running:
            gpu_stats = self.get_gpu_stats()
            gpu_processed = self.stats['batches_processed_by_gpu']
            
            print(f"🔍 Checking completion: GPU processed {gpu_processed}, Master sent {self.stats['batches_sent']}")
            
            if gpu_processed >= self.stats['batches_sent']:
                print("✅ All batches processed by GPU!")
                break
            
            if gpu_processed == 0:
                print("⚠️  Cannot get GPU stats, assuming completion...")
                break
            
            time.sleep(2)
        
        # Shutdown GPU
        if self.shutdown_gpu():
            print("✅ GPU server shutdown successfully")
        else:
            print("⚠️  Could not shutdown GPU server gracefully")
        
        self.print_final_stats()