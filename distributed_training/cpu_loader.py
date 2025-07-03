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
    
    def setup_dataset(self, dataset, batch_size=64, shuffle=True, preprocessing_fn=None, 
                      val_split=0.0, val_dataset=None):
        """
        Setup your dataset and preprocessing with optional train/validation split
        
        Args:
            dataset: Your PyTorch dataset or data path
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            preprocessing_fn: Optional preprocessing function
            val_split: Fraction of data to use for validation (0.0-1.0)
            val_dataset: Optional separate validation dataset
        """
        self.dataset = dataset
        self.preprocessor = preprocessing_fn
        self.val_split = val_split
        self.val_dataset = val_dataset
        
        # Handle different dataset types
        if isinstance(dataset, str):
            # Assume it's a file path
            self.dataset = self._load_from_path(dataset)
        elif isinstance(dataset, (tuple, list)) and len(dataset) == 2:
            # Assume it's (X, y) data
            self.dataset = SimpleDataset(dataset[0], dataset[1])
        elif not isinstance(dataset, Dataset):
            raise ValueError("Dataset must be a PyTorch Dataset, file path, or (X, y) tuple")
        
        # Handle train/validation split
        if val_split > 0.0 and val_dataset is None:
            # Split the dataset
            total_size = len(self.dataset)
            val_size = int(total_size * val_split)
            train_size = total_size - val_size
            
            # Create random split
            import torch
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size]
            )
            
            self.dataset = train_dataset  # Use only training portion
            self.val_dataset = val_dataset
            
            print(f"🔀 Dataset split:")
            print(f"   - Training: {train_size:,} samples ({(1-val_split)*100:.1f}%)")
            print(f"   - Validation: {val_size:,} samples ({val_split*100:.1f}%)")
        
        # Create training DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else 2
        )
        
        # Create validation DataLoader if validation data exists
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=batch_size * 2,  # Larger batch for validation
                shuffle=False,  # Don't shuffle validation
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False
            )
        else:
            self.val_dataloader = None
        
        print(f"✅ Dataset setup complete:")
        print(f"   - Training samples: {len(self.dataset):,}")
        if self.val_dataset:
            print(f"   - Validation samples: {len(self.val_dataset):,}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Batches per epoch: {len(self.dataloader):,}")
        print(f"   - Workers: {self.num_workers}")
        if preprocessing_fn:
            print(f"   - Preprocessing: {preprocessing_fn.__name__}")
    
    def start_loading(self, gpu_host, epochs=10, gpu_port=29500, early_stopping=None):
        """
        Start distributed data loading - sends data to GPU server
        
        Args:
            gpu_host: IP address of the GPU server
            epochs: Number of training epochs
            gpu_port: Port of the GPU server
            early_stopping: Dict with early stopping config (patience, min_delta, monitor)
        """
        if self.dataset is None:
            raise ValueError("Must call setup_dataset() first!")
        
        print(f"🌐 Starting CPU data master, connecting to GPU at {gpu_host}:{gpu_port}")
        print(f"🔄 Training for {epochs} epochs")
        
        if self.val_dataset is not None:
            print("📊 Integrated training with validation enabled")
            if early_stopping:
                print(f"⏹️  Early stopping: patience={early_stopping.get('patience', 5)}")
        
        # Create the master with our dataset
        self.master = TrainingWithValidationMaster(
            gpu_host=gpu_host,
            gpu_port=gpu_port,
            num_workers=self.num_workers,
            train_dataloader=self.dataloader,
            val_dataloader=self.val_dataloader,
            preprocessor=self.preprocessor,
            early_stopping=early_stopping
        )
        
        try:
            # Test connection first
            print("🔍 Testing connection to GPU server...")
            test_response = self.master.send_message_to_gpu({'command': 'get_stats'})
            
            if test_response.get('status') == 'success':
                print("✅ Successfully connected to GPU server")
                
                # Start training with validation
                results = self.master.train_epochs_with_validation(epochs)
                return results
                
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


class TrainingWithValidationMaster(CPUDataMaster):
    """
    CPU master that integrates training with per-epoch validation
    """
    
    def __init__(self, gpu_host, gpu_port, num_workers, train_dataloader, val_dataloader=None, 
                 preprocessor=None, early_stopping=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.preprocessor = preprocessor
        
        # Early stopping configuration
        self.early_stopping = early_stopping or {}
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'epochs': []
        }
        
        # Initialize parent
        super().__init__(gpu_host, gpu_port, num_workers)
    
    def train_epochs_with_validation(self, epochs):
        """Train with integrated validation after each epoch"""
        print("🚀 Starting training with integrated validation")
        
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback tqdm implementation
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
                def close(self):
                    pass
                @staticmethod
                def write(msg):
                    print(msg)
        
        # Setup overall progress tracking
        epoch_pbar = tqdm(
            total=epochs,
            desc="🎯 Training Progress",
            position=0,
            leave=True
        )
        
        # Start background threads for batch processing
        import threading
        processor_thread = threading.Thread(target=self.batch_processor, daemon=True)
        sender_thread = threading.Thread(target=self.gpu_sender, daemon=True)
        
        processor_thread.start()
        sender_thread.start()
        
        try:
            for epoch in range(epochs):
                print(f"\n{'='*60}")
                print(f"🔄 EPOCH {epoch + 1}/{epochs}")
                print('='*60)
                
                # 1. Training phase
                epoch_start = time.time()
                train_loss = self._train_one_epoch(epoch)
                
                # 2. Validation phase (if validation data exists)
                val_results = None
                if self.val_dataloader is not None:
                    print("\n📊 Running validation...")
                    val_results = self._validate_one_epoch()
                    
                    # Update training history
                    self.training_history['train_loss'].append(train_loss)
                    self.training_history['val_loss'].append(val_results['average_loss'])
                    self.training_history['val_accuracy'].append(val_results['accuracy'])
                    self.training_history['epochs'].append(epoch + 1)
                    
                    # Print epoch summary
                    print(f"\n📈 Epoch {epoch + 1} Summary:")
                    print(f"   - Train Loss: {train_loss:.4f}")
                    print(f"   - Val Loss: {val_results['average_loss']:.4f}")
                    print(f"   - Val Accuracy: {val_results['accuracy']:.4f}")
                    
                    # Check for early stopping
                    if self._should_early_stop(val_results):
                        print(f"\n⏹️  Early stopping triggered at epoch {epoch + 1}")
                        break
                
                epoch_time = time.time() - epoch_start
                self.stats['total_epochs'] += 1
                
                # Update progress bar
                epoch_pbar.update(1)
                postfix = {'Train Loss': f"{train_loss:.4f}"}
                if val_results:
                    postfix.update({
                        'Val Loss': f"{val_results['average_loss']:.4f}",
                        'Val Acc': f"{val_results['accuracy']:.4f}"
                    })
                epoch_pbar.set_postfix(postfix)
                
        finally:
            epoch_pbar.close()
            
            # Final summary
            print(f"\n{'='*60}")
            print("🏁 TRAINING COMPLETED")
            print('='*60)
            self.print_training_summary()
        
        return self.training_history
    
    def _train_one_epoch(self, epoch):
        """Train for one epoch and return average loss"""
        epoch_losses = []
        batch_count = 0
        
        train_pbar = tqdm(
            total=len(self.train_dataloader),
            desc="🔥 Training",
            position=1,
            leave=False
        )
        
        try:
            for batch_data, batch_targets in self.train_dataloader:
                # Apply preprocessing if provided
                if self.preprocessor:
                    batch_data, batch_targets = self.preprocessor(batch_data, batch_targets)
                
                # Send training batch to GPU
                metadata = {
                    'epoch': epoch,
                    'batch_id': batch_count,
                    'augment': True,
                    'normalize': True
                }
                
                # Queue the batch for processing
                try:
                    self.batch_queue.put((batch_data, batch_targets, metadata), timeout=5.0)
                except queue.Full:
                    tqdm.write("⏳ Batch queue full, waiting...")
                    self.batch_queue.put((batch_data, batch_targets, metadata))
                
                batch_count += 1
                train_pbar.update(1)
                
                # Periodically check GPU stats
                if batch_count % 10 == 0:
                    gpu_stats = self.get_gpu_stats()
                    if gpu_stats:
                        recent_results = gpu_stats.get('recent_results', [])
                        if recent_results:
                            recent_loss = recent_results[-1].get('loss', 0.0)
                            epoch_losses.append(recent_loss)
                            train_pbar.set_postfix({'Loss': f"{recent_loss:.4f}"})
        
        finally:
            train_pbar.close()
        
        # Return average training loss for this epoch
        return sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    
    def _validate_one_epoch(self):
        """Run validation for current epoch"""
        return self.validate_dataset(self.val_dataloader, show_progress=True)
    
    def _should_early_stop(self, val_results):
        """Check if training should stop early based on validation results"""
        if not self.early_stopping:
            return False
        
        monitor = self.early_stopping.get('monitor', 'val_loss')
        patience = self.early_stopping.get('patience', 5)
        min_delta = self.early_stopping.get('min_delta', 0.001)
        
        current_val_loss = val_results['average_loss']
        current_val_acc = val_results['accuracy']
        
        if monitor == 'val_loss':
            # Lower is better for loss
            if current_val_loss < (self.best_val_loss - min_delta):
                self.best_val_loss = current_val_loss
                self.patience_counter = 0
                print(f"✅ New best validation loss: {current_val_loss:.4f}")
                return False
            else:
                self.patience_counter += 1
        
        elif monitor == 'val_accuracy':
            # Higher is better for accuracy
            if current_val_acc > (self.best_val_acc + min_delta):
                self.best_val_acc = current_val_acc
                self.patience_counter = 0
                print(f"✅ New best validation accuracy: {current_val_acc:.4f}")
                return False
            else:
                self.patience_counter += 1
        
        if self.patience_counter >= patience:
            return True
        
        print(f"⏳ No improvement for {self.patience_counter}/{patience} epochs")
        return False
    
    def print_training_summary(self):
        """Print comprehensive training summary"""
        history = self.training_history
        if not history['epochs']:
            print("No training history available")
            return
        
        print(f"📊 Training Summary:")
        print(f"   - Epochs completed: {len(history['epochs'])}")
        print(f"   - Final train loss: {history['train_loss'][-1]:.4f}")
        
        if history['val_loss']:
            print(f"   - Final val loss: {history['val_loss'][-1]:.4f}")
            print(f"   - Final val accuracy: {history['val_accuracy'][-1]:.4f}")
            print(f"   - Best val loss: {min(history['val_loss']):.4f}")
            print(f"   - Best val accuracy: {max(history['val_accuracy']):.4f}")
        
        # Print basic stats from parent
        runtime = time.time() - self.stats['start_time']
        print(f"   - Total runtime: {runtime:.1f}s")
        print(f"   - Batches sent: {self.stats['batches_sent']}")


class CustomCPUMaster(CPUDataMaster):
    """
    Custom CPU master that uses your provided dataset (legacy - for backward compatibility)
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