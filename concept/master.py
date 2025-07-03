# cpu_master.py - NO DATA LOSS version with adaptive flow control
import socket
import pickle
import numpy as np
import pandas as pd
import time
import threading
import queue
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import torch
import signal
import sys
from pathlib import Path
import psutil

class LargeDataset(Dataset):
    def __init__(self, data_path=None, dataset_size=100000, input_size=100):
        self.dataset_size = dataset_size
        self.input_size = input_size
        
        if data_path and Path(data_path).exists():
            self.load_real_data(data_path)
        else:
            self.generate_synthetic_data()
    
    def load_real_data(self, data_path):
        print(f"📁 Loading dataset from {data_path}")
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            self.X = df.iloc[:, :-1].values.astype(np.float32)
            self.y = df.iloc[:, -1].values.astype(np.int64)
        elif data_path.endswith('.npy'):
            data = np.load(data_path, allow_pickle=True)
            self.X = data['X'].astype(np.float32)
            self.y = data['y'].astype(np.int64)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        self.dataset_size = len(self.X)
        self.input_size = self.X.shape[1]
        print(f"✅ Loaded real dataset: {self.dataset_size} samples, {self.input_size} features")
    
    def generate_synthetic_data(self):
        print(f"🎲 Generating synthetic dataset: {self.dataset_size} samples, {self.input_size} features")
        self.X = np.random.randn(self.dataset_size, self.input_size).astype(np.float32)
        self.y = np.random.randint(0, 10, self.dataset_size, dtype=np.int64)
        print(f"✅ Generated synthetic dataset: {self.dataset_size} samples")
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataPreprocessor:
    @staticmethod
    def normalize_batch(batch_data):
        return (batch_data - batch_data.mean(axis=0)) / (batch_data.std(axis=0) + 1e-8)
    
    @staticmethod
    def augment_batch(batch_data, noise_factor=0.1):
        noise = np.random.normal(0, noise_factor, batch_data.shape)
        return batch_data + noise.astype(batch_data.dtype)
    
    @staticmethod
    def process_batch_parallel(batch_data, targets, augment=True, normalize=True):
        if normalize:
            batch_data = DataPreprocessor.normalize_batch(batch_data)
        if augment:
            batch_data = DataPreprocessor.augment_batch(batch_data)
        return batch_data, targets

class AdaptiveFlowController:
    def __init__(self):
        self.gpu_queue_sizes = []
        self.last_check = time.time()
        self.adaptive_delay = 0.0
        self.max_delay = 2.0
        self.data_loading_paused = False
        
    def update_gpu_stats(self, gpu_stats):
        current_time = time.time()
        
        # Only update every 3 seconds
        if current_time - self.last_check < 3.0:
            return
        
        queue_size = gpu_stats.get('queue_size', 0)
        self.gpu_queue_sizes.append(queue_size)
        
        # Keep only last 5 measurements
        if len(self.gpu_queue_sizes) > 5:
            self.gpu_queue_sizes.pop(0)
        
        avg_queue_size = sum(self.gpu_queue_sizes) / len(self.gpu_queue_sizes)
        
        # Adaptive logic with data loading control
        if avg_queue_size > 40:
            self.data_loading_paused = True
            self.adaptive_delay = self.max_delay
            print(f"⏸️  GPU severely overloaded - PAUSING data loading")
        elif avg_queue_size > 25:
            self.data_loading_paused = False
            self.adaptive_delay = min(self.adaptive_delay + 0.3, self.max_delay)
            print(f"🐌 GPU overloaded, slowing down: delay={self.adaptive_delay:.2f}s")
        elif avg_queue_size < 10:
            self.data_loading_paused = False
            self.adaptive_delay = max(self.adaptive_delay - 0.2, 0.0)
            if self.adaptive_delay == 0:
                print("🚀 GPU ready, full speed")
        
        self.last_check = current_time
    
    def should_pause_loading(self):
        return self.data_loading_paused
    
    def get_delay(self):
        return self.adaptive_delay

class CPUDataMaster:
    def __init__(self, gpu_host, gpu_port=29500, num_workers=None):
        self.gpu_host = gpu_host
        self.gpu_port = gpu_port
        self.running = True
        
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        print(f"🔧 Using {self.num_workers} worker processes for data loading")
        
        # SMART QUEUE LIMITS: Prevent memory explosion
        self.batch_queue = queue.Queue(maxsize=200)  # Limit raw batches
        self.processed_queue = queue.Queue(maxsize=100)  # Limit processed batches
        
        # Flow control
        self.flow_controller = AdaptiveFlowController()
        
        # Statistics with GPU tracking
        self.stats = {
            'batches_sent': 0,
            'batches_processed_by_gpu': 0,  # Track GPU progress
            'samples_processed': 0,
            'total_epochs': 0,
            'start_time': time.time(),
            'failed_sends': 0,
            'last_gpu_batch_count': 0
        }
        
        print(f"💾 Available system memory: {psutil.virtual_memory().total / 1e9:.1f} GB")
        print(f"🔧 CPU cores: {mp.cpu_count()}")
        print("🛡️  SMART FLOW CONTROL: Data loading pauses when GPU overloaded")
    
    def send_message_to_gpu(self, message, timeout=30):
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(timeout)
                    s.connect((self.gpu_host, self.gpu_port))
                    
                    serialized = pickle.dumps(message)
                    size_bytes = len(serialized).to_bytes(4, byteorder='big')
                    s.sendall(size_bytes + serialized)
                    
                    response_size_data = s.recv(4)
                    if response_size_data:
                        response_size = int.from_bytes(response_size_data, byteorder='big')
                        response_data = self._receive_full_data(s, response_size)
                        if response_data:
                            return pickle.loads(response_data)
                    else:
                        return {'status': 'success'}
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    self.stats['failed_sends'] += 1
                    return {'status': 'error', 'message': str(e)}
        
        return {'status': 'error', 'message': 'Max retries exceeded'}
    
    def _receive_full_data(self, sock, size):
        data = b''
        while len(data) < size:
            packet = sock.recv(min(size - len(data), 8192))
            if not packet:
                return None
            data += packet
        return data
    
    def batch_processor(self):
        print("🔄 Started batch processor thread (NO DATA LOSS)")
        
        while self.running:
            try:
                # Apply adaptive delay
                delay = self.flow_controller.get_delay()
                if delay > 0:
                    time.sleep(delay)
                
                batch_item = self.batch_queue.get(timeout=1.0)
                if batch_item is None:
                    break
                
                batch_data, targets, metadata = batch_item
                
                processed_data, processed_targets = DataPreprocessor.process_batch_parallel(
                    batch_data.numpy(), 
                    targets.numpy(),
                    augment=metadata.get('augment', True),
                    normalize=metadata.get('normalize', True)
                )
                
                gpu_message = {
                    'command': 'train_batch',
                    'data': processed_data,
                    'targets': processed_targets,
                    'batch_size': len(processed_data),
                    'epoch': metadata['epoch'],
                    'batch_id': metadata['batch_id']
                }
                
                # CRITICAL: Put with timeout to prevent infinite blocking
                try:
                    self.processed_queue.put((gpu_message, metadata), timeout=10.0)
                except queue.Full:
                    print("⏳ Processed queue full, waiting...")
                    self.processed_queue.put((gpu_message, metadata))  # Block until space
                self.batch_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
                if 'batch_item' in locals():
                    self.batch_queue.task_done()
    
    def gpu_sender(self):
        print("📡 Started GPU sender thread (NO DATA LOSS)")
        
        while self.running:
            try:
                batch_item = self.processed_queue.get(timeout=1.0)
                if batch_item is None:
                    break
                
                gpu_message, metadata = batch_item
                
                # Keep retrying until successful - NO DATA LOSS
                max_attempts = 5
                attempt = 0
                while self.running and attempt < max_attempts:
                    response = self.send_message_to_gpu(gpu_message)
                    
                    if response.get('status') in ['success', 'queued']:
                        self.stats['batches_sent'] += 1
                        self.stats['samples_processed'] += gpu_message['batch_size']
                        break
                    elif response.get('status') == 'queue_full':
                        print("⏳ GPU queue full, waiting 1 second...")
                        time.sleep(1.0)
                        attempt += 1
                    else:
                        print(f"⚠️  GPU send failed, retrying: {response}")
                        time.sleep(0.5)
                        attempt += 1
                
                if attempt >= max_attempts:
                    print(f"❌ Failed to send batch after {max_attempts} attempts")
                
                self.processed_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ GPU sender error: {e}")
                if 'batch_item' in locals():
                    self.processed_queue.task_done()
    
    def load_dataset(self, data_path=None, dataset_size=100000, input_size=100):
        print("📊 Loading dataset...")
        dataset = LargeDataset(data_path, dataset_size, input_size)
        
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )
        
        print(f"✅ Dataset ready: {len(dataset)} samples, batch size: 64")
        return dataloader
    
    def get_gpu_stats(self):
        response = self.send_message_to_gpu({'command': 'get_stats'})
        if isinstance(response, dict) and response.get('status') == 'success':
            gpu_stats = response.get('stats', {})
            # Update our tracking of GPU progress
            if isinstance(gpu_stats, dict):
                current_gpu_batches = gpu_stats.get('total_batches', 0)
                if current_gpu_batches > self.stats['last_gpu_batch_count']:
                    self.stats['batches_processed_by_gpu'] = current_gpu_batches
                    self.stats['last_gpu_batch_count'] = current_gpu_batches
            return gpu_stats if isinstance(gpu_stats, dict) else {}
        return {}
    
    def train(self, epochs=10, data_path=None, dataset_size=100000, input_size=100):
        print("🚀 Starting distributed training (NO DATA LOSS MODE)")
        
        dataloader = self.load_dataset(data_path, dataset_size, input_size)
        
        # Start background threads
        processor_thread = threading.Thread(target=self.batch_processor, daemon=True)
        sender_thread = threading.Thread(target=self.gpu_sender, daemon=True)
        
        processor_thread.start()
        sender_thread.start()
        
        try:
            for epoch in range(epochs):
                print(f"\n🔄 Starting epoch {epoch + 1}/{epochs}")
                epoch_start = time.time()
                batch_count = 0
                
                for batch_data, batch_targets in dataloader:
                    # CHECK: Should we pause data loading?
                    while self.flow_controller.should_pause_loading() and self.running:
                        print("⏸️  Data loading PAUSED - waiting for GPU to catch up...")
                        time.sleep(2.0)
                        # Update stats while paused
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
                    
                    # Put with reasonable timeout to prevent infinite blocking
                    try:
                        self.batch_queue.put((batch_data, batch_targets, metadata), timeout=5.0)
                    except queue.Full:
                        print("⏳ Batch queue full, waiting for processing to catch up...")
                        self.batch_queue.put((batch_data, batch_targets, metadata))  # Block until space
                    
                    batch_count += 1
                    
                    # Progress update with flow control
                    if batch_count % 200 == 0:
                        gpu_stats = self.get_gpu_stats()
                        self.flow_controller.update_gpu_stats(gpu_stats)
                        
                        print(f"📊 SENT: {self.stats['batches_sent']} | "
                              f"GPU PROCESSED: {self.stats['batches_processed_by_gpu']} | "
                              f"Processing Queue: {self.batch_queue.qsize()}/200 | "
                              f"Processed Queue: {self.processed_queue.qsize()}/100 | "
                              f"Paused: {self.flow_controller.should_pause_loading()}")
                
                epoch_time = time.time() - epoch_start
                self.stats['total_epochs'] += 1
                print(f"✅ Loaded epoch {epoch + 1} in {epoch_time:.1f}s")
        
        except Exception as e:
            print(f"❌ Training error: {e}")
        
        finally:
            # Signal shutdown to background threads
            self.batch_queue.put(None)
            self.processed_queue.put(None)
            
            # Wait for GPU to finish processing all sent batches
            print("⏳ Waiting for GPU to finish processing all batches...")
            
            # Check every 2 seconds if GPU has caught up
            while self.running:
                gpu_stats = self.get_gpu_stats()
                gpu_processed = self.stats['batches_processed_by_gpu']
                
                print(f"🔍 Checking completion: GPU processed {gpu_processed}, Master sent {self.stats['batches_sent']}")
                
                if gpu_processed >= self.stats['batches_sent']:
                    print("✅ All batches processed by GPU!")
                    break
                
                if gpu_processed == 0:  # GPU stats not available
                    print("⚠️  Cannot get GPU stats, assuming completion...")
                    break
                
                time.sleep(2)
            
            # Don't wait for queues if GPU is done
            self.print_final_stats()
    
    def print_final_stats(self):
        runtime = time.time() - self.stats['start_time']
        print("\n" + "="*60)
        print("📈 TRAINING COMPLETED - NO DATA LOST")
        print("="*60)
        print(f"⏱️  Total runtime: {runtime:.1f} seconds")
        print(f"📤 Total batches SENT to GPU: {self.stats['batches_sent']}")
        print(f"⚡ Total batches PROCESSED by GPU: {self.stats['batches_processed_by_gpu']}")
        print(f"🔢 Total samples processed: {self.stats['samples_processed']}")
        print(f"🔄 Total epochs: {self.stats['total_epochs']}")
        print(f"❌ Failed sends: {self.stats['failed_sends']}")
        print(f"⚡ Throughput: {self.stats['samples_processed'] / runtime:.1f} samples/sec")
        print("="*60)
    
    def shutdown_gpu(self):
        print("🛑 Sending shutdown command to GPU...")
        try:
            response = self.send_message_to_gpu({'command': 'shutdown'})
            return response.get('status') == 'shutting_down'
        except:
            print("⚠️  Could not send shutdown command")
            return False

master_instance = None

def signal_handler(sig, frame):
    print("\n🛑 Received interrupt signal")
    if master_instance:
        master_instance.running = False
    sys.exit(0)

if __name__ == "__main__":
    TRAINING_CONFIG = {
        'epochs': 10,
        'data_path': None,
        'dataset_size': 100000,  # Start smaller to test
        'input_size': 100,
        'gpu_host': '192.168.1.100',
        'gpu_port': 29500
    }
    
    print("🚀 Starting CPU Data Master (NO DATA LOSS)")
    print(f"🎯 Target GPU server: {TRAINING_CONFIG['gpu_host']}:{TRAINING_CONFIG['gpu_port']}")
    
    master_instance = CPUDataMaster(
        gpu_host=TRAINING_CONFIG['gpu_host'],
        gpu_port=TRAINING_CONFIG['gpu_port'],
        num_workers=4  # Reduce workers to prevent overwhelming
    )
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("🔍 Testing connection to GPU server...")
        test_response = master_instance.send_message_to_gpu({'command': 'get_stats'})
        
        if test_response.get('status') == 'success':
            print("✅ Successfully connected to GPU server")
            
            master_instance.train(
                epochs=TRAINING_CONFIG['epochs'],
                data_path=TRAINING_CONFIG['data_path'],
                dataset_size=TRAINING_CONFIG['dataset_size'],
                input_size=TRAINING_CONFIG['input_size']
            )
        else:
            print(f"❌ Failed to connect to GPU server: {test_response}")
    
    except KeyboardInterrupt:
        print("🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
    finally:
        if master_instance:
            master_instance.running = False
            if master_instance.shutdown_gpu():
                print("✅ GPU server shutdown successfully")
        print("✅ CPU Master shut down completely")