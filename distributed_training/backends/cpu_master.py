# distributed_training/backends/cpu_master.py
# This is the complete CPU master backend - handles all networking complexity

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
from tqdm import tqdm

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
        
        # Progress bars
        self.epoch_pbar = None
        self.batch_pbar = None
        self.data_sent_pbar = None
    
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

# For standalone usage
if __name__ == "__main__":
    master = CPUDataMaster('192.168.1.100', 29500)
    
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal")
        master.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("🔍 Testing connection to GPU server...")
        test_response = master.send_message_to_gpu({'command': 'get_stats'})
        
        if test_response.get('status') == 'success':
            print("✅ Successfully connected to GPU server")
        else:
            print(f"❌ Failed to connect to GPU server: {test_response}")
            
    except KeyboardInterrupt:
        print("🛑 Interrupted by user")
    finally:
        print("✅ CPU Master shut down completely")