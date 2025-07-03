# distributed_training/backends/gpu_worker.py
# This is the complete GPU worker backend - handles all networking complexity

import socket
import pickle
import torch
import torch.nn as nn
import threading
import queue
import time
from collections import deque
import signal
import sys
from tqdm import tqdm

class GPUTrainingWorker:
    def __init__(self, host='0.0.0.0', port=29500, model=None, optimizer=None, criterion=None, device=None):
        self.host = host
        self.port = port
        self.running = True
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"🚀 GPU Worker using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Use provided model components or create defaults
        if model is not None:
            self.model = model.to(self.device)
            self.optimizer = optimizer
            self.criterion = criterion
        else:
            # Create default model for testing
            self.model = self._create_default_model().to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
            self.criterion = nn.CrossEntropyLoss()
        
        # Add scheduler
        if self.optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        else:
            self.scheduler = None
        
        # Training metrics
        self.batch_count = 0
        self.epoch_losses = deque(maxlen=100)
        self.training_stats = {
            'total_batches': 0,
            'total_samples': 0,
            'average_loss': 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer is not None else 0.0
        }
        
        # Batch processing
        self.batch_queue = queue.Queue(maxsize=50)
        self.result_queue = queue.Queue(maxsize=100)
        
        # Progress bars
        self.batch_pbar = None
        self.epoch_pbar = None
        self.current_epoch = 0
        self.batches_in_current_epoch = 0
        
        # Start background threads
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.stats_thread = threading.Thread(target=self.stats_logger, daemon=True)
        
        self.training_thread.start()
        self.stats_thread.start()
        
        print("✅ GPU Worker initialized successfully")
    
    def _create_default_model(self):
        """Create default model for testing"""
        return nn.Sequential(
            nn.Linear(100, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )
    
    def training_loop(self):
        """Training loop with tqdm progress bars"""
        print("🔥 GPU Training started with progress bars")
        
        while self.running:
            try:
                # Get batch from queue
                batch_data = self.batch_queue.get(timeout=1.0)
                
                if batch_data is None:  # Shutdown signal
                    break
                
                # Check if we're starting a new epoch
                batch_epoch = batch_data.get('epoch', 0)
                if batch_epoch != self.current_epoch:
                    # Close previous epoch progress bar
                    if self.epoch_pbar is not None:
                        self.epoch_pbar.close()
                    
                    # Start new epoch progress bar
                    self.current_epoch = batch_epoch
                    self.batches_in_current_epoch = 0
                    self.epoch_pbar = tqdm(
                        desc=f"🎯 Epoch {self.current_epoch + 1}",
                        position=0,
                        leave=True,
                        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]"
                    )
                
                # Initialize batch progress bar if needed
                if self.batch_pbar is None:
                    self.batch_pbar = tqdm(
                        desc="⚡ Processing Batches",
                        position=1,
                        leave=False,
                        bar_format="{desc}: {n} | Loss: {postfix} | Queue: {rate_fmt}"
                    )
                
                # Process batch
                loss = self._process_batch(batch_data)
                
                # Update stats
                self.epoch_losses.append(loss)
                self.batch_count += 1
                self.batches_in_current_epoch += 1
                self.training_stats['total_batches'] += 1
                self.training_stats['total_samples'] += batch_data['batch_size']
                self.training_stats['average_loss'] = sum(self.epoch_losses) / len(self.epoch_losses)
                if self.optimizer is not None:
                    self.training_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
                
                # Update progress bars
                queue_size = self.batch_queue.qsize()
                self.batch_pbar.set_postfix_str(f"{loss:.4f}")
                self.batch_pbar.set_description_str(f"⚡ Batch {self.batch_count} | Queue: {queue_size}")
                self.batch_pbar.update(1)
                
                if self.epoch_pbar is not None:
                    self.epoch_pbar.update(1)
                    self.epoch_pbar.set_postfix({
                        'Avg Loss': f"{self.training_stats['average_loss']:.4f}",
                        'LR': f"{self.training_stats['learning_rate']:.6f}"
                    })
                
                # Step scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Add result to queue for CPU server
                result = {
                    'batch_id': batch_data.get('batch_id', 0),
                    'epoch': batch_data.get('epoch', 0),
                    'loss': loss,
                    'samples_processed': batch_data['batch_size']
                }
                
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    self.result_queue.get()  # Remove oldest result
                    self.result_queue.put_nowait(result)
                
                self.batch_queue.task_done()
                
            except queue.Empty:
                # If we're not running anymore and queue is empty, exit
                if not self.running:
                    break
                continue
            except Exception as e:
                print(f"❌ Training error: {e}")
                if 'batch_data' in locals():
                    self.batch_queue.task_done()
        
        # Close progress bars on shutdown
        if self.batch_pbar is not None:
            self.batch_pbar.close()
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
    
    def _process_batch(self, batch_data):
        """Process a single training batch"""
        # Move data to device
        data = torch.tensor(
            batch_data['data'], 
            dtype=torch.float32, 
            device=self.device
        )
        targets = torch.tensor(
            batch_data['targets'], 
            dtype=torch.long, 
            device=self.device
        )
        
        # Forward pass
        self.model.train()
        outputs = self.model(data)
        if self.criterion is not None:
            loss = self.criterion(outputs, targets)
        else:
            # Default loss if no criterion provided
            loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        # Backward pass
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        return loss.item()
    
    def stats_logger(self):
        """Background thread for summary statistics only"""
        while self.running:
            time.sleep(30)  # Summary every 30 seconds only
            if self.batch_count > 0:
                # Print summary without interfering with progress bars
                tqdm.write(f"📊 SUMMARY: {self.training_stats['total_batches']} batches completed, "
                          f"Avg Loss: {self.training_stats['average_loss']:.4f}")
    
    def handle_connection(self, conn, addr):
        """Handle incoming requests from CPU server"""
        # Remove connection logging - too verbose
        
        try:
            while self.running:
                # Receive message size
                size_data = conn.recv(4)
                if not size_data:
                    break
                
                message_size = int.from_bytes(size_data, byteorder='big')
                
                # Receive full message
                message_data = self._receive_full_message(conn, message_size)
                if not message_data:
                    break
                
                # Process message
                response = self._process_message(message_data)
                
                # Send response
                if response:
                    response_data = pickle.dumps(response)
                    size_bytes = len(response_data).to_bytes(4, byteorder='big')
                    conn.sendall(size_bytes + response_data)
                else:
                    conn.sendall(b'ACK\n')
                
        except Exception as e:
            # Only log actual errors, not normal disconnections
            if "Connection reset" not in str(e) and "Broken pipe" not in str(e):
                print(f"❌ Connection error: {e}")
        finally:
            conn.close()
    
    def _receive_full_message(self, conn, message_size):
        """Receive complete message of specified size"""
        data = b''
        while len(data) < message_size:
            packet = conn.recv(min(message_size - len(data), 8192))
            if not packet:
                return None
            data += packet
        return data
    
    def _process_message(self, message_data):
        """Process different types of messages from CPU server"""
        try:
            message = pickle.loads(message_data)
            command = message.get('command')
            
            if command == 'train_batch':
                # Add batch to training queue
                try:
                    self.batch_queue.put(message, timeout=0.1)
                    return {'status': 'queued'}
                except queue.Full:
                    return {'status': 'queue_full', 'queue_size': self.batch_queue.qsize()}
            
            elif command == 'get_stats':
                # Return current training statistics
                return {
                    'status': 'success',
                    'stats': self.training_stats.copy(),
                    'queue_size': self.batch_queue.qsize(),
                    'recent_results': list(self.result_queue.queue)[-10:]  # Last 10 results
                }
            
            elif command == 'save_model':
                # Save model checkpoint
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'training_stats': self.training_stats,
                    'batch_count': self.batch_count
                }
                if self.optimizer is not None:
                    checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                return {'status': 'success', 'checkpoint': checkpoint}
            
            elif command == 'load_model':
                # Load model checkpoint
                checkpoint = message.get('checkpoint')
                if checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    return {'status': 'loaded'}
                return {'status': 'error', 'message': 'No checkpoint provided'}
            
            elif command == 'shutdown':
                print("🛑 Received shutdown command from CPU")
                self.running = False
                self.batch_queue.put(None)  # Signal training thread
                return {'status': 'shutting_down'}
            
            else:
                return {'status': 'error', 'message': f'Unknown command: {command}'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def start(self):
        """Start the GPU worker server"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.bind((self.host, self.port))
                server_socket.listen(5)
                
                print(f"🌐 GPU Worker listening on {self.host}:{self.port}")
                print("⏳ Waiting for CPU server connection...")
                
                while self.running:
                    try:
                        server_socket.settimeout(1.0)
                        conn, addr = server_socket.accept()
                        
                        # Handle each connection in a separate thread
                        thread = threading.Thread(
                            target=self.handle_connection, 
                            args=(conn, addr),
                            daemon=True
                        )
                        thread.start()
                        
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"❌ Server error: {e}")
                            
        except Exception as e:
            print(f"❌ Failed to start GPU worker: {e}")
        finally:
            print("🏁 GPU Worker server stopped")
    
    def stop(self):
        """Gracefully stop the worker"""
        print("🛑 Stopping GPU worker...")
        self.running = False
        self.batch_queue.put(None)

# For standalone usage
if __name__ == "__main__":
    worker = GPUTrainingWorker()
    
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal")
        worker.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        worker.start()
    except KeyboardInterrupt:
        print("🛑 GPU Worker interrupted by user")
    finally:
        worker.stop()
        print("✅ GPU Worker shut down completely")