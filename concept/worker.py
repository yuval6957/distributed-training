# gpu_worker.py - Optimized for your CPU/GPU split architecture
import socket
import pickle
import torch
import torch.nn as nn
import threading
import queue
import time
import json
from collections import deque
import signal
import sys

class TrainingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TrainingModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

class GPUTrainingWorker:
    def __init__(self, host='0.0.0.0', port=29500, model_config=None):
        self.host = host
        self.port = port
        self.running = True
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 GPU Worker using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Model configuration
        config = model_config or {
            'input_size': 100,
            'hidden_size': 512,
            'output_size': 10,
            'learning_rate': 0.001
        }
        
        # Initialize model
        self.model = TrainingModel(
            config['input_size'], 
            config['hidden_size'], 
            config['output_size']
        ).to(self.device)
        
        # Initialize optimizer and criterion
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        self.batch_count = 0
        self.epoch_losses = deque(maxlen=100)
        self.training_stats = {
            'total_batches': 0,
            'total_samples': 0,
            'average_loss': 0.0,
            'learning_rate': config['learning_rate']
        }
        
        # Batch processing (larger queue for high-throughput CPU)
        self.batch_queue = queue.Queue(maxsize=50)  # Increased from 20
        self.result_queue = queue.Queue(maxsize=100)
        
        # Start background threads
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.stats_thread = threading.Thread(target=self.stats_logger, daemon=True)
        
        self.training_thread.start()
        self.stats_thread.start()
        
        print("✅ GPU Worker initialized successfully")
    
    def training_loop(self):
        """Training loop with clear batch progress"""
        print("🔥 GPU Training started - showing batch progress")
        
        while self.running:
            try:
                # Get batch from queue
                batch_data = self.batch_queue.get(timeout=1.0)
                
                if batch_data is None:  # Shutdown signal
                    break
                
                # Process batch
                loss = self._process_batch(batch_data)
                
                # Update stats
                self.epoch_losses.append(loss)
                self.batch_count += 1
                self.training_stats['total_batches'] += 1
                self.training_stats['total_samples'] += batch_data['batch_size']
                self.training_stats['average_loss'] = sum(self.epoch_losses) / len(self.epoch_losses)
                self.training_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']
                
                # Show current batch being processed
                print(f"⚡ PROCESSING BATCH {self.batch_count} | "
                      f"Epoch {batch_data.get('epoch', '?')} | "
                      f"Loss: {loss:.4f} | "
                      f"Queue: {self.batch_queue.qsize()} remaining")
                
                # Step scheduler
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
        loss = self.criterion(outputs, targets)
        
        # Backward pass
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
                print(f"📊 SUMMARY: {self.training_stats['total_batches']} batches completed, "
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
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'training_stats': self.training_stats,
                    'batch_count': self.batch_count
                }
                return {'status': 'success', 'checkpoint': checkpoint}
            
            elif command == 'load_model':
                # Load model checkpoint
                checkpoint = message.get('checkpoint')
                if checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

# Signal handler for graceful shutdown
worker_instance = None

def signal_handler(sig, frame):
    print("\n🛑 Received interrupt signal")
    if worker_instance:
        worker_instance.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Configuration
    MODEL_CONFIG = {
        'input_size': 100,
        'hidden_size': 512,
        'output_size': 10,
        'learning_rate': 0.001
    }
    
    print("🚀 Starting GPU Training Worker...")
    
    # Create worker
    worker_instance = GPUTrainingWorker(
        host='0.0.0.0', 
        port=29500, 
        model_config=MODEL_CONFIG
    )
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        worker_instance.start()
    except KeyboardInterrupt:
        print("🛑 GPU Worker interrupted by user")
    finally:
        if worker_instance:
            worker_instance.stop()
        print("✅ GPU Worker shut down completely")