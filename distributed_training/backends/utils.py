# distributed_training/backends/utils.py
# Complete utility functions and classes shared across backends

import time
import psutil
import socket
import logging
import threading

def check_port_available(host, port):
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False

def get_system_info():
    """Get system information for debugging"""
    import platform
    import torch
    
    info = {
        'platform': platform.platform(),
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / 1e9,
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info

def print_system_info():
    """Print system information"""
    info = get_system_info()
    
    print("🖥️  System Information:")
    print(f"   Platform: {info['platform']}")
    print(f"   CPU cores: {info['cpu_count']}")
    print(f"   Memory: {info['memory_gb']:.1f} GB")
    print(f"   Python: {info['python_version']}")
    print(f"   PyTorch: {info['torch_version']}")
    
    if info['cuda_available']:
        print(f"   GPU: {info['gpu_name']}")
        print(f"   GPU Memory: {info['gpu_memory_gb']:.1f} GB")
    else:
        print("   GPU: Not available")

class ConnectionTester:
    """Test network connections between servers"""
    
    @staticmethod
    def test_connection(host, port, timeout=5):
        """Test if we can connect to a host:port"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                result = s.connect_ex((host, port))
                return result == 0
        except Exception:
            return False
    
    @staticmethod
    def find_available_port(host='localhost', start_port=29500, max_attempts=100):
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            if check_port_available(host, port):
                return port
        return None

class NetworkDebugger:
    """Debug network issues between CPU and GPU servers"""
    
    @staticmethod
    def diagnose_connection(gpu_host, gpu_port, cpu_host=None):
        """Diagnose connection issues"""
        print("🔍 Network Diagnostics")
        print("=" * 40)
        
        # Test GPU server availability
        print(f"Testing GPU server {gpu_host}:{gpu_port}...")
        if ConnectionTester.test_connection(gpu_host, gpu_port):
            print("✅ GPU server is reachable")
        else:
            print("❌ GPU server is NOT reachable")
            print("   - Check if GPU worker is running")
            print("   - Check firewall settings")
            print("   - Check network connectivity")
        
        # Test port availability on CPU side
        if cpu_host:
            print(f"\nTesting CPU server {cpu_host}...")
            if ConnectionTester.test_connection(cpu_host, 22):  # SSH port
                print("✅ CPU server is reachable")
            else:
                print("❌ CPU server is NOT reachable")
        
        # System info
        print("\n🖥️  Local System:")
        print_system_info()
    
    @staticmethod
    def test_bandwidth(host, port, test_size_mb=10, timeout=30):
        """Test network bandwidth between servers"""
        try:
            test_data = b'x' * (test_size_mb * 1024 * 1024)  # Create test data
            
            start_time = time.time()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((host, port))
                s.sendall(test_data)
            
            elapsed = time.time() - start_time
            bandwidth_mbps = (test_size_mb * 8) / elapsed  # Convert to Mbps
            
            print(f"📊 Bandwidth test: {bandwidth_mbps:.1f} Mbps")
            return bandwidth_mbps
            
        except Exception as e:
            print(f"❌ Bandwidth test failed: {e}")
            return 0.0

class PerformanceMonitor:
    """Monitor performance metrics during training"""
    
    def __init__(self):
        self.start_time = time.time()
        self.batch_times = []
        self.memory_usage = []
        
    def log_batch(self, batch_size):
        """Log completion of a batch"""
        current_time = time.time()
        self.batch_times.append(current_time)
        
        # Log memory usage
        memory_percent = psutil.virtual_memory().percent
        self.memory_usage.append(memory_percent)
        
        # Keep only last 100 measurements
        if len(self.batch_times) > 100:
            self.batch_times.pop(0)
            self.memory_usage.pop(0)
    
    def get_throughput(self, batch_size):
        """Calculate current throughput in samples/second"""
        if len(self.batch_times) < 2:
            return 0.0
        
        time_diff = self.batch_times[-1] - self.batch_times[-2]
        if time_diff > 0:
            return batch_size / time_diff
        return 0.0
    
    def get_avg_memory_usage(self):
        """Get average memory usage percentage"""
        if not self.memory_usage:
            return 0.0
        return sum(self.memory_usage) / len(self.memory_usage)
    
    def get_runtime(self):
        """Get total runtime in seconds"""
        return time.time() - self.start_time

class ResourceMonitor:
    """Monitor system resources during training"""
    
    def __init__(self, check_interval=5):
        self.check_interval = check_interval
        self.monitoring = False
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'gpu_memory_used': [],
            'timestamps': []
        }
    
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        print(f"📊 Resource monitoring started (interval: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        print("📊 Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        import torch
        
        while self.monitoring:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # GPU memory if available
                gpu_memory_used = 0
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / 1e9  # GB
                
                # Store stats
                self.stats['cpu_percent'].append(cpu_percent)
                self.stats['memory_percent'].append(memory_percent)
                self.stats['gpu_memory_used'].append(gpu_memory_used)
                self.stats['timestamps'].append(time.time())
                
                # Keep only last 1000 measurements
                for key in self.stats:
                    if len(self.stats[key]) > 1000:
                        self.stats[key] = self.stats[key][-1000:]
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"❌ Resource monitoring error: {e}")
                break
    
    def get_summary(self):
        """Get resource usage summary"""
        if not self.stats['cpu_percent']:
            return "No monitoring data available"
        
        avg_cpu = sum(self.stats['cpu_percent']) / len(self.stats['cpu_percent'])
        avg_memory = sum(self.stats['memory_percent']) / len(self.stats['memory_percent'])
        max_memory = max(self.stats['memory_percent'])
        
        summary = f"""
📊 Resource Usage Summary:
   Average CPU: {avg_cpu:.1f}%
   Average Memory: {avg_memory:.1f}%
   Peak Memory: {max_memory:.1f}%
"""
        
        if any(gpu > 0 for gpu in self.stats['gpu_memory_used']):
            avg_gpu = sum(self.stats['gpu_memory_used']) / len(self.stats['gpu_memory_used'])
            max_gpu = max(self.stats['gpu_memory_used'])
            summary += f"   Average GPU Memory: {avg_gpu:.1f} GB\n"
            summary += f"   Peak GPU Memory: {max_gpu:.1f} GB\n"
        
        return summary

def create_distributed_config(gpu_host, cpu_host, port=29500):
    """Create a configuration dict for distributed training"""
    return {
        'gpu': {
            'host': gpu_host,
            'port': port,
            'device': 'auto'
        },
        'cpu': {
            'host': cpu_host,
            'num_workers': 'auto',
            'batch_size': 64,
            'queue_sizes': {
                'batch_queue': 200,
                'processed_queue': 100
            }
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'gradient_clip': 1.0
        },
        'flow_control': {
            'pause_threshold': 40,
            'slow_threshold': 25,
            'resume_threshold': 10,
            'max_delay': 2.0
        }
    }

def validate_config(config):
    """Validate distributed training configuration"""
    required_keys = ['gpu', 'cpu', 'training']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # Validate GPU config
    if 'host' not in config['gpu']:
        raise ValueError("GPU host not specified")
    
    # Validate CPU config  
    if 'host' not in config['cpu']:
        raise ValueError("CPU host not specified")
    
    return True

class ConfigValidator:
    """Validate and sanitize configuration options"""
    
    @staticmethod
    def validate_network_config(gpu_host, cpu_host, port):
        """Validate network configuration"""
        # Check if hosts are valid IP addresses or hostnames
        try:
            socket.inet_aton(gpu_host)  # Check if valid IP
        except socket.error:
            # Not an IP, try as hostname
            try:
                socket.gethostbyname(gpu_host)
            except socket.gaierror:
                raise ValueError(f"Invalid GPU host: {gpu_host}")
        
        try:
            socket.inet_aton(cpu_host)
        except socket.error:
            try:
                socket.gethostbyname(cpu_host)
            except socket.gaierror:
                raise ValueError(f"Invalid CPU host: {cpu_host}")
        
        # Check port range
        if not (1024 <= port <= 65535):
            raise ValueError(f"Port must be between 1024 and 65535, got {port}")
        
        return True
    
    @staticmethod
    def validate_training_config(epochs, batch_size, learning_rate):
        """Validate training configuration"""
        if epochs <= 0:
            raise ValueError(f"Epochs must be positive, got {epochs}")
        
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        
        if learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {learning_rate}")
        
        return True

def setup_logging(level='INFO', filename=None):
    """Setup logging for distributed training"""
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    date_format = '%H:%M:%S'
    
    # Setup basic config
    if filename:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            datefmt=date_format,
            filename=filename,
            filemode='a'
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            datefmt=date_format
        )
    
    # Create logger for this package
    logger = logging.getLogger('distributed_training')
    
    # Add console handler if logging to file
    if filename:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(log_format, date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

# Convenience functions for quick setup
def quick_gpu_setup(port=29500):
    """Quick setup for GPU server"""
    from .gpu_worker import GPUTrainingWorker
    print_system_info()
    return GPUTrainingWorker(port=port)

def quick_cpu_setup(gpu_host, port=29500):
    """Quick setup for CPU server"""
    from .cpu_master import CPUDataMaster
    print_system_info()
    return CPUDataMaster(gpu_host, port)

# Version info
__version__ = "1.0.0"