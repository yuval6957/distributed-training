# test_connection.py - Run this first to test network connectivity

try:
    from ..backends import NetworkDebugger, print_system_info
    from ..config import get_config
except ImportError:
    from distributed_training.backends import NetworkDebugger, print_system_info
    from distributed_training.config import get_config

def test_network():
    print("🔍 Testing Distributed Training Network Setup")
    print("=" * 50)
    
    # System info
    print_system_info()
    
    # Get network configuration
    config = get_config()
    network_config = config.get_network_config()
    
    GPU_HOST = network_config['gpu_host']
    CPU_HOST = network_config['cpu_host']
    PORT = network_config['port']
    
    print(f"\n🌐 Testing network between:")
    print(f"   GPU Server: {GPU_HOST}:{PORT}")
    print(f"   CPU Server: {CPU_HOST}")
    print(f"💡 Configuration loaded from: config.yaml and environment variables")
    
    # Run diagnostics
    NetworkDebugger.diagnose_connection(GPU_HOST, PORT, CPU_HOST)
    
    print("\n✅ Network test complete!")
    print("If you see connection errors above, fix them before proceeding.")

if __name__ == "__main__":
    test_network()