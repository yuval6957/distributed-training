#!/usr/bin/env python3
"""
Repository initialization script for distributed-training framework.
Run this after cloning to set up your environment and test connectivity.
"""

import os
import sys
import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and handle errors gracefully"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def setup_virtual_environment():
    """Set up Python virtual environment"""
    print("🐍 Setting up virtual environment...")
    
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    # Activate and install requirements
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
        
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing requirements"):
        return False
    
    print("✅ Virtual environment setup complete!")
    print(f"💡 To activate: {activate_cmd}")
    return True

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test import
        sys.path.insert(0, '.')
        from distributed_training import GPUTrainer, CPULoader, ColabGPUTrainer
        print("✅ All modules imported successfully")
        
        # Test basic functionality
        print("📊 Testing basic functionality...")
        trainer = GPUTrainer(port=29500)
        loader = CPULoader(num_workers=2)
        colab_trainer = ColabGPUTrainer(port=29500)
        
        print("✅ All components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def show_next_steps():
    """Show next steps to the user"""
    print("\n🎉 Setup complete! Here's what you can do next:")
    print("\n📖 Documentation:")
    print("   - README.md: General overview and quick start")
    print("   - GOOGLE_COLAB_SETUP.md: Detailed Colab setup guide")
    print("   - examples/: Complete example scripts")
    
    print("\n🧪 Testing:")
    print("   - python distributed_training/tests/test_connection.py")
    print("   - python distributed_training/tests/simple_test.py cpu")
    print("   - python distributed_training/tests/simple_test.py gpu")
    
    print("\n⚙️ Configuration Setup:")
    print("   1. Copy templates: cp config.yaml.example config.yaml")
    print("   2. Copy env file: cp .env.example .env")
    print("   3. Edit config.yaml with your GPU/CPU server IPs")
    print("   4. Or set environment variables: export GPU_HOST=your_ip")
    
    print("\n🚀 Quick Start:")
    print("   1. Run GPU server: python examples/train_gpu.py")
    print("   2. Run CPU server: python examples/train_cpu.py")
    print("   (Configuration loaded automatically from config.yaml/.env)")
    
    print("\n🌐 For Google Colab:")
    print("   1. Upload to Colab: examples/colab_gpu_example.py")
    print("   2. Run locally: examples/colab_cpu_local.py")
    
    print("\n📖 Documentation:")
    print("   - CONFIGURATION.md: Complete configuration guide")
    print("   - README.md: General overview and examples")
    print("   - Default GPU server: 192.168.1.100:29500")
    print("   - Default CPU server: 192.168.1.200")

def main():
    parser = argparse.ArgumentParser(description="Initialize distributed-training repository")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment setup")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    args = parser.parse_args()
    
    print("🚀 Distributed Training Framework Setup")
    print("=" * 50)
    
    if args.test_only:
        if test_installation():
            show_next_steps()
        return
    
    # Setup virtual environment
    if not args.skip_venv:
        if not setup_virtual_environment():
            print("❌ Virtual environment setup failed. Try running with --skip-venv")
            return
    
    # Test installation
    if test_installation():
        show_next_steps()
    else:
        print("❌ Setup completed but tests failed. Check your Python environment.")

if __name__ == "__main__":
    main()