# distributed_training/colab_gpu_trainer.py
import torch
import torch.nn as nn
from .gpu_trainer import GPUTrainer, CustomGPUWorker

class ColabGPUTrainer(GPUTrainer):
    """
    Google Colab-specific GPU trainer with ngrok tunnel support.
    Enables training with GPU in Colab and CPU/data on local machine.
    """
    
    def __init__(self, port=29500, device='auto', ngrok_auth_token=None):
        """
        Initialize Colab GPU trainer with ngrok support
        
        Args:
            port: Port to listen on (default: 29500)
            device: 'auto', 'cuda', or 'cpu'
            ngrok_auth_token: Optional ngrok auth token for custom domains
        """
        super().__init__(port, device)
        
        # Import ngrok only when needed
        try:
            from pyngrok import ngrok
            self.ngrok = ngrok
            if ngrok_auth_token:
                ngrok.set_auth_token(ngrok_auth_token)
                print("✅ ngrok auth token set")
        except ImportError:
            print("❌ pyngrok not found. Install with: !pip install pyngrok")
            raise ImportError("pyngrok is required for ColabGPUTrainer")
        
        self.tunnel_url = None
        self.tunnel = None
        
        print("🔗 ColabGPUTrainer initialized with ngrok support")
    
    def start_training_with_tunnel(self, show_local_instructions=True, tunnel_region='us'):
        """
        Start training with automatic ngrok tunnel creation
        
        Args:
            show_local_instructions: Whether to display local CPU setup instructions
            tunnel_region: ngrok tunnel region ('us', 'eu', 'ap', 'au', 'sa', 'jp', 'in')
        
        Returns:
            tuple: (tunnel_host, tunnel_port) for use in local CPU script
        """
        if self.model is None:
            raise ValueError("Must call setup_model() first!")
        
        # Create ngrok tunnel
        print("🌐 Creating ngrok tunnel for Colab...")
        try:
            self.tunnel = self.ngrok.connect(self.port, bind_tls=True)
            self.tunnel_url = self.tunnel.public_url
            
            # Extract host and port
            tunnel_host = self.tunnel_url.replace('https://', '').replace('http://', '')
            tunnel_port = 443 if 'https' in self.tunnel_url else 80
            
            print(f"✅ ngrok tunnel created: {self.tunnel_url}")
            
            if show_local_instructions:
                self._show_local_instructions(tunnel_host, tunnel_port)
            
            # Start the worker
            print(f"🚀 Starting GPU worker on {self.device}")
            print("📡 Waiting for local CPU server to connect...")
            
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
                self._cleanup()
            
            return tunnel_host, tunnel_port
            
        except Exception as e:
            print(f"❌ Failed to create ngrok tunnel: {e}")
            print("💡 Try installing pyngrok: !pip install pyngrok")
            raise
    
    def _show_local_instructions(self, tunnel_host, tunnel_port):
        """Display instructions for local CPU setup"""
        print("\n" + "="*70)
        print("🏠 INSTRUCTIONS FOR YOUR LOCAL COMPUTER:")
        print("="*70)
        print("1. Copy and paste this into your local CPU script:")
        print()
        print(f"   COLAB_GPU_HOST = '{tunnel_host}'")
        print(f"   COLAB_GPU_PORT = {tunnel_port}")
        print()
        print("2. Run your local CPU script with these values")
        print("3. Keep this Colab notebook running!")
        print()
        print("Example local CPU script:")
        print("   python examples/colab_cpu_local.py")
        print("="*70)
        print()
    
    def _cleanup(self):
        """Clean up ngrok tunnel and worker"""
        if self.worker:
            self.worker.stop()
            print("✅ GPU Worker stopped")
        
        if self.tunnel:
            try:
                self.ngrok.disconnect(self.tunnel.public_url)
                print("✅ ngrok tunnel closed")
            except:
                pass
        
        print("✅ ColabGPUTrainer shut down")
    
    def get_tunnel_info(self):
        """Get current tunnel information"""
        if self.tunnel_url:
            tunnel_host = self.tunnel_url.replace('https://', '').replace('http://', '')
            tunnel_port = 443 if 'https' in self.tunnel_url else 80
            return {
                'url': self.tunnel_url,
                'host': tunnel_host,
                'port': tunnel_port,
                'active': True
            }
        return {'active': False}
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self._cleanup()


# Quick setup function for Colab notebooks
def quick_colab_setup(model_class, model_kwargs=None, learning_rate=0.001, ngrok_auth_token=None):
    """
    Quick setup function for Colab notebooks
    
    Args:
        model_class: Your PyTorch model class
        model_kwargs: Keyword arguments for model initialization
        learning_rate: Learning rate for optimizer
        ngrok_auth_token: Optional ngrok auth token
    
    Returns:
        ColabGPUTrainer: Configured trainer ready to use
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    # Create trainer
    trainer = ColabGPUTrainer(ngrok_auth_token=ngrok_auth_token)
    
    # Setup model
    model = model_class(**model_kwargs)
    trainer.setup_model(model, learning_rate=learning_rate)
    
    print("🎉 Colab setup complete! Call trainer.start_training_with_tunnel() to begin")
    return trainer