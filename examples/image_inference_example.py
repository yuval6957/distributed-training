# examples/image_inference_example.py - Memory-efficient inference for image processing

import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from distributed_training import CPULoader

# Simulate image dataset
class ImageDataset(torch.utils.data.Dataset):
    """Simulated image dataset for demonstrating memory-efficient inference"""
    
    def __init__(self, num_images=1000, image_size=(3, 224, 224)):
        print(f"🖼️  Creating simulated image dataset: {num_images:,} images")
        print(f"📐 Image size: {image_size}")
        
        # Simulate image data - normally you'd load from files
        self.data = np.random.randn(num_images, *image_size).astype(np.float32)
        
        # Add some realistic patterns to images
        for i in range(num_images):
            # Add color channel patterns
            self.data[i, 0] += np.random.normal(0.485, 0.229)  # Red channel
            self.data[i, 1] += np.random.normal(0.456, 0.224)  # Green channel  
            self.data[i, 2] += np.random.normal(0.406, 0.225)  # Blue channel
        
        # Flatten for the model (in real scenario, you'd use CNN)
        self.data = self.data.reshape(num_images, -1)
        
        print(f"✅ Image dataset created: {len(self.data):,} images")
        print(f"📊 Memory usage: ~{self.data.nbytes / 1e6:.1f} MB")
        print(f"🔢 Feature size per image: {self.data.shape[1]:,}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def run_memory_efficient_inference():
    """Demonstrate memory-efficient inference for image processing"""
    
    print("🖼️  Image Processing Inference Example")
    print("="*50)
    
    # 1. Create image dataset
    dataset = ImageDataset(num_images=1000, image_size=(3, 224, 224))
    
    # 2. Setup CPU loader
    loader = CPULoader(num_workers=4)
    loader.setup_dataset(
        dataset=dataset,
        batch_size=32,  # Reasonable batch size for images
        shuffle=False
    )
    
    # 3. Memory-efficient inference configurations
    configs = [
        {
            'name': 'Minimal (predictions only)',
            'params': {
                'top_k': 1,
                'return_full_probabilities': False,
                'return_raw_outputs': False,
                'confidence_threshold': 0.0
            }
        },
        {
            'name': 'Top-3 predictions',
            'params': {
                'top_k': 3,
                'return_full_probabilities': False,
                'return_raw_outputs': False,
                'confidence_threshold': 0.0
            }
        },
        {
            'name': 'High confidence only',
            'params': {
                'top_k': 1,
                'return_full_probabilities': False,
                'return_raw_outputs': False,
                'confidence_threshold': 0.8
            }
        },
        {
            'name': 'Full probabilities (MEMORY INTENSIVE!)',
            'params': {
                'top_k': 1,
                'return_full_probabilities': True,
                'return_raw_outputs': False,
                'confidence_threshold': 0.0
            }
        }
    ]
    
    # GPU server connection (adjust IP as needed)
    gpu_host = '192.168.1.100'
    gpu_port = 29500
    
    print(f"\n🌐 Connecting to GPU server: {gpu_host}:{gpu_port}")
    print("💡 Make sure image_inference_gpu.py is running on the GPU server")
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"🔬 Testing: {config['name']}")
        print('='*60)
        
        try:
            results = loader.run_inference(
                gpu_host=gpu_host,
                inference_dataset=dataset,
                gpu_port=gpu_port,
                **config['params']
            )
            
            # Analyze memory usage
            est_memory = results.get('estimated_memory_mb', 0)
            print(f"📊 Results:")
            print(f"   - Samples processed: {results['total_samples']:,}")
            print(f"   - Estimated memory: {est_memory:.1f} MB")
            print(f"   - Data returned: {results['config']}")
            
            if 'confidences' in results:
                avg_confidence = np.mean(results['confidences'])
                print(f"   - Average confidence: {avg_confidence:.3f}")
            
            if config['params']['confidence_threshold'] > 0:
                high_conf_count = sum(1 for c in results['confidences'] 
                                    if c >= config['params']['confidence_threshold'])
                print(f"   - High confidence samples: {high_conf_count}/{results['total_samples']}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            print("   Make sure the GPU server is running and accessible")
    
    print(f"\n{'='*60}")
    print("💡 MEMORY EFFICIENCY TIPS FOR IMAGE PROCESSING:")
    print("="*60)
    print("1. Use top_k instead of full_probabilities for classification")
    print("2. Set confidence_threshold to filter low-confidence predictions")
    print("3. Use larger batch_size to reduce network overhead")
    print("4. Process images in chunks for very large datasets")
    print("5. Consider returning only predictions + confidences for real-time inference")
    print("\n📈 For 1000 ImageNet classes:")
    print("   - Predictions only: ~8 MB for 100k images")
    print("   - Top-5 predictions: ~40 MB for 100k images") 
    print("   - Full probabilities: ~400 MB for 100k images (50x larger!)")

if __name__ == "__main__":
    run_memory_efficient_inference()