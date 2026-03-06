"""
GPU Configuration for TensorFlow with RTX 4060
Ensures efficient memory usage for 16GB RAM system
"""

import tensorflow as tf
import numpy as np

def configure_gpu(gpu_memory_limit_mb=12000):
    """
    Configure TensorFlow GPU settings
    
    Args:
        gpu_memory_limit_mb: Maximum GPU memory (12000 MB = 12GB)
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠️  No GPU detected. Using CPU mode.")
        return False
    
    try:
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set memory limit
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=gpu_memory_limit_mb
            )]
        )
        
        print(f"✓ GPU configured: {len(gpus)} device(s)")
        print(f"✓ Memory limit: {gpu_memory_limit_mb} MB")
        return True
        
    except Exception as e:
        print(f"❌ GPU configuration failed: {e}")
        return False

def get_device_info():
    """Print available computing devices"""
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"GPUs: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
    
    print(f"CPUs: {len(cpus)}")
    for cpu in cpus:
        print(f"  - {cpu}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    configure_gpu()
    get_device_info()