import os
import sys
import tensorflow as tf

def check_gpu():
    print("="*50)
    print("TENSORFLOW GPU VERIFICATION")
    print("="*50)
    
    # 1. Check if built with CUDA
    cuda_built = tf.test.is_built_with_cuda()
    print(f"Built with CUDA: {cuda_built}")
    
    # 2. Check for GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Physical GPUs: {gpus}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  - Device {i}: {gpu}")
    else:
        print("\n[!] WARNING: No GPU detected by TensorFlow.")
        
        # 3. Check for specific DLLs (Windows only)
        if os.name == 'nt':
            print("\nScanning for missing CUDA/cuDNN DLLs on Windows...")
            important_dlls = [
                "cudart64_110.dll", "cublas64_11.dll", "cublasLt64_11.dll",
                "cudnn64_8.dll", "cufft64_10.dll", "curand64_10.dll",
                "cusolver64_11.dll", "cusparse64_11.dll"
            ]
            
            path_dirs = os.environ.get('PATH', '').split(os.pathsep)
            
            for dll in important_dlls:
                found = False
                for d in path_dirs:
                    if os.path.exists(os.path.join(d, dll)):
                        found = True
                        break
                status = "✓ FOUND" if found else "✗ MISSING"
                print(f"  {dll:20} : {status}")
            
            print("\nIf DLLs are missing, ensure CUDA Toolkit 11.x and cuDNN 8.x are in your System PATH.")

    print("="*50)

if __name__ == "__main__":
    check_gpu()
