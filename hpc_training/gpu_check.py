"""
GPU Verification Script
Checks if all GPUs are accessible and working properly
"""

import torch
import sys
import subprocess


def check_nvidia_smi():
    """Check if nvidia-smi is available"""
    print("="*80)
    print("1. CHECKING NVIDIA-SMI")
    print("="*80)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ nvidia-smi is available")
            print("\nGPU Summary:")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("✗ nvidia-smi not found (NVIDIA drivers not installed?)")
        return False
    except Exception as e:
        print(f"✗ Error running nvidia-smi: {e}")
        return False


def check_cuda_available():
    """Check if CUDA is available in PyTorch"""
    print("\n" + "="*80)
    print("2. CHECKING PYTORCH CUDA")
    print("="*80)
    
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("✓ CUDA is available in PyTorch")
        return True
    else:
        print("✗ CUDA is NOT available in PyTorch")
        print("\nPossible reasons:")
        print("  - PyTorch CPU-only version installed")
        print("  - CUDA drivers not properly installed")
        print("  - PyTorch CUDA version mismatch with drivers")
        return False


def check_gpu_count():
    """Check number of GPUs"""
    print("\n" + "="*80)
    print("3. CHECKING GPU COUNT")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("✗ Cannot check GPU count (CUDA not available)")
        return 0
    
    count = torch.cuda.device_count()
    print(f"✓ Found {count} GPU(s)")
    
    return count


def check_individual_gpus(n_gpus):
    """Check each GPU individually"""
    print("\n" + "="*80)
    print("4. CHECKING INDIVIDUAL GPUS")
    print("="*80)
    
    if n_gpus == 0:
        print("✗ No GPUs to check")
        return False
    
    all_ok = True
    
    for i in range(n_gpus):
        print(f"\n--- GPU {i} ---")
        
        try:
            # Get GPU properties
            props = torch.cuda.get_device_properties(i)
            name = torch.cuda.get_device_name(i)
            
            print(f"  Name: {name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Multi-Processors: {props.multi_processor_count}")
            
            # Try to allocate memory
            torch.cuda.set_device(i)
            x = torch.zeros(1000, 1000).cuda()
            y = x + 1
            del x, y
            torch.cuda.empty_cache()
            
            print(f"  ✓ GPU {i} is accessible and working")
            
        except Exception as e:
            print(f"  ✗ GPU {i} error: {e}")
            all_ok = False
    
    return all_ok


def check_gpu_memory():
    """Check available memory on each GPU"""
    print("\n" + "="*80)
    print("5. CHECKING GPU MEMORY")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("✗ Cannot check GPU memory (CUDA not available)")
        return False
    
    n_gpus = torch.cuda.device_count()
    all_ok = True
    
    for i in range(n_gpus):
        try:
            torch.cuda.set_device(i)
            
            total = torch.cuda.get_device_properties(i).total_memory
            reserved = torch.cuda.memory_reserved(i)
            allocated = torch.cuda.memory_allocated(i)
            free = total - reserved
            
            print(f"\nGPU {i}:")
            print(f"  Total:     {total / 1e9:.2f} GB")
            print(f"  Reserved:  {reserved / 1e9:.2f} GB")
            print(f"  Allocated: {allocated / 1e9:.2f} GB")
            print(f"  Free:      {free / 1e9:.2f} GB")
            
            # Check if enough free memory (at least 2GB)
            if free < 2e9:
                print(f"  ⚠ Warning: Less than 2GB free memory")
                if allocated > 0:
                    print(f"  → {allocated / 1e9:.2f} GB currently in use")
            else:
                print(f"  ✓ Sufficient free memory")
                
        except Exception as e:
            print(f"  ✗ GPU {i} memory check failed: {e}")
            all_ok = False
    
    return all_ok


def test_parallel_access(n_gpus):
    """Test if we can access multiple GPUs simultaneously"""
    print("\n" + "="*80)
    print("6. TESTING PARALLEL GPU ACCESS")
    print("="*80)
    
    if n_gpus < 2:
        print("⚠ Only 1 GPU, skipping parallel test")
        return True
    
    try:
        print(f"Testing simultaneous access to {n_gpus} GPUs...")
        
        tensors = []
        for i in range(n_gpus):
            torch.cuda.set_device(i)
            x = torch.randn(1000, 1000).cuda()
            tensors.append(x)
            print(f"  ✓ Created tensor on GPU {i}")
        
        # Perform operations on all GPUs
        print("\nPerforming operations on all GPUs simultaneously...")
        results = []
        for i, tensor in enumerate(tensors):
            torch.cuda.set_device(i)
            result = tensor @ tensor.T
            results.append(result)
            print(f"  ✓ Computation on GPU {i} successful")
        
        # Cleanup
        for i, tensor in enumerate(tensors):
            torch.cuda.set_device(i)
            del tensor
        for result in results:
            del result
        
        for i in range(n_gpus):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        
        print("\n✓ All GPUs can be accessed simultaneously")
        return True
        
    except Exception as e:
        print(f"\n✗ Parallel GPU access failed: {e}")
        return False


def test_cuda_operations():
    """Test basic CUDA operations"""
    print("\n" + "="*80)
    print("7. TESTING CUDA OPERATIONS")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("✗ Cannot test CUDA operations (CUDA not available)")
        return False
    
    try:
        device = torch.device('cuda:0')
        
        # Test tensor creation
        print("Testing tensor creation...")
        x = torch.randn(100, 100, device=device)
        print("  ✓ Tensor creation successful")
        
        # Test matrix multiplication
        print("Testing matrix multiplication...")
        y = torch.matmul(x, x.T)
        print("  ✓ Matrix multiplication successful")
        
        # Test data transfer
        print("Testing CPU <-> GPU transfer...")
        x_cpu = x.cpu()
        x_gpu = x_cpu.cuda()
        print("  ✓ Data transfer successful")
        
        # Test gradient computation
        print("Testing gradient computation...")
        a = torch.randn(10, 10, device=device, requires_grad=True)
        b = a ** 2
        c = b.sum()
        c.backward()
        print("  ✓ Gradient computation successful")
        
        # Cleanup
        del x, y, x_cpu, x_gpu, a, b, c
        torch.cuda.empty_cache()
        
        print("\n✓ All CUDA operations working correctly")
        return True
        
    except Exception as e:
        print(f"\n✗ CUDA operations failed: {e}")
        return False


def main():
    """Run all GPU checks"""
    
    print("\n" + "="*80)
    print("GPU VERIFICATION SCRIPT")
    print("="*80)
    print()
    
    results = {}
    
    # Run all checks
    results['nvidia_smi'] = check_nvidia_smi()
    results['cuda_available'] = check_cuda_available()
    
    n_gpus = check_gpu_count()
    results['gpu_count'] = n_gpus > 0
    
    if n_gpus > 0:
        results['individual_gpus'] = check_individual_gpus(n_gpus)
        results['gpu_memory'] = check_gpu_memory()
        results['parallel_access'] = test_parallel_access(n_gpus)
        results['cuda_operations'] = test_cuda_operations()
    else:
        results['individual_gpus'] = False
        results['gpu_memory'] = False
        results['parallel_access'] = False
        results['cuda_operations'] = False
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    print(f"\n{'Check':<30} {'Status':<10}")
    print("-" * 40)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check:<30} {status:<10}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - GPUs ready for training!")
        print("="*80)
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Fix issues before training")
        print("="*80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
