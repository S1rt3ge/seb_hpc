#!/usr/bin/env python
"""
Test GPU assignment with CUDA_VISIBLE_DEVICES
"""

import os
import sys
import torch

print("="*80)
print("GPU ASSIGNMENT TEST")
print("="*80)

# Check CUDA_VISIBLE_DEVICES
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print(f"\nCUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    print("\nCUDA_VISIBLE_DEVICES not set")

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

# Check device count
n_devices = torch.cuda.device_count()
print(f"Visible GPU count: {n_devices}")

# Test each visible device
print("\nTesting visible devices:")
for i in range(n_devices):
    try:
        torch.cuda.set_device(i)
        name = torch.cuda.get_device_name(i)
        
        # Allocate tensor
        x = torch.randn(1000, 1000).cuda()
        y = x @ x.T
        
        print(f"  Device {i}: ✓ {name}")
        
        del x, y
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  Device {i}: ✗ {e}")

# Test model creation and move to GPU
print("\nTesting model move to GPU:")
try:
    from fast_dl_models import create_model
    
    model = create_model('lstm', input_size=10, hidden_size=32)
    print(f"  ✓ Model created")
    
    # Move to GPU 0 (the only visible one if CUDA_VISIBLE_DEVICES is set)
    device = 'cuda:0'
    model = model.to(device)
    print(f"  ✓ Model moved to {device}")
    
    # Test forward pass
    x = torch.randn(2, 5, 10).to(device)
    y = model(x)
    print(f"  ✓ Forward pass successful: {y.shape}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✓ GPU ASSIGNMENT TEST COMPLETE")
print("="*80)
