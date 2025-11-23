#!/bin/bash

# Diagnose and fix Python environment issues

echo "=========================================="
echo "Python Environment Diagnostic"
echo "=========================================="
echo ""

# Check current environment
echo "1. Current Python:"
which python
python --version
echo ""

echo "2. Current Python3:"
which python3
python3 --version
echo ""

echo "3. Loaded modules:"
module list 2>&1
echo ""

# Unload everything
echo "4. Unloading all modules..."
module purge
echo ""

# Load correct PyTorch module
echo "5. Loading AI/pytorch-1.13.1-gpu-conda..."
module load AI/pytorch-1.13.1-gpu-conda
echo ""

# Verify
echo "6. After loading module:"
which python
python --version
echo ""

# Test PyTorch
echo "7. Testing PyTorch..."
python -c "import sys; print(f'Python: {sys.version}'); import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo ""

# Test imports
echo "8. Testing critical imports..."
python -c "
try:
    import pandas
    print('✓ pandas')
except:
    print('✗ pandas')

try:
    import numpy
    print('✓ numpy')
except:
    print('✗ numpy')

try:
    import sklearn
    print('✓ sklearn')
except:
    print('✗ sklearn')

try:
    import torch
    print('✓ torch')
except:
    print('✗ torch')
"
echo ""

echo "=========================================="
echo "If all tests pass, run:"
echo "  python train_parallel_fixed.py"
echo "=========================================="
