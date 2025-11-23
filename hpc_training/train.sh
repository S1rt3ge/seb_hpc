#!/bin/sh
#PBS -N SME_CashFlow
#PBS -q batch
#PBS -l nodes=1:ppn=128:gpus=3
#PBS -j oe

cd $PBS_O_WORKDIR

echo "=========================================="
echo "SME Cash Flow Forecasting Training"
echo "=========================================="
echo "Job started: $(date)"
echo "Working dir: $(pwd)"
echo "Node: $(hostname)"
echo ""

# Load modules
module purge
module load AI/pytorch-1.13.1-gpu-conda

# Install additional requirements
pip install --user -q scikit-learn==1.2.2
pip install --user -q pandas==1.5.3
pip install --user -q statsmodels==0.14.0

# Check GPU availability
echo "Checking GPUs..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Run training
echo "Starting training..."
python fast_training.py

echo ""
echo "Job finished: $(date)"
echo "=========================================="
