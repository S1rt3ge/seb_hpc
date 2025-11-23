#!/usr/bin/env python
"""
Test training a single cluster with explicit GPU assignment
"""

import os
import sys

# Set GPU BEFORE importing torch
GPU_ID = "0"  # Change this to test different GPUs: 0, 1, 2, or 3
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

print("="*80)
print(f"SINGLE CLUSTER TRAINING TEST (GPU {GPU_ID})")
print("="*80)

import pandas as pd
import torch

print(f"\nPython: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count (visible): {torch.cuda.device_count()}")
    print(f"GPU 0 name: {torch.cuda.get_device_name(0)}")
else:
    print("ERROR: CUDA not available!")
    sys.exit(1)

# Load features
print("\nLoading features...")
if not os.path.exists('customer_timeseries_features.csv'):
    print("ERROR: customer_timeseries_features.csv not found!")
    sys.exit(1)

features = pd.read_csv('customer_timeseries_features.csv')
print(f"✓ Features loaded: {features.shape}")

# Get first cluster
clusters = sorted(features['cluster'].unique())
test_cluster = clusters[0]  # Train first cluster only

print(f"\nTesting with cluster: {test_cluster}")

# Import and run training
from fast_training_ddp import train_cluster

print("\n" + "="*80)
print(f"TRAINING {test_cluster}")
print("="*80)

try:
    result = train_cluster(test_cluster, features, gpu_id=0)
    
    print("\n" + "="*80)
    print("✓ TRAINING SUCCESSFUL!")
    print("="*80)
    print(f"Cluster: {result['cluster']}")
    print(f"Best model: {result['best_model']}")
    print(f"Test RMSE: {result['results'][result['best_model']]['test_rmse']:.4f}")
    print(f"Training time: {result['training_time']/60:.1f} minutes")
    
except Exception as e:
    print("\n" + "="*80)
    print("✗ TRAINING FAILED")
    print("="*80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
