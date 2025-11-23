#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import pandas as pd
import torch
import json
import sys

print(f"Cluster: Retail_Steady, GPU: 2")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

from fast_training_ddp import train_cluster

features = pd.read_csv('customer_timeseries_features.csv')

print(f"Starting training for Retail_Steady...")
result = train_cluster('Retail_Steady', features, gpu_id=0)

# Save result
result_data = {
    'cluster': result['cluster'],
    'best_model': result['best_model'],
    'training_time': result['training_time'],
    'n_customers': result['n_customers'],
    'n_observations': result['n_observations'],
    'knn_rmse': result['results']['KNN']['test_rmse'],
    'lstm_rmse': result['results']['LSTM']['test_rmse'],
    'gru_rmse': result['results']['GRU']['test_rmse'],
    'transformer_rmse': result['results']['Transformer']['test_rmse']
}

with open('result_Retail_Steady.json', 'w') as f:
    json.dump(result_data, f)

print(f"✓ Retail_Steady complete! Time: {result['training_time']/60:.1f} min")
