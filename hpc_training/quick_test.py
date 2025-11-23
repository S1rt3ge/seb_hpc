"""
Quick test to verify everything works
Run this before submitting full job
"""

import torch
import pandas as pd
from config import *

print("="*80)
print("QUICK TEST")
print("="*80)

# Test 1: Check data files
print("\n1. Checking data files...")
try:
    transactions = pd.read_csv('synthetic_sme_transactions_processed.csv', nrows=50000)
    customers = pd.read_csv('customer_clusters_FINAL.csv')

    # Find which customers actually exist in loaded transactions
    cust_col = 'Cust_id' if 'Cust_id' in transactions.columns else 'cust_id'
    available_custs = transactions[cust_col].unique()
    customers = customers[customers['cust_id'].isin(available_custs)]

    print(f"   OK Data files found")
    print(f"   Transactions: {len(transactions)}, Customers matched: {len(customers)}")
except Exception as e:
    print(f"   X Data files error: {e}")
    exit(1)

# Test 2: Check CUDA
print("\n2. Checking CUDA...")
if torch.cuda.is_available():
    print(f"   OK CUDA available")
    print(f"   GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("   WARNING CUDA not available (will use CPU)")

# Test 3: Test models
print("\n3. Testing models...")
from fast_dl_models import create_model

try:
    for model_type in ['lstm', 'gru', 'transformer']:
        model = create_model(model_type, input_size=10, hidden_size=16)
        x = torch.randn(2, 5, 10)  # batch=2, seq=5, features=10
        y = model(x)
        assert y.shape == (2, 1)
    print("   OK All models working")
except Exception as e:
    print(f"   X Model error: {e}")
    exit(1)

# Test 4: Test feature engineering
print("\n4. Testing feature engineering...")
from fast_feature_engineering import prepare_timeseries_features

try:
    # Test on 10 customers
    customers_subset = customers.groupby('cluster').head(2)
    features = prepare_timeseries_features(transactions, customers_subset)
    print(f"   OK Features generated: {features.shape}")
except Exception as e:
    print(f"   X Feature error: {e}")
    exit(1)

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("Ready to submit job: qsub train.pbs")
print("="*80)
