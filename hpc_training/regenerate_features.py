#!/usr/bin/env python3
"""
Regenerate Features - Use FIXED Feature Engineering
This will create a NEW customer_timeseries_features.csv with proper clipping
"""

import pandas as pd
import os
import sys
from datetime import datetime

print("="*80)
print("REGENERATE FEATURES WITH FIXED CLIPPING")
print("="*80)
print(f"Start: {datetime.now()}")
print("="*80)

# Import the FIXED version
print("\n1. Loading fixed feature engineering...")
from fast_feature_engineering import prepare_timeseries_features
from config import MIN_OBSERVATIONS, MAX_CUSTOMERS_PER_CLUSTER

# Load raw data
print("\n2. Loading raw data...")
if not os.path.exists('synthetic_sme_transactions_processed.csv'):
    print("ERROR: synthetic_sme_transactions_processed.csv not found!")
    sys.exit(1)

if not os.path.exists('customer_clusters_FINAL.csv'):
    print("ERROR: customer_clusters_FINAL.csv not found!")
    sys.exit(1)

transactions = pd.read_csv('synthetic_sme_transactions_processed.csv', low_memory=False)
customers = pd.read_csv('customer_clusters_FINAL.csv')

print(f"   Transactions: {len(transactions):,} rows")
print(f"   Customers: {len(customers):,} rows")

# Backup old file
print("\n3. Backing up old features file...")
if os.path.exists('customer_timeseries_features.csv'):
    backup_name = f'customer_timeseries_features_BACKUP_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    os.rename('customer_timeseries_features.csv', backup_name)
    print(f"   Old file backed up to: {backup_name}")
else:
    print("   No existing file to backup")

# Generate NEW features
print("\n4. Generating features with FIXED clipping...")
try:
    features = prepare_timeseries_features(transactions, customers)
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save
print("\n5. Saving new features...")
features.to_csv('customer_timeseries_features.csv', index=False)
print("   [OK] Saved to: customer_timeseries_features.csv")

# Verification
print("\n6. Verification:")
print(f"   Shape: {features.shape}")
print(f"   Customers: {features['cust_id'].nunique()}")
print(f"   Clusters: {features['cluster'].nunique()}")

print("\n7. Target variable check:")
target_stats = features['cash_flow_growth'].describe()
print(target_stats)

print(f"\n8. Target variable check (now standardized):")
target_mean = features['cash_flow_growth'].mean()
target_std = features['cash_flow_growth'].std()
print(f"   Mean: {target_mean:.6f} (should be ~0)")
print(f"   Std:  {target_std:.6f} (should be ~1)")

if abs(target_mean) < 0.01 and abs(target_std - 1.0) < 0.1:
    print("   [OK] Target properly standardized")
else:
    print("   [WARNING] Target standardization off")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"End: {datetime.now()}")
print("\nNext step: Re-run training with new features file")
print("  python3 train_final.py")
print("="*80)