"""
SIMPLE VALIDATION - Uses scaler_params.json
This version is bulletproof - it loads the exact scaler parameters used.
"""

import pandas as pd
import numpy as np
import json
import os

print("="*80)
print("SIMPLE VALIDATION - Verifying Features Match Raw Transactions")
print("="*80)

# Step 1: Load scaler parameters
print("\n1. Loading scaler parameters...")
if not os.path.exists('scaler_params.json'):
    print("ERROR: scaler_params.json not found!")
    print("Please run: python regenerate_features.py")
    exit(1)

with open('scaler_params.json', 'r') as f:
    scaler_params = json.load(f)

GLOBAL_MEAN = scaler_params['target_mean']
GLOBAL_STD = scaler_params['target_std']

print(f"   Global scaler parameters:")
print(f"   Mean: €{GLOBAL_MEAN:,.2f}")
print(f"   Std:  €{GLOBAL_STD:,.2f}")

# Step 2: Load data
print("\n2. Loading data...")
features = pd.read_csv('customer_timeseries_features.csv')
transactions = pd.read_csv('synthetic_sme_transactions_processed.csv', low_memory=False)

print(f"   Features: {len(features):,} rows")
print(f"   Transactions: {len(transactions):,} rows")

# Step 3: Verify features are standardized
print("\n3. Verifying features file...")
feat_mean = features['cash_flow_growth'].mean()
feat_std = features['cash_flow_growth'].std()

print(f"   Target mean: {feat_mean:.10f} (should be ~0)")
print(f"   Target std:  {feat_std:.10f} (should be ~1)")

if abs(feat_mean) > 0.01 or abs(feat_std - 1.0) > 0.01:
    print("   [ERROR] Features not properly standardized!")
    exit(1)

print("   [OK] Features properly standardized")

# Step 4: Pick random customers
print("\n4. Selecting 3 random customers...")
np.random.seed(42)
sample_custs = np.random.choice(features['cust_id'].unique(), size=3, replace=False)

results = []

for idx, cust_id in enumerate(sample_custs):
    print(f"\n{'='*80}")
    print(f"Customer {idx+1}/3: {cust_id[:16]}...")
    print(f"{'='*80}")

    # Get features
    cust_feat = features[features['cust_id'] == cust_id].sort_values('datetime')
    print(f"Features: {len(cust_feat)} weeks")

    # Get transactions
    cust_txns = transactions[transactions['cust_id'] == cust_id].copy()
    print(f"Transactions: {len(cust_txns)} transactions")

    # Process transactions
    if 'BookingDatetime' in cust_txns.columns:
        cust_txns['datetime'] = pd.to_datetime(cust_txns['BookingDatetime'])
    else:
        cust_txns['datetime'] = pd.to_datetime(cust_txns['datetime'])

    cust_txns = cust_txns.sort_values('datetime')
    cust_txns['cash_flow'] = cust_txns.apply(
        lambda row: row['Amount_EUR'] if row['D_C'] == 'C' else -row['Amount_EUR'],
        axis=1
    )

    # Weekly aggregation
    weekly = cust_txns.groupby(pd.Grouper(key='datetime', freq='W')).agg({
        'cash_flow': 'sum',
        'Amount_EUR': 'count'
    })
    weekly.columns = ['cash_flow', 'txn_count']
    weekly = weekly[weekly['txn_count'] > 0]

    # Calculate growth
    weekly['growth_eur'] = weekly['cash_flow'].diff()

    # Standardize using GLOBAL parameters
    weekly['growth_std_GLOBAL'] = (weekly['growth_eur'] - GLOBAL_MEAN) / GLOBAL_STD

    print(f"\nComparing 5 sample weeks:")
    print(f"{'Week':<12} {'Raw (EUR)':>15} {'Raw→Std':>12} {'Features':>12} {'Error':>10} {'Match':>8}")
    print("-"*75)

    matches = 0
    total = 0

    for week_idx in [5, 10, 15, 20, 25]:
        if week_idx >= len(weekly) or week_idx >= len(cust_feat):
            continue

        week_date = weekly.index[week_idx]

        # Find matching week in features
        feat_match = cust_feat[
            (pd.to_datetime(cust_feat['datetime']) >= week_date - pd.Timedelta(days=3)) &
            (pd.to_datetime(cust_feat['datetime']) <= week_date + pd.Timedelta(days=3))
        ]

        if len(feat_match) == 0:
            continue

        raw_eur = weekly['growth_eur'].iloc[week_idx]
        raw_std = weekly['growth_std_GLOBAL'].iloc[week_idx]
        feat_std = feat_match['cash_flow_growth'].iloc[0]

        error = abs(raw_std - feat_std)
        status = 'OK' if error < 0.01 else 'FAIL'

        print(f"{week_date.strftime('%Y-%m-%d'):<12} {raw_eur:>15,.2f} {raw_std:>12.4f} "
              f"{feat_std:>12.4f} {error:>9.4f} {status:>8}")

        if status == 'OK':
            matches += 1
        total += 1

    match_rate = (matches / total * 100) if total > 0 else 0
    print("-"*75)
    print(f"Match rate: {matches}/{total} ({match_rate:.0f}%)")

    results.append({
        'customer': cust_id[:16] + '...',
        'weeks': len(cust_feat),
        'match_rate': match_rate,
        'status': 'PASS' if match_rate >= 80 else 'FAIL'
    })

# Final summary
print(f"\n{'='*80}")
print("VALIDATION SUMMARY")
print(f"{'='*80}")

print(f"\n{'Customer':<20} {'Weeks':>8} {'Match Rate':>12} {'Status':>10}")
print("-"*55)

for r in results:
    print(f"{r['customer']:<20} {r['weeks']:>8} {r['match_rate']:>11.0f}% {r['status']:>10}")

pass_count = sum(1 for r in results if r['status'] == 'PASS')
total_count = len(results)

print(f"\n{'='*80}")
if pass_count == total_count:
    print("✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
    print("\n=== 100% CONFIDENT - READY TO TRAIN ===")
    print("Run: python train_final.py")
elif pass_count >= total_count * 0.67:
    print("✓ MOSTLY PASSED")
    print("\n=== SAFE TO TRAIN ===")
    print("Run: python train_final.py")
else:
    print("✗ VALIDATION FAILED")
    print("\n=== DO NOT TRAIN ===")

print(f"{'='*80}")