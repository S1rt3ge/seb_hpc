"""
Diagnostic script to identify and verify the RMSE issue
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

print("="*80)
print("DIAGNOSTIC: High RMSE Issue Analysis")
print("="*80)

# Check if features file exists
try:
    features = pd.read_csv('customer_timeseries_features.csv')
    print(f"\n✓ Features loaded: {features.shape}")
except FileNotFoundError:
    print("\n✗ customer_timeseries_features.csv not found!")
    print("   Please run feature engineering first")
    exit(1)

# 1. Check target variable distribution
print("\n" + "="*80)
print("1. TARGET VARIABLE ANALYSIS")
print("="*80)

target_col = 'cash_flow_growth'
if target_col not in features.columns:
    print(f"✗ Target column '{target_col}' not found!")
    exit(1)

target = features[target_col].dropna()
print(f"\nTarget: {target_col}")
print(f"  Count: {len(target)}")
print(f"  Mean: {target.mean():.6f}")
print(f"  Std: {target.std():.6f}")
print(f"  Min: {target.min():.6f}")
print(f"  Max: {target.max():.6f}")
print(f"  Median: {target.median():.6f}")

# Check for extreme values
extreme_threshold = 10
extreme_values = target[np.abs(target) > extreme_threshold]
if len(extreme_values) > 0:
    print(f"\n⚠️  WARNING: {len(extreme_values)} extreme values (|value| > {extreme_threshold})")
    print(f"    This WILL cause trillion-scale RMSE!")
    print(f"    Max absolute value: {np.abs(target).max():.2f}")
else:
    print(f"\n✓ No extreme values detected")

# Check for inf/nan
n_inf = np.isinf(target).sum()
n_nan = target.isna().sum()
print(f"\nData quality:")
print(f"  Inf values: {n_inf}")
print(f"  NaN values: {n_nan}")

# 2. Analyze by cluster
print("\n" + "="*80)
print("2. CLUSTER-WISE ANALYSIS")
print("="*80)

clusters = sorted(features['cluster'].unique())
print(f"\nClusters: {len(clusters)}")

problem_clusters = []

for cluster in clusters:
    cluster_data = features[features['cluster'] == cluster]
    cluster_target = cluster_data[target_col].dropna()
    
    mean_val = cluster_target.mean()
    std_val = cluster_target.std()
    min_val = cluster_target.min()
    max_val = cluster_target.max()
    abs_max = np.abs(cluster_target).max()
    
    status = "✓" if abs_max < 3 else "⚠️"
    
    print(f"\n{status} {cluster}:")
    print(f"    N: {len(cluster_target)}, "
          f"Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    print(f"    Range: [{min_val:.4f}, {max_val:.4f}], "
          f"Max|val|: {abs_max:.4f}")
    
    if abs_max > 3:
        problem_clusters.append(cluster)
        print(f"    ⚠️  PROBLEM: Extreme values will cause high RMSE!")

# 3. Test scaling impact
print("\n" + "="*80)
print("3. SCALING IMPACT TEST")
print("="*80)

# Simulate what happens without scaling
test_cluster = clusters[0]
cluster_data = features[features['cluster'] == test_cluster][target_col].dropna().values

print(f"\nTest cluster: {test_cluster}")
print(f"Original scale:")
print(f"  Mean: {cluster_data.mean():.6f}")
print(f"  Std: {cluster_data.std():.6f}")
print(f"  Range: [{cluster_data.min():.6f}, {cluster_data.max():.6f}]")

# Apply scaling
scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data.reshape(-1, 1)).flatten()

print(f"\nAfter StandardScaler:")
print(f"  Mean: {cluster_scaled.mean():.6f}")
print(f"  Std: {cluster_scaled.std():.6f}")
print(f"  Range: [{cluster_scaled.min():.6f}, {cluster_scaled.max():.6f}]")

# Simulate RMSE calculation
# If model predicts 0 for everything (baseline)
baseline_rmse_unscaled = np.sqrt(np.mean((cluster_data - 0)**2))
baseline_rmse_scaled = np.sqrt(np.mean((cluster_scaled - 0)**2))

print(f"\nBaseline RMSE (all predictions = 0):")
print(f"  Unscaled: {baseline_rmse_unscaled:.4f}")
print(f"  Scaled: {baseline_rmse_scaled:.4f}")
print(f"  Ratio: {baseline_rmse_unscaled / baseline_rmse_scaled:.2f}x")

# 4. Summary and recommendations
print("\n" + "="*80)
print("4. DIAGNOSIS SUMMARY")
print("="*80)

print("\nIssues found:")
if len(problem_clusters) > 0:
    print(f"  ⚠️  {len(problem_clusters)} clusters with extreme values:")
    for c in problem_clusters:
        print(f"      - {c}")
    print("\n  ROOT CAUSE: Feature engineering didn't clip values properly")
    print("  IMPACT: Models predict scaled values on unscaled targets")
    print("  RESULT: RMSE in trillions instead of < 1.0")
else:
    print("  ✓ No extreme values detected")

print("\n" + "="*80)
print("SOLUTIONS IMPLEMENTED IN FIXED VERSION:")
print("="*80)
print("""
1. ✓ Target scaling for deep learning models
   - Fit StandardScaler on training target
   - Transform train/val/test targets
   - Train models on scaled data
   - Convert RMSE back to original scale

2. ✓ Additional safety checks
   - Clip extreme values to [-2, 2] range
   - Remove any inf/nan values
   - Print target statistics per cluster

3. ✓ JSON serialization fixes
   - Convert numpy float32 → Python float
   - Convert numpy int64 → Python int
   - Ensure all values are JSON-serializable

4. ✓ KNN training fix
   - No longer slice data by SEQUENCE_LENGTH
   - Use all available training data
""")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("""
1. Use fast_training_ddp_FIXED.py instead of fast_training_ddp.py
2. Use train_final_FIXED.py instead of train_final.py
3. Re-run training and verify RMSE < 1.0 for all models
""")
print("="*80)
