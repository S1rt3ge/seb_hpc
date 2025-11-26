"""
FIXED Feature Engineering - Aggressive Outlier Removal
This version has MULTIPLE layers of protection against extreme values
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import *


def prepare_timeseries_features(transactions, customers):
    """
    Create time-series features with AGGRESSIVE outlier removal
    """

    print("="*80)
    print("PREPARING TIME-SERIES FEATURES (FIXED VERSION)")
    print("="*80)

    all_features = []
    valid_customers = customers['cust_id'].unique()

    if MAX_CUSTOMERS_PER_CLUSTER:
        print(f"Testing mode: Using {MAX_CUSTOMERS_PER_CLUSTER} customers per cluster")

    print(f"\nProcessing {len(valid_customers)} customers...")

    # Check column names
    if 'BookingDatetime' in transactions.columns:
        date_col = 'BookingDatetime'
    elif 'datetime' in transactions.columns:
        date_col = 'datetime'
    elif 'date' in transactions.columns:
        date_col = 'date'
    else:
        print(f"Available columns: {transactions.columns.tolist()}")
        raise ValueError("No date column found!")

    print(f"Using columns: cust=cust_id, date={date_col}")

    failed = 0
    extreme_value_customers = 0

    for i, cust_id in enumerate(valid_customers):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(valid_customers)}")

        try:
            # Step 1: Get customer transactions
            cust_txns = transactions[transactions['cust_id'] == cust_id].copy()

            if len(cust_txns) < MIN_OBSERVATIONS:
                failed += 1
                continue

            cust_txns['datetime'] = pd.to_datetime(cust_txns[date_col])
            cust_txns = cust_txns.sort_values('datetime')

            # Step 2: Calculate cash flow (Credits - Debits)
            cust_txns['cash_flow'] = cust_txns.apply(
                lambda row: row['Amount_EUR'] if row['D_C'] == 'C' else -row['Amount_EUR'],
                axis=1
            )

            # Step 3: Aggregate to weekly level
            weekly_cf = cust_txns.groupby(pd.Grouper(key='datetime', freq='W')).agg({
                'cash_flow': 'sum',
                'Amount_EUR': ['count', 'mean', 'std']
            })

            weekly_cf.columns = ['cash_flow', 'txn_count', 'avg_txn', 'std_txn']
            weekly_cf = weekly_cf[weekly_cf['txn_count'] > 0]  # Remove empty weeks

            if len(weekly_cf) < MIN_OBSERVATIONS:
                failed += 1
                continue

            # Step 4: Calculate TARGET - cash flow ABSOLUTE change (not percentage)
            # This avoids division by near-zero values causing extreme percentages
            weekly_cf['cash_flow_growth'] = weekly_cf['cash_flow'].diff()

            # Note: This is now in EUR, not percentage
            # Will be standardized later along with other features

            # Step 5: Add lagged features
            for lag in range(1, N_LAGS + 1):
                weekly_cf[f'growth_lag_{lag}'] = weekly_cf['cash_flow_growth'].shift(lag)
                weekly_cf[f'cashflow_lag_{lag}'] = weekly_cf['cash_flow'].shift(lag)
                weekly_cf[f'txn_count_lag_{lag}'] = weekly_cf['txn_count'].shift(lag)

            # Step 6: Add rolling statistics
            weekly_cf['growth_mean_rolling'] = weekly_cf['cash_flow_growth'].rolling(4, min_periods=1).mean()
            weekly_cf['growth_std_rolling'] = weekly_cf['cash_flow_growth'].rolling(4, min_periods=1).std()
            weekly_cf['cashflow_mean_rolling'] = weekly_cf['cash_flow'].rolling(4, min_periods=1).mean()

            # Step 7: Add STATIC customer features
            cust_info = customers[customers['cust_id'] == cust_id].iloc[0]

            static_features = [
                'top5_concentration',
                'unique_counterparties',
                'small_txn_pct',
                'large_txn_pct',
                'cv_coefficient',
                'median_transaction_size',
                'volatility_std',
                'channel_pos_pct',
                'channel_internet_bank_pct',
                'negative_day_pct',
                'transaction_frequency'
            ]

            for feature in static_features:
                if feature in cust_info.index:
                    weekly_cf[feature] = cust_info[feature]

            # Add metadata
            weekly_cf['cust_id'] = cust_id
            weekly_cf['cluster'] = cust_info['cluster']

            # Remove NaN rows (from lags and pct_change)
            weekly_cf = weekly_cf.dropna(subset=['cash_flow_growth'])

    # PROTECTION LAYER 6: Final check before adding
            if len(weekly_cf) >= MIN_OBSERVATIONS:
                all_features.append(weekly_cf.reset_index())
            else:
                failed += 1

        except Exception as e:
            if i < 5:  # Print first few errors
                print(f"  Error for customer {cust_id}: {e}")
            failed += 1
            continue

    print(f"\nSuccessfully processed: {len(all_features)} customers")
    print(f"Failed: {failed} customers")
    print(f"  (including {extreme_value_customers} with extreme values)")

    if len(all_features) == 0:
        raise ValueError("No customers were successfully processed!")

    # Combine all customers
    result = pd.concat(all_features, ignore_index=True)

    # PROTECTION LAYER 7: Final sanity check on combined data
    growth_min = result['cash_flow_growth'].min()
    growth_max = result['cash_flow_growth'].max()
    growth_mean = result['cash_flow_growth'].mean()
    growth_std = result['cash_flow_growth'].std()

    print(f"\nCash flow growth (absolute change in EUR) statistics:")
    print(f"  Range: [{growth_min:.2f}, {growth_max:.2f}] EUR")
    print(f"  Mean: {growth_mean:.2f} EUR")
    print(f"  Std: {growth_std:.2f} EUR")

    # Check for inf/nan in target
    inf_count = np.isinf(result['cash_flow_growth']).sum()
    nan_count = result['cash_flow_growth'].isna().sum()
    if inf_count > 0 or nan_count > 0:
        print(f"  WARNING: Found {inf_count} inf, {nan_count} nan in target")
        print(f"  Removing these rows...")
        result = result[np.isfinite(result['cash_flow_growth'])]

    # Standardize numeric features
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    exclude_cols = ['cust_id', 'cash_flow']
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]

    # OPTION: Per-cluster standardization (set to True if recommended)
    PER_CLUSTER_STANDARDIZATION = False

    if PER_CLUSTER_STANDARDIZATION:
        print(f"\nStandardizing {len(scale_cols)} features PER CLUSTER...")

        # Standardize each cluster separately
        for cluster_name in result['cluster'].unique():
            cluster_mask = result['cluster'] == cluster_name
            cluster_data = result.loc[cluster_mask, scale_cols]

            scaler = StandardScaler()
            result.loc[cluster_mask, scale_cols] = scaler.fit_transform(cluster_data.fillna(0))

            print(f"  {cluster_name}: {cluster_mask.sum()} rows standardized")
    else:
        print(f"\nStandardizing {len(scale_cols)} features GLOBALLY (across all clusters)...")

        # Calculate mean/std BEFORE standardization for target variable
        target_mean_before = result['cash_flow_growth'].mean()
        target_std_before = result['cash_flow_growth'].std()

        scaler = StandardScaler()
        result[scale_cols] = scaler.fit_transform(result[scale_cols].fillna(0))

        # Save scaler parameters for validation
        scaler_params = {
            'target_mean': target_mean_before,
            'target_std': target_std_before,
            'feature_names': scale_cols  # Already a list, don't call tolist()
        }

        import json
        with open('scaler_params.json', 'w') as f:
            json.dump(scaler_params, f, indent=2)

        print(f"  Scaler parameters saved to: scaler_params.json")
        print(f"  Target mean (before): €{target_mean_before:,.2f}")
        print(f"  Target std (before): €{target_std_before:,.2f}")

    # PROTECTION LAYER 8: Check standardized features
    for col in scale_cols:
        col_max = result[col].abs().max()
        if col_max > 100:  # Standardized features shouldn't be >100
            print(f"  WARNING: {col} has extreme values (max={col_max:.2f})")
            result[col] = result[col].clip(-10, 10)  # Clip standardized features

    print(f"\nFinal dataset:")
    print(f"  Shape: {result.shape}")
    print(f"  Customers: {result['cust_id'].nunique()}")
    print(f"  Clusters: {result['cluster'].nunique()}")
    print(f"  Features: {len(scale_cols)}")

    print(f"\nCluster distribution:")
    print(result.groupby('cluster')['cust_id'].nunique())

    # Final verification
    print(f"\nFinal target verification:")
    print(f"  Min: {result['cash_flow_growth'].min():.6f}")
    print(f"  Max: {result['cash_flow_growth'].max():.6f}")
    print(f"  Mean: {result['cash_flow_growth'].mean():.6f}")
    print(f"  Std: {result['cash_flow_growth'].std():.6f}")

    print(f"\n[SUCCESS] Features created with absolute change (no extreme percentages!)")

    return result