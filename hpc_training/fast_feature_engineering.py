"""
Feature engineering combining transactions + customer features
Creates time-series dataset with cash flow growth as target
FIXED: Clips extreme outliers that break KNN
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import *


def prepare_timeseries_features(transactions, customers):
    """
    Create time-series features from transactions + static customer features
    
    Input:
    - transactions: Transaction data with BookingDatetime, Amount_EUR, D_C, cust_id
    - customers: Customer features with cluster assignments
    
    Output:
    - Time-series dataframe with cash flow growth + customer features
    """
    
    print("="*80)
    print("PREPARING TIME-SERIES FEATURES")
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
            
            # Step 3: Aggregate to weekly level (faster than daily)
            weekly_cf = cust_txns.groupby(pd.Grouper(key='datetime', freq='W')).agg({
                'cash_flow': 'sum',
                'Amount_EUR': ['count', 'mean', 'std']
            })
            
            weekly_cf.columns = ['cash_flow', 'txn_count', 'avg_txn', 'std_txn']
            weekly_cf = weekly_cf[weekly_cf['txn_count'] > 0]  # Remove empty weeks
            
            if len(weekly_cf) < MIN_OBSERVATIONS:
                failed += 1
                continue
            
            # Step 4: Calculate TARGET - cash flow growth rate
            weekly_cf['cash_flow_growth'] = weekly_cf['cash_flow'].pct_change()
            
            # CRITICAL FIX: Clip extreme outliers (cap at ±200% growth)
            # This prevents values like +200,000% from breaking models
            weekly_cf['cash_flow_growth'] = weekly_cf['cash_flow_growth'].clip(-2.0, 2.0)
            
            # Remove infinite values
            weekly_cf['cash_flow_growth'] = weekly_cf['cash_flow_growth'].replace([np.inf, -np.inf], np.nan)
            
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
            
            # Key features from paper + clustering
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
            
            # Keep only if enough observations after removing NaN
            if len(weekly_cf) >= MIN_OBSERVATIONS:
                all_features.append(weekly_cf.reset_index())
            else:
                failed += 1
        
        except Exception as e:
            if i < 5:  # Print first few errors for debugging
                print(f"  Error for customer {cust_id}: {e}")
            failed += 1
            continue
    
    print(f"\nSuccessfully processed: {len(all_features)} customers")
    print(f"Failed: {failed} customers")
    
    if len(all_features) == 0:
        raise ValueError("No customers were successfully processed!")
    
    # Combine all customers
    result = pd.concat(all_features, ignore_index=True)
    
    # Check for remaining outliers
    growth_min = result['cash_flow_growth'].min()
    growth_max = result['cash_flow_growth'].max()
    print(f"\nCash flow growth range: [{growth_min:.4f}, {growth_max:.4f}]")
    
    if abs(growth_min) > 3 or abs(growth_max) > 3:
        print("WARNING: Extreme values detected, applying additional clipping...")
        result['cash_flow_growth'] = result['cash_flow_growth'].clip(-1.5, 1.5)
    
    # Standardize numeric features (excluding target and IDs)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    exclude_cols = ['cust_id', 'cash_flow_growth', 'cash_flow']
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"\nStandardizing {len(scale_cols)} features...")
    scaler = StandardScaler()
    result[scale_cols] = scaler.fit_transform(result[scale_cols].fillna(0))
    
    print(f"\nFinal dataset:")
    print(f"  Shape: {result.shape}")
    print(f"  Customers: {result['cust_id'].nunique()}")
    print(f"  Clusters: {result['cluster'].nunique()}")
    print(f"  Features: {len(scale_cols)}")
    
    print(f"\nCluster distribution:")
    print(result.groupby('cluster')['cust_id'].nunique())
    
    print(f"\nSample data:")
    print(result.head())
    
    return result
