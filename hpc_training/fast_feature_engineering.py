"""
Feature engineering combining transactions + customer features
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
    
    print(f"\nProcessing {len(valid_customers)} customers...")

    # Detect column names (handle different naming conventions)
    cust_col = 'Cust_id' if 'Cust_id' in transactions.columns else 'cust_id'
    date_col = 'BookingDatetime' if 'BookingDatetime' in transactions.columns else 'BookingDateTime'

    print(f"Using columns: cust={cust_col}, date={date_col}")

    for i, cust_id in enumerate(valid_customers):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(valid_customers)}")
        
        try:
            # Step 1: Get customer transactions
            cust_txns = transactions[transactions[cust_col] == cust_id].copy()

            if len(cust_txns) < MIN_OBSERVATIONS:
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
            
            # Step 4: Calculate TARGET - cash flow growth rate
            weekly_cf['cash_flow_growth'] = weekly_cf['cash_flow'].pct_change()
            
            # Step 5: Add lagged features (past 3 weeks)
            for lag in range(1, 4):
                weekly_cf[f'growth_lag_{lag}'] = weekly_cf['cash_flow_growth'].shift(lag)
                weekly_cf[f'cashflow_lag_{lag}'] = weekly_cf['cash_flow'].shift(lag)
            
            # Step 6: Add rolling statistics
            weekly_cf['growth_mean_rolling'] = weekly_cf['cash_flow_growth'].rolling(4, min_periods=1).mean()
            weekly_cf['growth_std_rolling'] = weekly_cf['cash_flow_growth'].rolling(4, min_periods=1).std()
            
            # Step 7: Add STATIC customer features from customer_clusters_FINAL.csv
            cust_info = customers[customers['cust_id'] == cust_id].iloc[0]
            
            # Key features from paper + your clustering
            static_features = [
                'top5_concentration',
                'unique_counterparties', 
                'small_txn_pct',
                'cv_coefficient',
                'median_transaction_size',
                'volatility_std',
                'channel_pos_pct',
                'channel_internet_bank_pct'
            ]
            
            for feature in static_features:
                if feature in cust_info.index:
                    weekly_cf[feature] = cust_info[feature]
            
            # Add metadata
            weekly_cf['cust_id'] = cust_id
            weekly_cf['cluster'] = cust_info['cluster']
            
            # Remove NaN rows
            weekly_cf = weekly_cf.dropna(subset=['cash_flow_growth'])
            
            # Keep only if enough observations
            if len(weekly_cf) >= MIN_OBSERVATIONS:
                all_features.append(weekly_cf.reset_index())
        
        except Exception as e:
            if i < 5:  # Print first few errors for debugging
                print(f"  Error for customer {cust_id}: {e}")
            continue
    
    print(f"\nSuccessfully processed: {len(all_features)} customers")
    
    if len(all_features) == 0:
        raise ValueError("No customers were successfully processed!")
    
    # Combine all customers
    result = pd.concat(all_features, ignore_index=True)
    
    # Standardize numeric features
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    exclude_cols = ['cust_id', 'cash_flow_growth']
    scale_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    scaler = StandardScaler()
    result[scale_cols] = scaler.fit_transform(result[scale_cols].fillna(0))
    
    print(f"\nFinal dataset:")
    print(f"  Shape: {result.shape}")
    print(f"  Customers: {result['cust_id'].nunique()}")
    print(f"  Clusters: {result['cluster'].nunique()}")
    print(f"  Features: {len(scale_cols)}")
    
    print(f"\nSample data:")
    print(result.head())
    
    return result


if __name__ == "__main__":
    print("Testing feature engineering...")
    
    # Load data
    print("\nLoading data...")
    transactions = pd.read_csv('synthetic_sme_transactions_processed.csv', low_memory=False)
    customers = pd.read_csv('customer_clusters_FINAL.csv')
    
    print(f"Transactions: {len(transactions):,} rows")
    print(f"Customers: {len(customers):,} rows")
    
    # Test on subset
    print("\nTesting on 20 customers...")
    test_customers = customers.groupby('cluster').head(3)
    
    features = prepare_timeseries_features(transactions, test_customers)
    
    print("\nFeature columns:")
    print(features.columns.tolist())
    
    # Save test
    features.to_csv('test_timeseries_features.csv', index=False)
    print("\nSaved: test_timeseries_features.csv")