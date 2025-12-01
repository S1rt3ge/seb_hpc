"""
Complete Feature Engineering from Raw Transactions ONLY
Works for NEW customers not in any training dataset
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta


class CompleteFeatureEngine:

    def __init__(self, scaler_params_path='scaler_params.json'):
        with open(scaler_params_path, 'r') as f:
            self.scaler_params = json.load(f)

        self.MIN_OBSERVATIONS = 20
        self.N_LAGS = 3

        print(f"✓ Loaded scaler params (target_std: €{self.scaler_params['target_std']:,.2f})")

    def calculate_static_features(self, transactions_df):
        """Calculate 11 static features"""

        transactions_df['BookingDatetime'] = pd.to_datetime(transactions_df['BookingDatetime'])
        transactions_df['date'] = transactions_df['BookingDatetime'].dt.date

        total_txns = len(transactions_df)
        channel_counts = transactions_df['Channel'].value_counts()

        channel_pos_pct = channel_counts.get('POS', 0) / total_txns if total_txns > 0 else 0
        channel_internet_bank_pct = channel_counts.get('Internet Bank', 0) / total_txns if total_txns > 0 else 0

        amounts = transactions_df['Amount_EUR'].abs()
        unique_counterparties = transactions_df['Counterparty_IBAN'].nunique()

        top5_sum = transactions_df.groupby('Counterparty_IBAN')['Amount_EUR'].sum().abs().nlargest(5).sum()
        total_sum = amounts.sum()
        top5_concentration = top5_sum / total_sum if total_sum > 0 else 0

        small_threshold = amounts.quantile(0.25)
        large_threshold = amounts.quantile(0.75)
        small_txn_pct = (amounts < small_threshold).sum() / total_txns if total_txns > 0 else 0
        large_txn_pct = (amounts > large_threshold).sum() / total_txns if total_txns > 0 else 0

        cv_coefficient = amounts.std() / amounts.mean() if amounts.mean() > 0 else 0
        median_transaction_size = amounts.median()
        volatility_std = amounts.std()

        daily_cf = transactions_df.groupby('date').apply(
            lambda x: (x[x['D_C'] == 'C']['Amount_EUR'].sum() -
                       x[x['D_C'] == 'D']['Amount_EUR'].sum())
        )
        negative_day_pct = (daily_cf < 0).sum() / len(daily_cf) if len(daily_cf) > 0 else 0

        date_range = (transactions_df['BookingDatetime'].max() -
                      transactions_df['BookingDatetime'].min()).days
        transaction_frequency = total_txns / date_range if date_range > 0 else 0

        return {
            'top5_concentration': top5_concentration,
            'unique_counterparties': unique_counterparties,
            'small_txn_pct': small_txn_pct,
            'large_txn_pct': large_txn_pct,
            'cv_coefficient': cv_coefficient,
            'median_transaction_size': median_transaction_size,
            'volatility_std': volatility_std,
            'channel_pos_pct': channel_pos_pct,
            'channel_internet_bank_pct': channel_internet_bank_pct,
            'negative_day_pct': negative_day_pct,
            'transaction_frequency': transaction_frequency
        }

    def classify_cluster(self, static_features):
        internet_pct = static_features['channel_internet_bank_pct']
        pos_pct = static_features['channel_pos_pct']
        freq = static_features['transaction_frequency']
        counterparties = static_features['unique_counterparties']
        concentration = static_features['top5_concentration']

        if internet_pct > 0.7 and freq > 2.0 and concentration > 0.6:
            return 'Subscription_SaaS'
        elif counterparties > 30 and concentration < 0.7:
            return 'B2B_Services_Combined'
        elif pos_pct > 0.2 and counterparties > 20:
            return 'Retail_Combined'
        elif internet_pct > 0.8 and pos_pct < 0.1:
            return 'Digital_B2C_Combined'
        else:
            return 'Mixed'

    def calculate_timeseries_features(self, transactions_df, static_features):
        transactions_df['datetime'] = pd.to_datetime(transactions_df['BookingDatetime'])
        transactions_df = transactions_df.sort_values('datetime')

        transactions_df['cash_flow'] = transactions_df.apply(
            lambda row: row['Amount_EUR'] if row['D_C'] == 'C' else -row['Amount_EUR'],
            axis=1
        )

        weekly_cf = transactions_df.groupby(pd.Grouper(key='datetime', freq='W')).agg({
            'cash_flow': 'sum',
            'Amount_EUR': ['count', 'mean', 'std']
        })

        weekly_cf.columns = ['cash_flow', 'txn_count', 'avg_txn', 'std_txn']
        weekly_cf = weekly_cf[weekly_cf['txn_count'] > 0]

        if len(weekly_cf) < self.MIN_OBSERVATIONS:
            raise ValueError(f"Not enough weekly observations")

        weekly_cf['cash_flow_growth'] = weekly_cf['cash_flow'].diff()

        for lag in range(1, self.N_LAGS + 1):
            weekly_cf[f'growth_lag_{lag}'] = weekly_cf['cash_flow_growth'].shift(lag)
            weekly_cf[f'cashflow_lag_{lag}'] = weekly_cf['cash_flow'].shift(lag)
            weekly_cf[f'txn_count_lag_{lag}'] = weekly_cf['txn_count'].shift(lag)

        weekly_cf['growth_mean_rolling'] = weekly_cf['cash_flow_growth'].rolling(4, min_periods=1).mean()
        weekly_cf['growth_std_rolling'] = weekly_cf['cash_flow_growth'].rolling(4, min_periods=1).std()
        weekly_cf['cashflow_mean_rolling'] = weekly_cf['cash_flow'].rolling(4, min_periods=1).mean()

        for key, value in static_features.items():
            weekly_cf[key] = value

        weekly_cf = weekly_cf.dropna(subset=['cash_flow_growth'])
        weekly_cf = weekly_cf.reset_index()

        return weekly_cf

    def standardize_features(self, df):
        feature_names = self.scaler_params['feature_names']
        target_mean = self.scaler_params['target_mean']
        target_std = self.scaler_params['target_std']

        for feature in feature_names:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0)

                if feature == 'cash_flow_growth':
                    df[feature] = (df[feature] - target_mean) / target_std
                else:
                    col_mean = df[feature].mean()
                    col_std = df[feature].std()
                    if col_std > 0:
                        df[feature] = (df[feature] - col_mean) / col_std

        for feature in feature_names:
            if feature in df.columns:
                df[feature] = df[feature].clip(-10, 10)

        return df

    def prepare_features_for_prediction(self, cust_id, transactions_df):
        if len(transactions_df) < self.MIN_OBSERVATIONS:
            raise ValueError(f"Need at least {self.MIN_OBSERVATIONS} transactions")

        print(f"  → Calculating static features...")
        static_features = self.calculate_static_features(transactions_df)

        print(f"  → Classifying to cluster...")
        cluster = self.classify_cluster(static_features)
        print(f"  → Cluster: {cluster}")

        print(f"  → Calculating time-series features...")
        features_df = self.calculate_timeseries_features(transactions_df, static_features)

        features_df['cluster'] = cluster
        features_df['cust_id'] = cust_id

        print(f"  → Standardizing features...")
        features_df = self.standardize_features(features_df)

        last_cashflow = transactions_df.sort_values('BookingDatetime').apply(
            lambda row: row['Amount_EUR'] if row['D_C'] == 'C' else -row['Amount_EUR'],
            axis=1
        ).iloc[-1]

        print(f"  ✓ Features ready: {len(features_df)} weeks")

        return {
            'features_df': features_df,
            'cluster': cluster,
            'last_cashflow': last_cashflow,
            'static_features': static_features
        }

    def get_last_n_weeks(self, features_df, n_weeks=4):
        """Get the most recent N weeks of features for prediction"""
        return features_df.tail(n_weeks).copy()


# Example usage
if __name__ == "__main__":
    import sqlite3

    # Initialize engine
    engine = CompleteFeatureEngine()

    # Get transactions for a new customer
    conn = sqlite3.connect('../db/customers.db')

    # Get first customer
    cust_id = pd.read_sql("SELECT DISTINCT cust_id FROM transactions LIMIT 1", conn).iloc[0]['cust_id']

    # Get their transactions
    transactions = pd.read_sql(
        f"SELECT * FROM transactions WHERE cust_id = '{cust_id}'",
        conn
    )
    conn.close()

    print(f"\nProcessing customer: {cust_id}")
    print(f"Transactions: {len(transactions)}")

    # Prepare features
    result = engine.prepare_features_for_prediction(cust_id, transactions)

    print(f"\nResults:")
    print(f"  Cluster: {result['cluster']}")
    print(f"  Last cash flow: €{result['last_cashflow']:,.2f}")
    print(f"  Features shape: {result['features_df'].shape}")
    print(f"  Ready for prediction!")