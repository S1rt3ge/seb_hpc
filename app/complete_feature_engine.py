"""
Complete Feature Engineering from Raw Transactions ONLY
Works for NEW customers not in any training dataset
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta


class CompleteFeatureEngine:
    """
    Calculate ALL features from raw transactions
    No dependency on pre-computed CSV files
    """

    def __init__(self, scaler_params_path='../prediction/scaler_params.json'):
        """Load only scaler parameters for standardization"""
        with open(scaler_params_path, 'r') as f:
            self.scaler_params = json.load(f)

        self.MIN_OBSERVATIONS = 20
        self.N_LAGS = 3

        print(f"✓ Loaded scaler params (target_std: €{self.scaler_params['target_std']:,.2f})")

    def calculate_static_features(self, transactions_df):
        """
        Calculate static customer features from raw transactions
        These describe the customer's business pattern
        """

        # Channel percentages
        channel_counts = transactions_df['Channel'].value_counts()
        total_txns = len(transactions_df)

        channels = {
            'channel_pos_pct': channel_counts.get('POS', 0) / total_txns,
            'channel_internet_bank_pct': channel_counts.get('Internet Bank', 0) / total_txns,
            'channel_mobile_bank_pct': channel_counts.get('Mobile Bank', 0) / total_txns,
            'channel_atm_pct': channel_counts.get('ATM', 0) / total_txns,
        }

        # Transaction statistics
        amounts = transactions_df['Amount_EUR'].abs()

        # Counterparty analysis
        counterparties = transactions_df['Counterparty_IBAN'].nunique()

        # Top 5 concentration
        top5_sum = transactions_df.groupby('Counterparty_IBAN')['Amount_EUR'].sum().abs().nlargest(5).sum()
        total_sum = amounts.sum()
        top5_concentration = top5_sum / total_sum if total_sum > 0 else 0

        # Transaction size categories
        small_threshold = amounts.quantile(0.25)
        large_threshold = amounts.quantile(0.75)

        small_txn_pct = (amounts < small_threshold).sum() / total_txns
        large_txn_pct = (amounts > large_threshold).sum() / total_txns

        # Volatility
        cv_coefficient = amounts.std() / amounts.mean() if amounts.mean() > 0 else 0
        median_transaction_size = amounts.median()
        volatility_std = amounts.std()

        # Negative days (days with net negative cash flow)
        transactions_df['date'] = pd.to_datetime(transactions_df['BookingDatetime']).dt.date
        daily_cf = transactions_df.groupby('date').apply(
            lambda x: (x[x['D_C'] == 'C']['Amount_EUR'].sum() -
                       x[x['D_C'] == 'D']['Amount_EUR'].sum())
        )
        negative_day_pct = (daily_cf < 0).sum() / len(daily_cf) if len(daily_cf) > 0 else 0

        # Transaction frequency (avg transactions per day)
        date_range = (transactions_df['BookingDatetime'].max() -
                      transactions_df['BookingDatetime'].min()).days
        transaction_frequency = total_txns / date_range if date_range > 0 else 0

        return {
            'channel_pos_pct': channels['channel_pos_pct'],
            'channel_internet_bank_pct': channels['channel_internet_bank_pct'],
            'top5_concentration': top5_concentration,
            'unique_counterparties': counterparties,
            'small_txn_pct': small_txn_pct,
            'large_txn_pct': large_txn_pct,
            'cv_coefficient': cv_coefficient,
            'median_transaction_size': median_transaction_size,
            'volatility_std': volatility_std,
            'negative_day_pct': negative_day_pct,
            'transaction_frequency': transaction_frequency
        }

    def classify_cluster(self, static_features):
        """
        Classify customer to cluster based on their features
        Uses business rules based on patterns from training
        """

        # Extract key features for classification
        internet_pct = static_features['channel_internet_bank_pct']
        pos_pct = static_features['channel_pos_pct']
        freq = static_features['transaction_frequency']
        counterparties = static_features['unique_counterparties']
        concentration = static_features['top5_concentration']

        # Rule-based classification (simplified from training clusters)

        # Subscription SaaS: High internet banking, regular frequency, concentrated counterparties
        if internet_pct > 0.7 and freq > 2.0 and concentration > 0.6:
            return 'Subscription_SaaS'

        # B2B Services: High counterparties, mixed channels, moderate concentration
        elif counterparties > 30 and concentration < 0.7:
            return 'B2B_Services_Combined'

        # Retail: High POS, many counterparties, low concentration
        elif pos_pct > 0.2 and counterparties > 20:
            return 'Retail_Combined'

        # Digital B2C: High internet, high frequency, low POS
        elif internet_pct > 0.8 and pos_pct < 0.1:
            return 'Digital_B2C_Combined'

        # Default: Mixed
        else:
            return 'Mixed'

    def calculate_timeseries_features(self, transactions_df, static_features):
        """
        Calculate time-series features from raw transactions
        Aggregates to weekly level with lags and rolling stats
        """

        # Ensure datetime column
        transactions_df['datetime'] = pd.to_datetime(transactions_df['BookingDatetime'])
        transactions_df = transactions_df.sort_values('datetime')

        # Calculate cash flow (Credits - Debits)
        transactions_df['cash_flow'] = transactions_df.apply(
            lambda row: row['Amount_EUR'] if row['D_C'] == 'C' else -row['Amount_EUR'],
            axis=1
        )

        # Aggregate to weekly level
        weekly_cf = transactions_df.groupby(pd.Grouper(key='datetime', freq='W')).agg({
            'cash_flow': 'sum',
            'Amount_EUR': ['count', 'mean', 'std']
        })

        weekly_cf.columns = ['cash_flow', 'txn_count', 'avg_txn', 'std_txn']
        weekly_cf = weekly_cf[weekly_cf['txn_count'] > 0]  # Remove empty weeks

        if len(weekly_cf) < self.MIN_OBSERVATIONS:
            raise ValueError(f"Not enough weekly observations (need {self.MIN_OBSERVATIONS})")

        # Calculate target: cash flow growth (absolute change in EUR)
        weekly_cf['cash_flow_growth'] = weekly_cf['cash_flow'].diff()

        # Add lagged features
        for lag in range(1, self.N_LAGS + 1):
            weekly_cf[f'growth_lag_{lag}'] = weekly_cf['cash_flow_growth'].shift(lag)
            weekly_cf[f'cashflow_lag_{lag}'] = weekly_cf['cash_flow'].shift(lag)
            weekly_cf[f'txn_count_lag_{lag}'] = weekly_cf['txn_count'].shift(lag)

        # Add rolling statistics
        weekly_cf['growth_mean_rolling'] = weekly_cf['cash_flow_growth'].rolling(4, min_periods=1).mean()
        weekly_cf['growth_std_rolling'] = weekly_cf['cash_flow_growth'].rolling(4, min_periods=1).std()
        weekly_cf['cashflow_mean_rolling'] = weekly_cf['cash_flow'].rolling(4, min_periods=1).mean()

        # Add static features to each row
        for feature_name, feature_value in static_features.items():
            weekly_cf[feature_name] = feature_value

        # Remove NaN rows (from lags and diff)
        weekly_cf = weekly_cf.dropna(subset=['cash_flow_growth'])

        weekly_cf = weekly_cf.reset_index()

        return weekly_cf

    def standardize_features(self, df):
        """
        Standardize features using training parameters
        """

        feature_names = self.scaler_params['feature_names']
        target_mean = self.scaler_params['target_mean']
        target_std = self.scaler_params['target_std']

        # Standardize each feature
        for feature in feature_names:
            if feature in df.columns:
                df[feature] = df[feature].fillna(0)

                if feature == 'cash_flow_growth':
                    # Use exact training parameters for target
                    df[feature] = (df[feature] - target_mean) / target_std
                else:
                    # For other features, use column statistics
                    col_mean = df[feature].mean()
                    col_std = df[feature].std()
                    if col_std > 0:
                        df[feature] = (df[feature] - col_mean) / col_std

        # Clip extreme values
        for feature in feature_names:
            if feature in df.columns:
                df[feature] = df[feature].clip(-10, 10)

        return df

    def prepare_features_for_prediction(self, cust_id, transactions_df):
        """
        Complete pipeline: Raw transactions → Ready for prediction

        Args:
            cust_id: Customer ID
            transactions_df: Raw transactions with columns:
                [BookingDatetime, D_C, Amount_EUR, Channel, Counterparty_IBAN]

        Returns:
            dict with:
                - features_df: DataFrame ready for model
                - cluster: Assigned cluster
                - last_cashflow: Last known cash flow
        """

        if len(transactions_df) < self.MIN_OBSERVATIONS:
            raise ValueError(f"Customer needs at least {self.MIN_OBSERVATIONS} transactions")

        # Step 1: Calculate static features from raw transactions
        print(f"  → Calculating static features...")
        static_features = self.calculate_static_features(transactions_df)

        # Step 2: Classify to cluster
        print(f"  → Classifying to cluster...")
        cluster = self.classify_cluster(static_features)
        print(f"  → Cluster: {cluster}")

        # Step 3: Calculate time-series features
        print(f"  → Calculating time-series features...")
        features_df = self.calculate_timeseries_features(transactions_df, static_features)

        # Step 4: Add cluster and customer ID
        features_df['cluster'] = cluster
        features_df['cust_id'] = cust_id

        # Step 5: Standardize
        print(f"  → Standardizing features...")
        features_df = self.standardize_features(features_df)

        # Get last cash flow
        last_cashflow = transactions_df.sort_values('BookingDatetime')['cash_flow'].iloc[
            -1] if 'cash_flow' in transactions_df.columns else 0

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