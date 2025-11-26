import pandas as pd
import numpy as np

df = pd.read_csv('customer_timeseries_features.csv')

# 1. Check target distribution
print("Target (cash_flow_growth):")
print(df['cash_flow_growth'].describe())
print(f"Range: [{df['cash_flow_growth'].min():.2f}, {df['cash_flow_growth'].max():.2f}]")

# Should be: Range ≈ [-2.0, 2.0] after clipping

# 2. Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")
# Lag columns will have NaN for first few rows (expected)

# 3. Check for extreme values
extreme = df[np.abs(df['cash_flow_growth']) > 3]
print(f"\nExtreme growth values: {len(extreme)}")
# Should be 0 after clipping!

# 4. Check cluster distribution
print(f"\nCluster distribution:\n{df['cluster'].value_counts()}")