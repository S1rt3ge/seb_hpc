"""
PRODUCTION: Load saved models and make predictions
Use this for investment simulation
"""

import pandas as pd
import numpy as np
import torch
import joblib
import json
import os
from config import *
from fast_dl_models import create_model, FastTimeSeriesDataset
from torch.utils.data import DataLoader


class CashFlowPredictor:
    """Load and use trained models for predictions"""
    
    def __init__(self, models_dir=MODELS_DL_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.metadata = {}
        self.scaler_params = None
        
        # Load scaler params
        if os.path.exists('scaler_params.json'):
            with open('scaler_params.json', 'r') as f:
                self.scaler_params = json.load(f)
                print(f"✓ Loaded scaler params (std: €{self.scaler_params['target_std']:,.2f})")
        else:
            print("⚠ Warning: scaler_params.json not found, predictions will be in standardized units")
    
    def load_cluster_models(self, cluster_name):
        """Load all models for a specific cluster"""
        
        print(f"\nLoading models for {cluster_name}...")
        
        # Load metadata
        metadata_path = os.path.join(self.models_dir, f'{cluster_name}_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata[cluster_name] = json.load(f)
            print(f"  Best model: {self.metadata[cluster_name]['best_model']}")
        
        cluster_models = {}
        
        # Load KNN
        knn_path = os.path.join(self.models_dir, f'{cluster_name}_KNN.pkl')
        if os.path.exists(knn_path):
            cluster_models['KNN'] = joblib.load(knn_path)
            print(f"  ✓ Loaded KNN")
        
        # Load deep learning models
        for model_type in ['LSTM', 'GRU', 'Transformer']:
            model_path = os.path.join(self.models_dir, f'{cluster_name}_{model_type}.pt')
            config_path = os.path.join(self.models_dir, f'{cluster_name}_{model_type}_config.pt')
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                config = torch.load(config_path)
                
                # Get number of features from metadata or default
                n_features = 26  # Default from your training
                
                # Create model
                config_copy = config.copy()
                config_copy.pop('lr', None)
                config_copy.pop('batch_size', None)
                model = create_model(model_type.lower(), n_features, **config_copy)
                
                # Load weights
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                
                cluster_models[model_type] = model
                print(f"  ✓ Loaded {model_type}")
        
        self.models[cluster_name] = cluster_models
        return cluster_models
    
    def predict_knn(self, cluster_name, X):
        """Predict using KNN model"""
        if cluster_name not in self.models:
            self.load_cluster_models(cluster_name)
        
        model = self.models[cluster_name]['KNN']
        return model.predict(X)
    
    def predict_dl(self, cluster_name, model_type, X, sequence_length=SEQUENCE_LENGTH):
        """Predict using deep learning model"""
        if cluster_name not in self.models:
            self.load_cluster_models(cluster_name)
        
        model = self.models[cluster_name][model_type]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # Create dummy targets for dataset (not used in prediction)
        y_dummy = np.zeros(len(X))
        
        dataset = FastTimeSeriesDataset(X, y_dummy, sequence_length)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(device)
                pred = model(batch_X)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions).flatten()
    
    def predict_best(self, cluster_name, X, sequence_length=SEQUENCE_LENGTH):
        """Predict using the best model for this cluster"""
        if cluster_name not in self.metadata:
            self.load_cluster_models(cluster_name)
        
        best_model = self.metadata[cluster_name]['best_model']
        
        if best_model == 'KNN':
            predictions = self.predict_knn(cluster_name, X)
        else:
            predictions = self.predict_dl(cluster_name, best_model, X, sequence_length)
        
        return predictions, best_model
    
    def to_euros(self, standardized_values):
        """Convert standardized predictions to EUR"""
        if self.scaler_params is None:
            raise ValueError("Scaler params not loaded! Cannot convert to EUR")
        
        return standardized_values * self.scaler_params['target_std']
    
    def predict_future_cashflow(self, customer_id, features_df, n_months=12):
        """
        Predict future cash flow for a customer
        
        Args:
            customer_id: Customer ID
            features_df: DataFrame with customer features (from customer_timeseries_features.csv)
            n_months: Number of months to predict ahead
        
        Returns:
            predictions: Array of predictions in EUR
            cluster: Customer's cluster
            model_used: Model type used
        """
        # Get customer data
        customer_data = features_df[features_df['cust_id'] == customer_id].copy()
        
        if len(customer_data) == 0:
            raise ValueError(f"Customer {customer_id} not found!")
        
        # Get cluster
        cluster = customer_data['cluster'].iloc[0]
        
        # Prepare features
        exclude_cols = ['cust_id', 'cluster', 'cash_flow_growth', 'datetime', 'cash_flow']
        feature_cols = [col for col in customer_data.columns if col not in exclude_cols]
        X = customer_data[feature_cols].fillna(0).values
        
        # Get predictions
        predictions_std, model_used = self.predict_best(cluster, X)
        
        # Convert to EUR (these are growth rates)
        # To get actual cash flow, you need the last known cash flow
        last_cashflow = customer_data['cash_flow'].iloc[-1]
        
        # Predictions are standardized growth rates, convert to EUR growth
        growth_eur = self.to_euros(predictions_std[-n_months:])
        
        # Generate future cash flows (cumulative)
        future_cashflows = [last_cashflow]
        for growth in growth_eur:
            future_cashflows.append(future_cashflows[-1] + growth)
        
        return {
            'predictions': np.array(future_cashflows[1:]),
            'growth_rates': growth_eur,
            'cluster': cluster,
            'model_used': model_used,
            'last_known_cashflow': last_cashflow
        }


def example_usage():
    """Example: How to use the predictor"""
    
    print("="*80)
    print("CASH FLOW PREDICTION EXAMPLE")
    print("="*80)
    
    # Initialize predictor
    predictor = CashFlowPredictor()
    
    # Load features
    features = pd.read_csv(FEATURES_FILE)
    print(f"\n✓ Loaded features: {len(features)} observations")
    
    # Get a sample customer
    sample_customer = features['cust_id'].iloc[0]
    print(f"\nPredicting for customer: {sample_customer}")
    
    # Make prediction
    result = predictor.predict_future_cashflow(sample_customer, features, n_months=6)
    
    print(f"\n{'='*80}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*80}")
    print(f"Customer: {sample_customer}")
    print(f"Cluster: {result['cluster']}")
    print(f"Model used: {result['model_used']}")
    print(f"Last known cash flow: €{result['last_known_cashflow']:,.2f}")
    print(f"\nNext 6 months forecast:")
    for i, (cashflow, growth) in enumerate(zip(result['predictions'], result['growth_rates']), 1):
        print(f"  Month {i}: €{cashflow:>10,.2f} (change: €{growth:>8,.2f})")
    
    # Calculate surplus for investment
    print(f"\n{'='*80}")
    print(f"INVESTMENT SIMULATION")
    print(f"{'='*80}")
    
    # Assume minimum cash requirement
    min_cash_requirement = 50000  # €50k minimum
    
    surplus_funds = []
    for month, cashflow in enumerate(result['predictions'], 1):
        surplus = max(0, cashflow - min_cash_requirement)
        surplus_funds.append(surplus)
        print(f"  Month {month}: Cash €{cashflow:>10,.2f} → Surplus €{surplus:>10,.2f}")
    
    total_surplus = sum(surplus_funds)
    print(f"\nTotal surplus available for investment: €{total_surplus:,.2f}")
    
    return predictor, result


if __name__ == "__main__":
    # Run example
    predictor, result = example_usage()
    
    print(f"\n{'='*80}")
    print("✓ Prediction module ready!")
    print(f"{'='*80}")
    print("\nTo use in your code:")
    print("""
from predict_cashflow import CashFlowPredictor
import pandas as pd

# Load predictor
predictor = CashFlowPredictor()

# Load features
features = pd.read_csv('customer_timeseries_features_conservative.csv')

# Predict for any customer
result = predictor.predict_future_cashflow(customer_id='CUST_001', 
                                          features_df=features, 
                                          n_months=12)

# Access results
print(f"Predictions: {result['predictions']}")
print(f"Model used: {result['model_used']}")
    """)
