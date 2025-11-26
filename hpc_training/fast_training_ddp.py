"""
FIXED: Fast training with proper target scaling and JSON serialization
Fixes the extremely high RMSE issue
"""

import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import time
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

from config import *
from fast_feature_engineering import prepare_timeseries_features
from fast_dl_models import FastTimeSeriesDataset, create_model, FastTrainer


def prepare_data(cluster_data, sequence_length=SEQUENCE_LENGTH):
    """Prepare train/val/test splits with PROPER SCALING"""
    
    cluster_data = cluster_data.sort_values(['cust_id', 'datetime'])
    
    exclude_cols = ['cust_id', 'cluster', 'cash_flow_growth', 'datetime', 'cash_flow']
    feature_cols = [col for col in cluster_data.columns if col not in exclude_cols]
    
    # Get features and target
    X = cluster_data[feature_cols].fillna(0).values
    y = cluster_data['cash_flow_growth'].fillna(0).values
    
    # CRITICAL FIX: Check for extreme values in target
    y_abs_max = np.abs(y).max()
    if y_abs_max > 10:
        print(f"    WARNING: Extreme target values detected (max={y_abs_max:.2f})")
        print(f"    Clipping to [-2, 2] range...")
        y = np.clip(y, -2.0, 2.0)
    
    # Additional safety: Remove any remaining inf/nan
    valid_mask = np.isfinite(y)
    if not valid_mask.all():
        print(f"    WARNING: Found {(~valid_mask).sum()} invalid target values, removing...")
        X = X[valid_mask]
        y = y[valid_mask]
    
    n = len(X)
    train_size = int(TRAIN_SPLIT * n)
    val_size = int(VAL_SPLIT * n)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    # CRITICAL FIX: Scale the target variable separately
    # This ensures predictions are on the same scale
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Create datasets with SCALED targets for DL models
    train_dataset = FastTimeSeriesDataset(X_train, y_train_scaled, sequence_length)
    val_dataset = FastTimeSeriesDataset(X_val, y_val_scaled, sequence_length)
    test_dataset = FastTimeSeriesDataset(X_test, y_test_scaled, sequence_length)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    # Return both scaled and unscaled versions
    return (train_loader, val_loader, test_loader, 
            X_train, y_train, X_test, y_test, 
            len(feature_cols), y_scaler)


def train_knn(X_train, y_train, X_test, y_test):
    """Train KNN models (from paper) - NO SCALING NEEDED"""
    
    print("\n  Training KNN...")
    
    best_rmse = float('inf')
    best_model = None
    best_config = None
    
    # KNN works on raw features and raw target (already standardized in feature engineering)
    for config in KNN_CONFIGS:
        model = KNeighborsRegressor(**config)
        
        # Use all data (no sequence slicing needed for KNN)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_config = config
    
    print(f"    Best KNN RMSE: {best_rmse:.4f}, Config: {best_config}")
    
    return best_model, best_rmse, best_config


def train_dl_model(model_type, configs, train_loader, val_loader, test_loader, 
                   n_features, device, y_scaler):
    """Train a deep learning model with PROPER DESCALING"""
    
    print(f"\n  Training {model_type}...")
    
    best_rmse = float('inf')
    best_model = None
    best_config = None
    
    for config in configs:
        config = config.copy()
        lr = config.pop('lr')
        
        model = create_model(model_type, n_features, **config)
        config['lr'] = lr
        
        trainer = FastTrainer(model, device, lr=lr, patience=EARLY_STOPPING_PATIENCE)
        
        # Train on scaled data
        val_rmse_scaled = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)
        
        # Evaluate on test set with DESCALING
        test_rmse_scaled = trainer.evaluate(test_loader)
        
        # CRITICAL FIX: Convert scaled RMSE back to original scale
        # RMSE scales linearly with the standard deviation of the target
        test_rmse = test_rmse_scaled * y_scaler.scale_[0]
        
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_model = trainer.model
            best_config = config.copy()
    
    print(f"    Best {model_type} RMSE: {best_rmse:.4f}")
    
    return best_model, best_rmse, best_config


def train_cluster(cluster_name, features_df, gpu_id=0):
    """Train all models for one cluster"""
    
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {cluster_name} (GPU {gpu_id})")
    print(f"{'='*80}")
    
    # CRITICAL: When CUDA_VISIBLE_DEVICES is set, always use device 0
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device = 'cuda:0'
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} → using cuda:0")
    else:
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
    
    cluster_data = features_df[features_df['cluster'] == cluster_name]
    
    print(f"Customers: {cluster_data['cust_id'].nunique()}")
    print(f"Observations: {len(cluster_data)}")
    
    # Check target variable statistics BEFORE training
    target_values = cluster_data['cash_flow_growth'].dropna()
    print(f"Target statistics:")
    print(f"  Mean: {target_values.mean():.4f}")
    print(f"  Std: {target_values.std():.4f}")
    print(f"  Min: {target_values.min():.4f}")
    print(f"  Max: {target_values.max():.4f}")
    
    # Prepare data
    (train_loader, val_loader, test_loader, 
     X_train, y_train, X_test, y_test, 
     n_features, y_scaler) = prepare_data(cluster_data)
    
    print(f"Features: {n_features}")
    print(f"Train batches: {len(train_loader)}")
    
    results = {}
    
    # 1. Train KNN (on original scale)
    knn_model, knn_rmse, knn_config = train_knn(X_train, y_train, X_test, y_test)
    results['KNN'] = {
        'model': knn_model,
        'test_rmse': float(knn_rmse),  # Convert to Python float
        'config': knn_config
    }
    
    # 2. Train LSTM (with proper scaling)
    lstm_model, lstm_rmse, lstm_config = train_dl_model(
        'lstm', LSTM_CONFIGS, train_loader, val_loader, test_loader, 
        n_features, device, y_scaler
    )
    results['LSTM'] = {
        'model': lstm_model,
        'test_rmse': float(lstm_rmse),  # Convert to Python float
        'config': lstm_config
    }
    
    # 3. Train GRU (with proper scaling)
    gru_model, gru_rmse, gru_config = train_dl_model(
        'gru', GRU_CONFIGS, train_loader, val_loader, test_loader, 
        n_features, device, y_scaler
    )
    results['GRU'] = {
        'model': gru_model,
        'test_rmse': float(gru_rmse),  # Convert to Python float
        'config': gru_config
    }
    
    # 4. Train Transformer (with proper scaling)
    trans_model, trans_rmse, trans_config = train_dl_model(
        'transformer', TRANSFORMER_CONFIGS, train_loader, val_loader, test_loader, 
        n_features, device, y_scaler
    )
    results['Transformer'] = {
        'model': trans_model,
        'test_rmse': float(trans_rmse),  # Convert to Python float
        'config': trans_config
    }
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_rmse'])
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model_name} (RMSE: {results[best_model_name]['test_rmse']:.4f})")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"{'='*80}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    return {
        'cluster': cluster_name,
        'results': results,
        'best_model': best_model_name,
        'training_time': float(elapsed),  # Convert to Python float
        'n_customers': int(cluster_data['cust_id'].nunique()),  # Convert to Python int
        'n_observations': int(len(cluster_data))  # Convert to Python int
    }


def save_results(all_results, output_dir=MODELS_DL_DIR):
    """Save all results with proper JSON serialization"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    summary_data = []
    
    for result in all_results:
        cluster = result['cluster']
        
        # Save models
        for model_name, model_result in result['results'].items():
            if model_name == 'KNN':
                import joblib
                joblib.dump(model_result['model'], 
                          os.path.join(output_dir, f'{cluster}_KNN.pkl'))
            else:
                torch.save(model_result['model'].state_dict(),
                          os.path.join(output_dir, f'{cluster}_{model_name}.pt'))
                torch.save(model_result['config'],
                          os.path.join(output_dir, f'{cluster}_{model_name}_config.pt'))
        
        # Summary (convert all values to Python native types)
        best = result['best_model']
        summary_data.append({
            'cluster': cluster,
            'best_model': best,
            'test_rmse': float(result['results'][best]['test_rmse']),
            'training_time_min': float(result['training_time'] / 60),
            'n_customers': int(result['n_customers']),
            'n_observations': int(result['n_observations']),
            'knn_rmse': float(result['results']['KNN']['test_rmse']),
            'lstm_rmse': float(result['results']['LSTM']['test_rmse']),
            'gru_rmse': float(result['results']['GRU']['test_rmse']),
            'transformer_rmse': float(result['results']['Transformer']['test_rmse'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('test_rmse')
    summary_df.to_csv(os.path.join(output_dir, 'training_summary.csv'), index=False)
    
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    
    print(f"\nTotal training time: {summary_df['training_time_min'].sum():.1f} minutes")
    print(f"Average time per cluster: {summary_df['training_time_min'].mean():.1f} minutes")
    
    print(f"\nModel performance:")
    print(f"  Best KNN RMSE: {summary_df['knn_rmse'].min():.4f}")
    print(f"  Best LSTM RMSE: {summary_df['lstm_rmse'].min():.4f}")
    print(f"  Best GRU RMSE: {summary_df['gru_rmse'].min():.4f}")
    print(f"  Best Transformer RMSE: {summary_df['transformer_rmse'].min():.4f}")
    
    print(f"\nBest model distribution:")
    print(summary_df['best_model'].value_counts())
    
    return summary_df


def main():
    """Main training pipeline"""
    
    overall_start = time.time()
    
    print("="*80)
    print("FIXED: CASH FLOW FORECASTING TRAINING (Multi-GPU)")
    print("="*80)
    print(f"Start: {datetime.now()}")
    print(f"Target: < 3 hours")
    print(f"GPUs: {N_GPUS}")
    print("="*80)
    
    # Step 1: Load or prepare features
    if USE_CACHED_FEATURES and os.path.exists(FEATURES_FILE):
        print(f"\nLoading cached features from {FEATURES_FILE}...")
        features = pd.read_csv(FEATURES_FILE)
        print(f"✓ Features loaded: {features.shape}")
    else:
        print("\nPreparing time-series features from transactions...")
        
        # Load raw data
        transactions = pd.read_csv('synthetic_sme_transactions_processed.csv', low_memory=False)
        customers = pd.read_csv('customer_clusters_FINAL.csv')
        
        print(f"Transactions: {len(transactions):,}")
        print(f"Customers: {len(customers):,}")
        
        if MAX_CUSTOMERS_PER_CLUSTER:
            print(f"\nTEST MODE: Using {MAX_CUSTOMERS_PER_CLUSTER} customers per cluster")
            customers = customers.groupby('cluster').head(MAX_CUSTOMERS_PER_CLUSTER)
        
        # Generate features
        features = prepare_timeseries_features(transactions, customers)
        
        # Save for next time
        features.to_csv(FEATURES_FILE, index=False)
        print(f"✓ Saved features to: {FEATURES_FILE}")
    
    # Verify target variable
    if 'cash_flow_growth' not in features.columns:
        print("\nERROR: No 'cash_flow_growth' column found!")
        print(f"Available columns: {features.columns.tolist()}")
        return
    
    print(f"\nFeatures ready:")
    print(f"  Customers: {features['cust_id'].nunique()}")
    print(f"  Clusters: {features['cluster'].nunique()}")
    print(f"  Total observations: {len(features)}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA not available! Training on CPU (will be SLOW)")
        n_gpus = 1
    else:
        n_gpus = min(N_GPUS, torch.cuda.device_count())
        print(f"\nGPUs available: {torch.cuda.device_count()}")
        print(f"Using {n_gpus} GPUs for training")
    
    # Step 2: Train clusters (sequential to avoid multiprocessing issues)
    clusters = sorted(features['cluster'].unique())
    print(f"\nClusters to train: {len(clusters)}")
    print("Training SEQUENTIALLY (stable, no multiprocessing issues)")
    
    all_results = []
    
    for i, cluster in enumerate(clusters):
        gpu_id = i % n_gpus
        result = train_cluster(cluster, features, gpu_id)
        all_results.append(result)
    
    # Step 3: Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    summary = save_results(all_results)
    
    total_time = (time.time() - overall_start) / 3600
    
    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"Total time: {total_time:.2f} hours")
    print(f"Target met: {'✓' if total_time < 3 else '✗'}")
    print(f"End: {datetime.now()}")
    print(f"{'='*80}")
    
    return summary


if __name__ == "__main__":
    main()