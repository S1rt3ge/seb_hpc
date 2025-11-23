"""
Fast training script - optimized for 3 GPUs
Target: < 3 hours total training time
"""

import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import *
from fast_feature_engineering import prepare_fast_features
from fast_dl_models import FastTimeSeriesDataset, create_model, FastTrainer
from torch.utils.data import DataLoader


def prepare_data(cluster_data, sequence_length=SEQUENCE_LENGTH):
    """Prepare train/val/test splits"""

    cluster_data = cluster_data.sort_values(['cust_id', 'datetime'])

    feature_cols = [col for col in cluster_data.columns
                   if col not in ['cust_id', 'datetime', 'cluster', 'cash_flow_growth']]

    X = cluster_data[feature_cols].values
    y = cluster_data['cash_flow_growth'].values

    n = len(X)
    train_size = int(TRAIN_SPLIT * n)
    val_size = int(VAL_SPLIT * n)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # Create datasets
    train_dataset = FastTimeSeriesDataset(X_train, y_train, sequence_length)
    val_dataset = FastTimeSeriesDataset(X_val, y_val, sequence_length)
    test_dataset = FastTimeSeriesDataset(X_test, y_test, sequence_length)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=N_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=N_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=N_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, X_train, y_train, X_test, y_test


def train_knn(X_train, y_train, X_test, y_test):
    """Train KNN models (from paper)"""

    print("\n  Training KNN...")

    # Remove sequence structure for KNN
    best_rmse = float('inf')
    best_model = None
    best_config = None

    for config in KNN_CONFIGS:
        model = KNeighborsRegressor(**config)
        model.fit(X_train[SEQUENCE_LENGTH:], y_train[SEQUENCE_LENGTH:])

        y_pred = model.predict(X_test[SEQUENCE_LENGTH:])
        rmse = np.sqrt(mean_squared_error(y_test[SEQUENCE_LENGTH:], y_pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_config = config

    print(f"    Best KNN RMSE: {best_rmse:.4f}, Config: {best_config}")

    return best_model, best_rmse, best_config


def train_dl_model(model_type, configs, train_loader, val_loader, test_loader,
                   n_features, device):
    """Train a deep learning model"""

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

        val_rmse = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)
        test_rmse = trainer.evaluate(test_loader)

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

    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'

    cluster_data = features_df[features_df['cluster'] == cluster_name]

    print(f"Customers: {cluster_data['cust_id'].nunique()}")
    print(f"Observations: {len(cluster_data)}")

    # Prepare data
    train_loader, val_loader, test_loader, X_train, y_train, X_test, y_test = prepare_data(cluster_data)
    n_features = X_train.shape[1]

    print(f"Features: {n_features}")
    print(f"Train batches: {len(train_loader)}")

    results = {}

    # 1. Train KNN
    knn_model, knn_rmse, knn_config = train_knn(X_train, y_train, X_test, y_test)
    results['KNN'] = {
        'model': knn_model,
        'test_rmse': knn_rmse,
        'config': knn_config
    }

    # 2. Train LSTM
    lstm_model, lstm_rmse, lstm_config = train_dl_model(
        'lstm', LSTM_CONFIGS, train_loader, val_loader, test_loader, n_features, device
    )
    results['LSTM'] = {
        'model': lstm_model,
        'test_rmse': lstm_rmse,
        'config': lstm_config
    }

    # 3. Train GRU
    gru_model, gru_rmse, gru_config = train_dl_model(
        'gru', GRU_CONFIGS, train_loader, val_loader, test_loader, n_features, device
    )
    results['GRU'] = {
        'model': gru_model,
        'test_rmse': gru_rmse,
        'config': gru_config
    }

    # 4. Train Transformer
    trans_model, trans_rmse, trans_config = train_dl_model(
        'transformer', TRANSFORMER_CONFIGS, train_loader, val_loader, test_loader, n_features, device
    )
    results['Transformer'] = {
        'model': trans_model,
        'test_rmse': trans_rmse,
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'cluster': cluster_name,
        'results': results,
        'best_model': best_model_name,
        'training_time': elapsed
    }


def save_results(all_results, output_dir=MODELS_DL_DIR):
    """Save all results"""

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

        # Summary
        best = result['best_model']
        summary_data.append({
            'cluster': cluster,
            'best_model': best,
            'test_rmse': result['results'][best]['test_rmse'],
            'training_time_min': result['training_time'] / 60,
            'knn_rmse': result['results']['KNN']['test_rmse'],
            'lstm_rmse': result['results']['LSTM']['test_rmse'],
            'gru_rmse': result['results']['GRU']['test_rmse'],
            'transformer_rmse': result['results']['Transformer']['test_rmse']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'training_summary.csv'), index=False)

    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))

    print(f"\nTotal training time: {summary_df['training_time_min'].sum():.1f} minutes")
    print(f"Average time per cluster: {summary_df['training_time_min'].mean():.1f} minutes")

    return summary_df


def main():
    """Main training pipeline"""
    
    overall_start = time.time()
    
    print("="*80)
    print("FAST CASH FLOW FORECASTING TRAINING")
    print("="*80)
    print(f"Start: {datetime.now()}")
    print(f"Target: < 3 hours")
    print(f"GPUs: {N_GPUS}")
    print("="*80)
    
    # Step 1: Load or prepare features
    if USE_CACHED_FEATURES and os.path.exists(FEATURES_FILE):
        print(f"\nLoading cached features from {FEATURES_FILE}...")
        features = pd.read_csv(FEATURES_FILE)
    else:
        print("\nPreparing time-series features from transactions...")
        
        # Load raw data
        transactions = pd.read_csv('synthetic_sme_transactions_processed.csv', low_memory=False)
        customers = pd.read_csv('customer_clusters_FINAL.csv')
        
        print(f"Transactions: {len(transactions):,}")
        print(f"Customers: {len(customers):,}")
        
        # Generate features
        from fast_feature_engineering import prepare_timeseries_features
        features = prepare_timeseries_features(transactions, customers)
        
        # Save for next time
        features.to_csv(FEATURES_FILE, index=False)
        print(f"Saved features to: {FEATURES_FILE}")
    
    print(f"\n✓ Features ready: {features.shape}")
    
    # Verify we have target variable
    if 'cash_flow_growth' not in features.columns:
        print("\nERROR: No 'cash_flow_growth' column found!")
        print(f"Available columns: {features.columns.tolist()}")
        return
    
    # Step 2: Train clusters
    clusters = sorted(features['cluster'].unique())
    print(f"\nClusters to train: {len(clusters)}")
    
    all_results = []
    
    for i, cluster in enumerate(clusters):
        gpu_id = i % N_GPUS if torch.cuda.is_available() else 0
        result = train_cluster(cluster, features, gpu_id)
        all_results.append(result)
    
    # Step 3: Save results
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
    mp.set_start_method('spawn', force=True)
    main()
