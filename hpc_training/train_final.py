"""
FINAL: Parallel training using python3 (verified working)
"""

import pandas as pd
import subprocess
import os
import time
from datetime import datetime
from config import *


def verify_gpus():
    """Verify all GPUs"""
    
    print("="*80)
    print("GPU VERIFICATION")
    print("="*80)
    
    import torch
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available!")
        return False, 0
    
    n_gpus = torch.cuda.device_count()
    print(f"✓ Found {n_gpus} GPU(s)")
    
    for i in range(n_gpus):
        torch.cuda.set_device(i)
        name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {name}")
    
    print("="*80)
    return True, n_gpus


def train_single_cluster_script(cluster_name, gpu_id):
    """Generate training script for one cluster"""
    
    script = f"""#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'

import pandas as pd
import torch
import json
import sys

print(f"Cluster: {cluster_name}, GPU: {gpu_id}")
print(f"CUDA available: {{torch.cuda.is_available()}}")

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

from fast_training_ddp import train_cluster

features = pd.read_csv('customer_timeseries_features.csv')

print(f"Starting training for {cluster_name}...")
result = train_cluster('{cluster_name}', features, gpu_id=0)

# Save models directly
import joblib
import os
os.makedirs('models_dl', exist_ok=True)

for model_name, model_result in result['results'].items():
    if model_name == 'KNN':
        joblib.dump(model_result['model'], f"models_dl/{cluster_name}_KNN.pkl")
    else:
        torch.save(model_result['model'].state_dict(), f"models_dl/{cluster_name}_{{model_name}}.pt")

# Save result summary
result_data = {{
    'cluster': result['cluster'],
    'best_model': result['best_model'],
    'training_time': result['training_time'],
    'n_customers': result['n_customers'],
    'n_observations': result['n_observations'],
    'knn_rmse': result['results']['KNN']['test_rmse'],
    'lstm_rmse': result['results']['LSTM']['test_rmse'],
    'gru_rmse': result['results']['GRU']['test_rmse'],
    'transformer_rmse': result['results']['Transformer']['test_rmse']
}}

with open('result_{cluster_name}.json', 'w') as f:
    json.dump(result_data, f)

print(f"✓ {cluster_name} complete! Time: {{result['training_time']/60:.1f}} min")
"""
    return script


def main():
    """Main parallel training"""
    
    overall_start = time.time()
    
    print("="*80)
    print("PARALLEL CASH FLOW FORECASTING")
    print("="*80)
    print(f"Start: {datetime.now()}")
    print("="*80)
    
    # Verify GPUs
    gpus_ok, n_gpus = verify_gpus()
    if not gpus_ok:
        return
    
    # Load features
    if not os.path.exists('customer_timeseries_features.csv'):
        print("\n✗ Features file not found!")
        return
    
    features = pd.read_csv('customer_timeseries_features.csv')
    clusters = sorted(features['cluster'].unique())
    
    print(f"\n✓ Clusters: {len(clusters)}")
    print(f"  {clusters}\n")
    
    all_results = []
    
    # Train in batches
    for batch_start in range(0, len(clusters), n_gpus):
        batch_clusters = clusters[batch_start:batch_start+n_gpus]
        batch_num = batch_start//n_gpus + 1
        
        print("="*80)
        print(f"BATCH {batch_num}: {batch_clusters}")
        print("="*80)
        
        processes = []
        
        for i, cluster in enumerate(batch_clusters):
            gpu_id = i
            
            # Create script
            script = train_single_cluster_script(cluster, gpu_id)
            script_file = f'train_{cluster}.py'
            
            with open(script_file, 'w') as f:
                f.write(script)
            
            os.chmod(script_file, 0o755)
            
            print(f"  Launching: {cluster} → GPU {gpu_id}")
            
            log_file = f'log_{cluster}.txt'
            with open(log_file, 'w') as log:
                # Use python3 explicitly
                p = subprocess.Popen(
                    ['python3', script_file],
                    stdout=log,
                    stderr=subprocess.STDOUT
                )
                processes.append((p, cluster, gpu_id, script_file, log_file))
        
        print(f"\n  Waiting for batch {batch_num}...\n")
        
        # Wait for completion
        for p, cluster, gpu_id, script_file, log_file in processes:
            p.wait()
            
            if p.returncode == 0:
                print(f"  ✓ {cluster} (GPU {gpu_id})")
                
                import json
                result_file = f'result_{cluster}.json'
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    print(f"      {data['best_model']}: RMSE {data[data['best_model'].lower() + '_rmse']:.4f}, "
                          f"Time {data['training_time']/60:.1f}min")
                    
                    all_results.append({
                        'cluster': data['cluster'],
                        'best_model': data['best_model'],
                        'training_time': data['training_time'],
                        'n_customers': data['n_customers'],
                        'n_observations': data['n_observations'],
                        'results': {
                            'KNN': {'test_rmse': data['knn_rmse']},
                            'LSTM': {'test_rmse': data['lstm_rmse']},
                            'GRU': {'test_rmse': data['gru_rmse']},
                            'Transformer': {'test_rmse': data['transformer_rmse']}
                        }
                    })
            else:
                print(f"  ✗ {cluster} FAILED (check {log_file})")
            
            if os.path.exists(script_file):
                os.remove(script_file)
    
    # Save summary (models already saved by subprocesses)
    if all_results:
        print("\n" + "="*80)
        print("SAVING SUMMARY")
        print("="*80)

        summary_data = []
        for result in all_results:
            best = result['best_model']
            summary_data.append({
                'cluster': result['cluster'],
                'best_model': best,
                'test_rmse': result['results'][best]['test_rmse'],
                'training_time_min': result['training_time'] / 60,
                'n_customers': result['n_customers'],
                'n_observations': result['n_observations'],
                'knn_rmse': result['results']['KNN']['test_rmse'],
                'lstm_rmse': result['results']['LSTM']['test_rmse'],
                'gru_rmse': result['results']['GRU']['test_rmse'],
                'transformer_rmse': result['results']['Transformer']['test_rmse']
            })

        summary_df = pd.DataFrame(summary_data)
        os.makedirs('models_dl', exist_ok=True)
        summary_df.to_csv('models_dl/training_summary.csv', index=False)
        print(summary_df.to_string(index=False))
        
        total_time = (time.time() - overall_start) / 3600
        
        print(f"\n{'='*80}")
        print("✓ COMPLETE!")
        print(f"{'='*80}")
        print(f"Time: {total_time:.2f} hours")
        print(f"Clusters: {len(all_results)}/{len(clusters)}")
        print(f"End: {datetime.now()}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
