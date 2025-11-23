"""
FIXED: Parallel training with proper Python environment handling
"""

import pandas as pd
import subprocess
import os
import time
from datetime import datetime
from config import *


def verify_gpus():
    """Verify all GPUs are accessible before training"""
    
    print("="*80)
    print("GPU VERIFICATION")
    print("="*80)
    
    import torch
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available!")
        return False, 0
    
    print("✓ CUDA is available")
    print(f"  PyTorch version: {torch.__version__}")
    
    n_gpus = torch.cuda.device_count()
    print(f"\n✓ Found {n_gpus} GPU(s)")
    
    if n_gpus == 0:
        return False, 0
    
    # Check each GPU
    print("\nVerifying individual GPUs:")
    for i in range(n_gpus):
        try:
            torch.cuda.set_device(i)
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            
            test_tensor = torch.zeros(100, 100).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            
            print(f"  GPU {i}: ✓ {name} ({memory_gb:.1f} GB)")
            
        except Exception as e:
            print(f"  GPU {i}: ✗ Error - {e}")
            return False, n_gpus
    
    print("\n" + "="*80)
    print(f"✓ ALL GPU CHECKS PASSED - Ready to use {n_gpus} GPUs")
    print("="*80)
    
    return True, n_gpus


def train_single_cluster_script(cluster_name, gpu_id):
    """Generate a standalone Python script for one cluster"""
    
    script = f"""#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Set GPU before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'

# Verify Python environment
print(f"Python version: {{sys.version}}")
print(f"Python executable: {{sys.executable}}")

import pandas as pd
import torch

print(f"\\nCluster: {cluster_name}")
print(f"Assigned GPU: {gpu_id}")

if torch.cuda.is_available():
    print(f"CUDA Device: {{torch.cuda.get_device_name(0)}}")
else:
    print("ERROR: CUDA not available!")
    sys.exit(1)

# Import training function
from fast_training_ddp import train_cluster
import json

# Load features
try:
    features = pd.read_csv('customer_timeseries_features.csv')
    print(f"Features loaded: {{features.shape}}")
except Exception as e:
    print(f"ERROR loading features: {{e}}")
    sys.exit(1)

# Train this cluster
print(f"\\nStarting training for {cluster_name}...")
try:
    # IMPORTANT: Always use gpu_id=0 because CUDA_VISIBLE_DEVICES is set
    # PyTorch will see only one GPU (renumbered as 0)
    result = train_cluster('{cluster_name}', features, gpu_id=0)
    
    # Save result
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
    
    print(f"\\n✓ Completed {cluster_name}! Time: {{result['training_time']/60:.1f}} min")
    
except Exception as e:
    print(f"\\nERROR training {cluster_name}: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    return script


def get_python_executable():
    """Get the correct Python executable"""
    import sys
    python_exe = sys.executable
    
    # Verify it's the right one
    try:
        result = subprocess.run(
            [python_exe, '-c', 'import torch; print(torch.cuda.is_available())'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and 'True' in result.stdout:
            return python_exe
    except:
        pass
    
    # Try common alternatives
    for exe in ['python3', 'python', sys.executable]:
        try:
            result = subprocess.run(
                [exe, '-c', 'import torch; print(torch.cuda.is_available())'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return exe
        except:
            continue
    
    return sys.executable  # Fallback


def main():
    """Main parallel training with proper Python environment"""
    
    overall_start = time.time()
    
    print("="*80)
    print("PARALLEL CASH FLOW FORECASTING (FIXED PYTHON ENV)")
    print("="*80)
    print(f"Start: {datetime.now()}")
    
    # Print Python info
    import sys
    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("="*80)
    
    # Get correct Python executable
    python_exe = get_python_executable()
    print(f"\nUsing Python: {python_exe}")
    
    # Verify GPUs
    gpus_ok, n_gpus = verify_gpus()
    
    if not gpus_ok:
        print("\n✗ GPU verification failed!")
        return
    
    print(f"\nWill use {n_gpus} GPUs for parallel training\n")
    
    # Load features
    if not os.path.exists('customer_timeseries_features.csv'):
        print("✗ ERROR: customer_timeseries_features.csv not found!")
        return
    
    print("Loading features...")
    features = pd.read_csv('customer_timeseries_features.csv')
    clusters = sorted(features['cluster'].unique())
    
    print(f"✓ Features loaded: {features.shape}")
    print(f"✓ Clusters to train: {len(clusters)}")
    print(f"  {clusters}\n")
    
    # Train in parallel batches
    all_results = []
    
    for batch_start in range(0, len(clusters), n_gpus):
        batch_clusters = clusters[batch_start:batch_start+n_gpus]
        batch_num = batch_start//n_gpus + 1
        
        print("="*80)
        print(f"BATCH {batch_num}: Training {len(batch_clusters)} clusters")
        print("="*80)
        print(f"Clusters: {batch_clusters}\n")
        
        processes = []
        
        for i, cluster in enumerate(batch_clusters):
            gpu_id = i
            
            # Create script
            script_content = train_single_cluster_script(cluster, gpu_id)
            script_file = f'train_{cluster}.py'
            
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Make executable
            os.chmod(script_file, 0o755)
            
            print(f"  Launching: {cluster} → GPU {gpu_id}")
            
            log_file = f'log_{cluster}.txt'
            
            # Use the same Python environment
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            with open(log_file, 'w') as log:
                p = subprocess.Popen(
                    [python_exe, script_file],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                processes.append((p, cluster, gpu_id, script_file, log_file))
        
        print(f"\n  Waiting for batch {batch_num}...\n")
        
        # Wait for completion
        for p, cluster, gpu_id, script_file, log_file in processes:
            return_code = p.wait()
            
            if return_code == 0:
                print(f"  ✓ {cluster} (GPU {gpu_id}) completed")
                
                # Load result
                import json
                result_file = f'result_{cluster}.json'
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        result_data = json.load(f)
                    
                    print(f"      Best: {result_data['best_model']}, "
                          f"RMSE: {result_data[result_data['best_model'].lower() + '_rmse']:.4f}")
                    
                    all_results.append({
                        'cluster': result_data['cluster'],
                        'best_model': result_data['best_model'],
                        'training_time': result_data['training_time'],
                        'n_customers': result_data['n_customers'],
                        'n_observations': result_data['n_observations'],
                        'results': {
                            'KNN': {'test_rmse': result_data['knn_rmse']},
                            'LSTM': {'test_rmse': result_data['lstm_rmse']},
                            'GRU': {'test_rmse': result_data['gru_rmse']},
                            'Transformer': {'test_rmse': result_data['transformer_rmse']}
                        }
                    })
            else:
                print(f"  ✗ {cluster} (GPU {gpu_id}) failed (code: {return_code})")
                print(f"     Check: {log_file}")
            
            # Cleanup
            if os.path.exists(script_file):
                os.remove(script_file)
        
        print(f"\n  Batch {batch_num} complete\n")
    
    # Save results
    if len(all_results) > 0:
        print("="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        from fast_training_ddp import save_results
        summary = save_results(all_results)
        
        total_time = (time.time() - overall_start) / 3600
        
        print(f"\n{'='*80}")
        print(f"✓ COMPLETE!")
        print(f"{'='*80}")
        print(f"Time: {total_time:.2f} hours")
        print(f"Trained: {len(all_results)}/{len(clusters)} clusters")
        print(f"End: {datetime.now()}")
        print(f"{'='*80}")
    else:
        print("\n✗ No results - all failed!")


if __name__ == "__main__":
    main()
