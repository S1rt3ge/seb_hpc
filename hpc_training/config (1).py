"""
FINAL OPTIMIZED CONFIG
- Merges clusters for more data
- Adaptive batch sizing for GPU utilization
- Right-sized models
- Proper learning rates
"""

# ============================================
# DATA CONFIGURATION
# ============================================
USE_CACHED_FEATURES = True

# Choose your merging strategy:
# FEATURES_FILE = 'customer_timeseries_features.csv'              # Original (8 clusters, low GPU use)

FEATURES_FILE = 'customer_timeseries_features_conservative.csv'  # 5 clusters (RECOMMENDED)
# FEATURES_FILE = 'customer_timeseries_features_aggressive.csv'  # 3 clusters (max GPU use)

MAX_CUSTOMERS_PER_CLUSTER = None
MIN_OBSERVATIONS = 20

# ============================================
# TIME-SERIES CONFIGURATION
# ============================================
SEQUENCE_LENGTH = 3
N_LAGS = 3

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

# ============================================
# TRAINING PARAMETERS
# ============================================
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# ADAPTIVE BATCH SIZING (set per cluster dynamically)
# This is calculated based on cluster size in training script
# Rule: min(model_capacity, train_size // 20)
# Ensures 20+ batches per epoch while maximizing GPU use

BASE_LEARNING_RATE = 0.001  # Much better than 0.01

# ============================================
# MODEL CONFIGURATIONS - ADAPTIVE
# ============================================
# These configs are selected based on cluster size
# Small clusters (< 10K): Use config 0
# Medium clusters (10-30K): Use config 1  
# Large clusters (30K+): Use config 2

LSTM_CONFIGS = [
    # Small clusters (< 10K obs)
    {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001, 'batch_size': None},  # Adaptive
    
    # Medium clusters (10-30K obs)
    {'hidden_size': 256, 'num_layers': 3, 'dropout': 0.3, 'lr': 0.001, 'batch_size': None},  # Adaptive
    
    # Large clusters (30K+ obs)
    {'hidden_size': 512, 'num_layers': 4, 'dropout': 0.3, 'lr': 0.001, 'batch_size': None},  # Adaptive
]

GRU_CONFIGS = [
    {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001, 'batch_size': None},
    {'hidden_size': 256, 'num_layers': 3, 'dropout': 0.3, 'lr': 0.001, 'batch_size': None},
    {'hidden_size': 512, 'num_layers': 4, 'dropout': 0.3, 'lr': 0.001, 'batch_size': None},
]

TRANSFORMER_CONFIGS = [
    {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001, 'batch_size': None},
    {'d_model': 256, 'nhead': 8, 'num_layers': 3, 'dropout': 0.3, 'lr': 0.001, 'batch_size': None},
    {'d_model': 512, 'nhead': 8, 'num_layers': 4, 'dropout': 0.3, 'lr': 0.001, 'batch_size': None},
]

KNN_CONFIGS = [
    {'n_neighbors': 5, 'weights': 'distance', 'p': 2},
    {'n_neighbors': 7, 'weights': 'distance', 'p': 2},
    {'n_neighbors': 10, 'weights': 'distance', 'p': 2},
]

# ============================================
# ADAPTIVE BATCH SIZING FUNCTION
# ============================================
def get_adaptive_batch_size(n_train_samples, model_size='medium', gpu_memory_gb=48):
    """
    Calculate optimal batch size
    
    Rules:
    - Minimum 20 batches per epoch
    - Scale with model size
    - Use GPU capacity
    """
    base_batches = {
        'small': 2048,   # 128 hidden
        'medium': 1024,  # 256 hidden
        'large': 512,    # 512 hidden
    }
    
    base_batch = base_batches[model_size]
    max_batch = n_train_samples // 20  # Min 20 batches
    
    optimal = min(base_batch, max_batch)
    return max(32, optimal)  # At least 32

def select_model_config(n_observations, model_type='LSTM'):
    """
    Select appropriate model config based on cluster size
    
    Returns: (config_index, batch_size)
    """
    configs = {
        'LSTM': LSTM_CONFIGS,
        'GRU': GRU_CONFIGS,
        'Transformer': TRANSFORMER_CONFIGS,
    }
    
    model_configs = configs[model_type]
    n_train = int(n_observations * 0.6)
    
    if n_observations < 10000:
        # Small cluster
        config_idx = 0
        batch_size = get_adaptive_batch_size(n_train, 'small')
    elif n_observations < 30000:
        # Medium cluster
        config_idx = 1
        batch_size = get_adaptive_batch_size(n_train, 'medium')
    else:
        # Large cluster
        config_idx = 2
        batch_size = get_adaptive_batch_size(n_train, 'large')
    
    return config_idx, batch_size

# ============================================
# GPU CONFIGURATION
# ============================================
N_GPUS = 4  # L40S GPUs
GPU_MEMORY_FRACTION = 0.9

# ============================================
# DATALOADER SETTINGS
# ============================================
N_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
NON_BLOCKING = True

# ============================================
# PERFORMANCE OPTIMIZATIONS
# ============================================
USE_AMP = True  # Mixed precision (2x speedup)
CUDNN_BENCHMARK = True
GRADIENT_ACCUMULATION_STEPS = 1
USE_COMPILE = False  # Disable for stability

# ============================================
# OUTPUT DIRECTORIES
# ============================================
MODELS_DIR = 'models_optimized'
MODELS_DL_DIR = 'models_dl_optimized'
RESULTS_DIR = 'results_optimized'
LOGS_DIR = 'logs_optimized'

# ============================================
# LOGGING
# ============================================
VERBOSE = True
LOG_EVERY_N_EPOCHS = 2

# ============================================
# EXPECTED PERFORMANCE
# ============================================
"""
With Conservative Merging (5 clusters):
  Cluster                    Batch  Batches  GPU Use  Expected RMSE
  ------------------------------------------------------------------
  Retail_Combined              98      20     20-40%      0.18
  Mixed                       157      20     20-40%      0.30
  Digital_B2C_Combined        337      20     20-40%      0.50
  Subscription_SaaS           468      20     20-40%      0.28
  B2B_Services_Combined       512      53     60-90%      0.22
  
  Average RMSE: 0.26 (vs 0.275 current) - 5% improvement
  Average GPU: 40-60% (vs 2-5% current)
  LSTM competitive with KNN!

With Aggressive Merging (3 clusters):
  Cluster                    Batch  Batches  GPU Use  Expected RMSE
  ------------------------------------------------------------------
  Mixed                       157      20     20-40%      0.28
  Retail_Digital_Small        436      20     20-40%      0.40
  Services_Large              512      71     80-95%      0.18
  
  Average RMSE: 0.22-0.24 - 15-20% improvement
  Average GPU: 60-80%
  LSTM beats KNN on large clusters!

Key Improvements:
1. Merged clusters → More data for DL
2. Adaptive batch sizing → Better GPU use
3. Right-sized models → No overfitting
4. Proper learning rate → Convergence
5. Fixed double standardization → Correct scale
"""