"""Centralized configuration for fast training
MAXIMUM GPU UTILIZATION - NO MORE 2%"""

# ============================================
# DATA CONFIGURATION
# ============================================
USE_CACHED_FEATURES = True
FEATURES_FILE = 'customer_timeseries_features.csv'

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
# TRAINING PARAMETERS - MASSIVE BATCH SIZE
# ============================================
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# CRITICAL: You have 46GB free. USE IT.
BATCH_SIZE = 8192  # Start with 8192, try 16384 or even 32768

BASE_LEARNING_RATE = 0.01  # Scale with batch size

# ============================================
# MODEL CONFIGURATIONS - MUCH BIGGER
# ============================================
# LSTM - Actually use your GPU
LSTM_CONFIGS = [
    {'hidden_size': 512, 'num_layers': 4, 'dropout': 0.3, 'lr': 0.01},
    {'hidden_size': 1024, 'num_layers': 4, 'dropout': 0.3, 'lr': 0.01},
]

# GRU - Bigger
GRU_CONFIGS = [
    {'hidden_size': 512, 'num_layers': 4, 'dropout': 0.3, 'lr': 0.01},
    {'hidden_size': 1024, 'num_layers': 4, 'dropout': 0.3, 'lr': 0.01},
]

# Transformer - Proper size
TRANSFORMER_CONFIGS = [
    {'d_model': 512, 'nhead': 8, 'num_layers': 6, 'dropout': 0.3, 'lr': 0.01},
    {'d_model': 1024, 'nhead': 16, 'num_layers': 6, 'dropout': 0.3, 'lr': 0.01},
]

KNN_CONFIGS = [
    {'n_neighbors': 3, 'weights': 'uniform', 'p': 2},
    {'n_neighbors': 5, 'weights': 'distance', 'p': 2},
    {'n_neighbors': 7, 'weights': 'distance', 'p': 2},
]

# ============================================
# GPU CONFIGURATION
# ============================================
N_GPUS = 3
GPU_MEMORY_FRACTION = 0.95

# ============================================
# DATALOADER - AGGRESSIVE SETTINGS
# ============================================
N_WORKERS = 24  # Even higher
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 8  # Prefetch more batches
NON_BLOCKING = True

# ============================================
# PERFORMANCE OPTIMIZATIONS
# ============================================
USE_AMP = True  # Mixed precision
CUDNN_BENCHMARK = True
GRADIENT_ACCUMULATION_STEPS = 1

# Compile model (PyTorch 2.0+)
USE_COMPILE = True  # torch.compile() for 2x speedup

# ============================================
# OUTPUT DIRECTORIES
# ============================================
MODELS_DIR = 'models'
MODELS_DL_DIR = 'models_dl'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

# ============================================
# LOGGING
# ============================================
VERBOSE = True
LOG_EVERY_N_EPOCHS = 5