"""
Configuration - will generate features from transactions first time
"""

# Feature configuration
USE_CACHED_FEATURES = False  # Set to False for first run
FEATURES_FILE = 'customer_timeseries_features.csv'  # Generated features

# Time-series configuration
SEQUENCE_LENGTH = 3  # Use 3 time steps for LSTM/GRU
MIN_OBSERVATIONS = 20  # Minimum weeks of data per customer

# Rest of config unchanged...
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 64

# Model configs...
LSTM_CONFIGS = [
    {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1, 'lr': 0.001},
    {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001},
]

GRU_CONFIGS = [
    {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1, 'lr': 0.001},
    {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001},
]

TRANSFORMER_CONFIGS = [
    {'d_model': 32, 'nhead': 2, 'num_layers': 1, 'dropout': 0.1, 'lr': 0.001},
    {'d_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001},
]

KNN_CONFIGS = [
    {'n_neighbors': 3, 'weights': 'uniform', 'p': 2},
    {'n_neighbors': 5, 'weights': 'distance', 'p': 2},
    {'n_neighbors': 7, 'weights': 'distance', 'p': 2},
]

N_GPUS = 4
N_WORKERS = 8

MODELS_DL_DIR = 'models_dl'