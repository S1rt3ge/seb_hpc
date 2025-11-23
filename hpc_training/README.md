# Fast SME Cash Flow Forecasting

Optimized for HPC with 3 GPUs, target training time < 3 hours.

## Quick Start

### 1. Upload Files to HPC
```bash
scp *.py your_username@hpc_address:~/sme_cashflow/
scp *.csv your_username@hpc_address:~/sme_cashflow/
scp train.pbs your_username@hpc_address:~/sme_cashflow/
```

### 2. Test Before Full Run
```bash
ssh your_username@hpc_address
cd ~/sme_cashflow
module load AI/pytorch-1.13.1-gpu-conda
python quick_test.py
```

### 3. Submit Job
```bash
qsub train.pbs
```

### 4. Monitor Progress
```bash
qstat -u your_username
tail -f SME_CashFlow.o*
```

## Models Trained

1. **KNN** (from paper) - 3 configs
2. **LSTM** - 2 configs
3. **GRU** - 2 configs
4. **Transformer** - 2 configs

Total: 9 model configurations per cluster

## Optimizations for Speed

- Weekly aggregation (vs daily)
- Minimal features (4 core features)
- Lightweight models (32-64 hidden units)
- Mixed precision training (FP16)
- Larger batch sizes (64)
- Early stopping (patience=10)
- Max 50 epochs
- Parallel GPU utilization
- Cached features

## Expected Performance

- Time per cluster: ~15-20 minutes
- Total time (8 clusters): **2-3 hours**
- GPU memory: ~6GB per model
- KNN baseline RMSE: 0.11-0.15 (from paper)
- DL models RMSE: 0.09-0.14

## Output Files

```
models_dl/
├── <cluster>_KNN.pkl
├── <cluster>_LSTM.pt
├── <cluster>_GRU.pt
├── <cluster>_Transformer.pt
└── training_summary.csv
```

## Configuration

Edit `config.py` to adjust:
- `MAX_CUSTOMERS_PER_CLUSTER` - Set to 50 for quick test
- `MAX_EPOCHS` - Reduce for faster training
- `N_GPUS` - Set to your available GPUs
- `BATCH_SIZE` - Increase if you have more memory

## Troubleshooting

### Out of Memory
```python
# In config.py
BATCH_SIZE = 32  # Reduce from 64
```

### Too Slow
```python
# In config.py
MAX_CUSTOMERS_PER_CLUSTER = 50
MAX_EPOCHS = 30
```

### Module Not Found
```bash
pip install --user -r requirements.txt
```
