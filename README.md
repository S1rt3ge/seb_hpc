# SME Cash Flow Forecasting Platform

End-to-end ML application for short-horizon SME cash flow forecasting, liquidity monitoring, and treasury-style decision support.

This project combines a FastAPI web application, an offline prediction layer, a custom feature-engineering pipeline, and HPC-oriented multi-model training scripts. The result is a working demo of how raw transaction history can be transformed into weekly forecasts and practical actions such as maintaining liquidity buffers or identifying investable surplus.

## Why this project matters

Small and medium-sized businesses often have fragmented visibility into near-term cash position. This repository explores a pragmatic forecasting workflow:

- ingest transaction-level history
- derive behavioral and time-series features from raw banking activity
- classify customers into behavioral segments
- select the best forecasting model per segment
- convert forecasts into actionable financial recommendations

From a portfolio perspective, this project is valuable because it is not just a notebook or a model dump. It includes:

- an actual web application
- reusable prediction code
- persisted model artifacts
- feature engineering from raw inputs
- training automation for multi-GPU / HPC environments
- lightweight API endpoints for integration

## What it does

At a high level, the system:

1. loads historical transaction data from SQLite
2. computes static and weekly time-series features for a selected customer
3. assigns that customer to a business behavior cluster
4. loads the best pre-trained model for that cluster
5. predicts short-horizon weekly cash flow change
6. reconstructs future balances and derives treasury recommendations

The web UI exposes this flow through a simple login-to-dashboard experience, while the API and prediction modules can also be used programmatically.

## Core capabilities

- Weekly cash flow forecasting from raw transaction history
- Customer behavioral segmentation based on transaction patterns
- Per-cluster model selection across classical ML and deep learning models
- Real-time feature generation for customers without precomputed features
- Liquidity interpretation layer with invest / maintain / warning recommendations
- HPC-ready training scripts for scaling experiments across multiple GPUs

## Architecture

```text
Raw transaction data (SQLite / CSV)
        |
        v
Feature engineering
- static behavioral features
- weekly aggregation
- lag features
- rolling statistics
        |
        v
Cluster assignment
- Subscription_SaaS
- B2B_Services_Combined
- Retail_Combined
- Digital_B2C_Combined
- Mixed
        |
        v
Model selection per cluster
- KNN
- LSTM
- GRU
- Transformer
        |
        v
Prediction layer
- standardized forecast output
- conversion back to EUR
- cumulative future balance reconstruction
        |
        v
Application layer
- FastAPI endpoints
- HTML dashboard
- treasury recommendation logic
```

## Repository structure

```text
app/
  main.py                     FastAPI app, routes, dashboard rendering
  complete_feature_engine.py  Real-time feature engineering from raw transactions
  scaler_params.json          Saved scaling metadata for inference

prediction/
  predict_cashflow.py         Offline predictor / model loading logic
  fast_dl_models.py           Lightweight LSTM, GRU, Transformer implementations
  config.py                   Training and inference configuration
  models_dl_optimized/        Saved model artifacts and training summary

hpc_training/
  fast_feature_engineering.py Batch feature generation for training
  fast_training.py            Model training pipeline
  fast_training_ddp.py        Parallelized training utilities
  train_final.py              Multi-GPU orchestration script
  train.sh                    PBS/HPC job script example

db/
  transactions.db             SQLite source used by the demo app
  *.csv                       Feature and customer data artifacts

templates/
  *.html                      Login and dashboard templates

static/
  css/                        Static assets
```

## Tech stack

### Application

- Python
- FastAPI
- Uvicorn
- Jinja2
- SQLite
- Pandas

### Machine learning

- scikit-learn
- PyTorch
- NumPy
- joblib

### Training / infrastructure

- Multi-GPU training scripts
- PBS job submission for HPC clusters
- Serialized model artifacts for reproducible inference

## Feature engineering design

The project does more than feed raw balances into a model. It constructs a combined behavioral + temporal representation.

### Static behavioral features

The app computes customer-level descriptors such as:

- concentration among top counterparties
- number of unique counterparties
- small vs large transaction share
- transaction size coefficient of variation
- median transaction size
- transaction volatility
- share of POS activity
- share of internet banking activity
- share of negative cash flow days
- transaction frequency

These signals are used to infer a cluster before forecasting.

### Time-series features

For each customer, the pipeline:

- converts credit/debit activity into signed cash flow
- aggregates activity at weekly frequency
- computes weekly change in cash flow as the prediction target
- builds lag features for growth, cash flow, and transaction count
- adds rolling mean and rolling standard deviation features

This produces a compact weekly forecasting dataset that can be consumed by both tabular and sequence models.

## Modeling approach

The repository compares several model families per customer cluster:

- `KNN` for strong tabular, low-complexity baselines
- `LSTM` for sequential dependency modeling
- `GRU` as a lighter recurrent alternative
- `Transformer` for sequence encoding with attention-based layers

At inference time, the predictor loads the metadata for a cluster and routes the request to the best saved model for that segment.

## Current results

The checked-in model artifacts include a training summary for five behavioral clusters.

| Cluster | Best model | Test RMSE |
| --- | --- | ---: |
| Retail_Combined | KNN | 0.1124 |
| Mixed | KNN | 0.2631 |
| B2B_Services_Combined | KNN | 0.3114 |
| Digital_B2C_Combined | KNN | 0.9333 |
| Subscription_SaaS | KNN | 1.4270 |

Important observation: on the currently persisted artifacts, `KNN` outperforms the deep learning models across all five clusters. That is a useful engineering result in itself: it shows the project evaluates model choice pragmatically instead of assuming that more complex architectures automatically win.

## Application flow

The FastAPI app implements a simple business workflow:

1. list customers with sufficient transaction history
2. select a customer in the login screen
3. generate features directly from raw transactions
4. run the best cluster-specific model
5. display weekly forecast changes, projected balances, and a recommendation

Recommendation logic is intentionally straightforward and interpretable:

- if projected surplus exceeds a threshold, suggest moving funds to savings
- if the current position is negative, raise a financing warning
- otherwise recommend maintaining the current position

## API surface

The app exposes lightweight endpoints for integration and demo usage:

- `GET /` - HTML login page
- `POST /login` - redirects to the dashboard for a selected customer
- `GET /dashboard?customer_id=<id>` - renders forecast dashboard
- `GET /api/customers` - returns eligible customers
- `GET /api/predict/{customer_id}?weeks=4` - returns forecast payload as JSON

## Running locally

### 1. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
pip install -r prediction/requirements.txt
```

### 2. Start the web app

Run the application from the `app/` directory because the current code uses relative paths for templates, static files, model artifacts, and scaler metadata.

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Open `http://localhost:8080`.

## Running predictions programmatically

The prediction module can be used independently of the web app.

```python
from predict_cashflow import CashFlowPredictor

predictor = CashFlowPredictor(models_dir='prediction/models_dl_optimized')
```

The full app path is more convenient when you want features generated directly from transaction history, because `CompleteFeatureEngine` handles feature creation for unseen customers at runtime.

## Training pipeline

The repository includes a separate training track focused on throughput and reproducibility.

### Training characteristics

- weekly feature generation from transaction data
- support for multiple customer clusters
- comparison of KNN and multiple sequence models
- multi-GPU orchestration scripts
- HPC job script example using PBS

### Relevant files

- `hpc_training/fast_feature_engineering.py`
- `hpc_training/fast_training.py`
- `hpc_training/fast_training_ddp.py`
- `hpc_training/train_final.py`
- `hpc_training/train.sh`

### HPC notes

The checked-in scripts assume an environment with:

- CUDA-capable GPUs
- PyTorch available through environment modules or local installation
- scheduler-based execution via PBS-style job scripts

This part of the repository demonstrates systems thinking beyond model development: resource-aware training, artifact persistence, and operational scripting.

## Engineering highlights

The project showcases several decisions that are good signals in a portfolio:

- separation between serving code, prediction code, and training code
- runtime feature engineering rather than relying only on precomputed offline tables
- segment-specific model selection instead of one global model for heterogeneous customers
- preservation of scaler parameters for inference-time consistency
- direct conversion from model output to business-facing recommendations
- reproducible saved artifacts for deployment-style usage

## Public-safe notes

This repository should be presented as a portfolio/demo project for SME cash flow forecasting.

- Do not position it as a production banking system.
- Do not imply access to confidential customer data.
- Treat all included datasets and local databases as demo, synthetic, anonymized, or development artifacts unless you explicitly control and can verify their provenance.
- Brand-related template assets should be understood as UI mock/demo material, not proof of commercial deployment.

If you use this repository in job applications, the strongest framing is:

> Built an end-to-end cash flow forecasting demo for SME treasury use cases, including real-time feature engineering, per-segment model selection, FastAPI serving, and HPC-oriented training workflows.

## Limitations

This is a strong engineering demo, but it is not a finished product. A production-grade next iteration would likely include:

- proper environment-based configuration instead of hardcoded local fallbacks
- containerized deployment
- automated tests for feature engineering and API behavior
- authentication and authorization
- monitoring, model versioning, and observability
- stricter data contracts and schema validation

## What I would improve next

If I continued this project, I would prioritize:

1. removing hardcoded filesystem paths and centralizing config
2. adding a reproducible training-to-serving artifact pipeline
3. introducing tests around cluster assignment and prediction payload integrity
4. adding Docker support and a cleaner local developer setup
5. instrumenting inference latency and forecast quality monitoring

## License

MIT License. See `LICENSE`.
