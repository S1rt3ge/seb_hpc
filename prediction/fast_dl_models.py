"""
Lightweight deep learning models optimized for speed
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ============================================
# DATASET
# ============================================

class FastTimeSeriesDataset(Dataset):
    """Fast dataset with pre-computed sequences"""

    def __init__(self, X, y, sequence_length=5):
        self.sequence_length = sequence_length

        # Pre-compute all sequences
        sequences = []
        targets = []

        for i in range(len(X) - sequence_length):
            sequences.append(X[i:i+sequence_length])
            targets.append(y[i+sequence_length])

        self.X = torch.FloatTensor(np.array(sequences))
        self.y = torch.FloatTensor(np.array(targets)).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================
# LIGHTWEIGHT MODELS
# ============================================

class FastLSTM(nn.Module):
    """Lightweight LSTM"""

    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class FastGRU(nn.Module):
    """Lightweight GRU"""

    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.1):
        super().__init__()

        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class FastTransformer(nn.Module):
    """Lightweight Transformer"""

    def __init__(self, input_size, d_model=32, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model*2, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x


# ============================================
# FAST TRAINER
# ============================================

class FastTrainer:
    """Optimized trainer with mixed precision"""

    def __init__(self, model, device, lr=0.001, patience=10):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.patience = patience

        # Mixed precision for faster training
        self.scaler = torch.cuda.amp.GradScaler() if device != 'cpu' else None

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:  # Use mixed precision
                with torch.cuda.amp.autocast():
                    pred = self.model(X)
                    loss = self.criterion(pred, y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(X)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, y)
                total_loss += loss.item()

        return total_loss / len(loader)

    def fit(self, train_loader, val_loader, epochs=50):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        self.model.load_state_dict(best_state)
        return np.sqrt(best_val_loss)

    def evaluate(self, loader):
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                pred = self.model(X)
                predictions.extend(pred.cpu().numpy().flatten())
                actuals.extend(y.numpy().flatten())

        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        return rmse


def create_model(model_type, input_size, **kwargs):
    """Model factory"""

    models = {
        'lstm': FastLSTM,
        'gru': FastGRU,
        'transformer': FastTransformer
    }

    # Map hidden_size to d_model for transformer
    if model_type.lower() == 'transformer' and 'hidden_size' in kwargs:
        kwargs['d_model'] = kwargs.pop('hidden_size')

    return models[model_type.lower()](input_size, **kwargs)
