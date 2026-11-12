"""Modèles de prévision : LSTM, GRU, Transformer (PyTorch)."""
import torch
import torch.nn as nn


class RNNForecaster(nn.Module):
    def __init__(self, n_features, horizon, hidden=64, layers=2, dropout=0.1, cell="lstm"):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_features, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x):                       # x : [B,K,F]
        out, _ = self.rnn(x)
        return self.head(out[:, -1])            # dernier pas de temps -> [B,H]


class TransformerForecaster(nn.Module):
    def __init__(self, n_features, horizon, d_model=64, nhead=4, layers=2, ff=128, dropout=0.1, max_len=64):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)   # position apprise
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, horizon))

    def forward(self, x):
        h = self.inp(x) + self.pos[:, : x.size(1)]
        h = self.encoder(h)
        return self.head(h[:, -1])


def build_model(name, n_features, horizon, hidden=64, layers=2, dropout=0.1):
    if name in ("lstm", "gru"):
        return RNNForecaster(n_features, horizon, hidden, layers, dropout, cell=name)
    if name == "transformer":
        return TransformerForecaster(n_features, horizon, d_model=hidden, nhead=4, layers=layers, ff=2 * hidden, dropout=dropout)
    raise ValueError(name)
