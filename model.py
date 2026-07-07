"""
model.py — Bidirectional LSTM encoder with CTC output head.

Architecture
------------
  Input  : (T, N, FEATURE_DIM)
  ├─ LayerNorm on feature dim
  ├─ Linear: FEATURE_DIM → HIDDEN_SIZE
  ├─ ReLU
  ├─ Dropout
  ├─ BiLSTM: HIDDEN_SIZE  (num_layers, bidirectional)
  ├─ Linear: HIDDEN_SIZE * 2 → VOCAB_SIZE
  └─ Log-Softmax over vocab dim
  Output : (T, N, VOCAB_SIZE)   ← what torch.nn.CTCLoss expects
"""

import torch
import torch.nn as nn
from config import FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, VOCAB_SIZE


class LSTMCtcModel(nn.Module):
    def __init__(
        self,
        feature_dim = FEATURE_DIM,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        vocab_size  = VOCAB_SIZE,
        dropout     = DROPOUT,
    ):
        super().__init__()

        # ── Input normalisation & projection ──────────────────────────────
        self.norm    = nn.LayerNorm(feature_dim)
        self.proj    = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Temporal encoder ──────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = hidden_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
            batch_first   = False,   # expects (T, N, F)
        )

        # ── CTC output head ───────────────────────────────────────────────
        self.fc = nn.Linear(hidden_size * 2, vocab_size)   # *2 for bidirectional

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor  (T, N, FEATURE_DIM)   — padded batch of sequences

        Returns
        -------
        log_probs : Tensor  (T, N, VOCAB_SIZE)
        """
        x = self.norm(x)          # (T, N, F)
        x = self.proj(x)          # (T, N, hidden_size)
        x, _ = self.lstm(x)       # (T, N, hidden_size*2)
        x = self.fc(x)            # (T, N, vocab_size)
        return self.log_softmax(x)


def build_model(device="cpu"):
    model = LSTMCtcModel()
    return model.to(device)
