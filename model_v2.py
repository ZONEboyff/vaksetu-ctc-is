"""
model_v2.py — Improved sign language model: Conv1D + BiLSTM + Multi-head Attention.

Architecture
------------
  Input  : (T, N, FEATURE_DIM)
  ├── LayerNorm(FEATURE_DIM)
  ├── Conv1D block (local temporal patterns):
  │   ├── Conv1D(288 → 192, k=3, pad=same) + GELU + Dropout
  │   └── Conv1D(192 → 192, k=3, pad=same) + GELU + Dropout
  ├── BiLSTM(192, 3 layers, dropout=0.4)  → 384-dim
  ├── Multi-head self-attention (4 heads, 384 dim)
  ├── LayerNorm(384)
  ├── Dropout(0.3)
  ├── Linear(384 → VOCAB_SIZE)
  └── LogSoftmax
  Output : (T, N, VOCAB_SIZE)
"""

import torch
import torch.nn as nn
from config import FEATURE_DIM, VOCAB_SIZE

# ── V2 hyperparameters ───────────────────────────────────────────────────────
V2_CONV_DIM    = 192
V2_HIDDEN_SIZE = 192     # per direction → BiLSTM output = 384
V2_NUM_LAYERS  = 3
V2_DROPOUT     = 0.4
V2_N_HEADS     = 4
V2_ATTN_DIM    = V2_HIDDEN_SIZE * 2   # 384 (BiLSTM output)


class Conv1DBlock(nn.Module):
    """Two-layer 1D convolution for capturing local temporal patterns."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.3):
        super().__init__()
        padding = kernel_size // 2   # same padding
        self.block = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            # Layer 2
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (T, N, C) → Conv1d wants (N, C, T)"""
        x = x.permute(1, 2, 0)    # (N, C, T)
        x = self.block(x)          # (N, out_channels, T)
        x = x.permute(2, 0, 1)    # (T, N, out_channels)
        return x


class LSTMCtcModelV2(nn.Module):
    def __init__(
        self,
        feature_dim = FEATURE_DIM,
        conv_dim    = V2_CONV_DIM,
        hidden_size = V2_HIDDEN_SIZE,
        num_layers  = V2_NUM_LAYERS,
        vocab_size  = VOCAB_SIZE,
        dropout     = V2_DROPOUT,
        n_heads     = V2_N_HEADS,
    ):
        super().__init__()

        # ── Input normalisation ──────────────────────────────────────────
        self.input_norm = nn.LayerNorm(feature_dim)

        # ── 1D Conv front-end ────────────────────────────────────────────
        self.conv = Conv1DBlock(feature_dim, conv_dim, kernel_size=3, dropout=dropout)

        # ── Temporal encoder (BiLSTM) ────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size    = conv_dim,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
            batch_first   = False,
        )

        lstm_out_dim = hidden_size * 2   # bidirectional

        # ── Multi-head self-attention ────────────────────────────────────
        self.attn = nn.MultiheadAttention(
            embed_dim  = lstm_out_dim,
            num_heads  = n_heads,
            dropout    = dropout,
            batch_first = False,   # expects (T, N, E)
        )
        self.attn_norm = nn.LayerNorm(lstm_out_dim)
        self.attn_drop = nn.Dropout(dropout * 0.75)   # lighter dropout after attention

        # ── CTC output head ──────────────────────────────────────────────
        self.fc          = nn.Linear(lstm_out_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (T, N, FEATURE_DIM)

        Returns
        -------
        log_probs : Tensor (T, N, VOCAB_SIZE)
        """
        # Input norm
        x = self.input_norm(x)          # (T, N, F)

        # Conv front-end
        x = self.conv(x)                # (T, N, conv_dim)

        # BiLSTM
        x, _ = self.lstm(x)             # (T, N, hidden*2)

        # Multi-head self-attention with residual connection
        attn_out, _ = self.attn(x, x, x)   # (T, N, hidden*2)
        x = x + self.attn_drop(attn_out)   # residual
        x = self.attn_norm(x)              # layer norm

        # Output
        x = self.fc(x)                 # (T, N, vocab_size)
        return self.log_softmax(x)


def build_model_v2(device="cpu"):
    model = LSTMCtcModelV2()
    return model.to(device)
