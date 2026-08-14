from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class BasicTransformerBaseline(nn.Module):
    """
    A plain target-only Transformer encoder baseline.

    This intentionally avoids the PTL-specific WaterQualityTransformer design:
    no temporal adapter, no reversible/instance normalization, no feature-token
    encoder, no pretraining hooks, and no progressive-stage state transfer.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        target_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_dim = int(target_dim)

        self.input_projection = nn.Linear(self.input_dim, int(d_model))
        self.position_encoding = SinusoidalPositionalEncoding(
            d_model=int(d_model),
            max_len=self.seq_len,
            dropout=float(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_layers),
        )
        self.output_head = nn.Linear(int(d_model), self.pred_len * self.target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, input_dim], got shape {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        tokens = self.input_projection(x)
        tokens = self.position_encoding(tokens)
        encoded = self.encoder(tokens)
        output = self.output_head(encoded[:, -1])
        return output.view(x.size(0), self.pred_len, self.target_dim)
