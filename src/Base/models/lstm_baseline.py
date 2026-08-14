from __future__ import annotations

import torch
import torch.nn as nn


def build_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class DailyLSTMBaseline(nn.Module):
    """
    A compact LSTM baseline for daily forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        target_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
        activation: str = "gelu",
        head_hidden_dim: int = 128,
        bidirectional: bool = False,
        use_input_layer_norm: bool = True,
    ):
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if int(num_layers) <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if int(head_hidden_dim) <= 0:
            raise ValueError(f"head_hidden_dim must be positive, got {head_hidden_dim}")

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.direction_count = 2 if self.bidirectional else 1

        self.input_norm = nn.LayerNorm(self.input_dim) if use_input_layer_norm else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        summary_dim = self.hidden_dim * self.direction_count
        self.head = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, int(head_hidden_dim)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(int(head_hidden_dim), self.pred_len * self.target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, input_dim], got shape {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        x = self.input_norm(x)
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            summary = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            summary = hidden[-1]
        output = self.head(summary)
        return output.view(x.size(0), self.pred_len, self.target_dim)


class DailyBiLSTMBaseline(DailyLSTMBaseline):
    """
    Explicit Bi-LSTM baseline for daily forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        target_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
        activation: str = "gelu",
        head_hidden_dim: int = 128,
        use_input_layer_norm: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            target_dim=target_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            head_hidden_dim=head_hidden_dim,
            bidirectional=True,
            use_input_layer_norm=use_input_layer_norm,
        )
