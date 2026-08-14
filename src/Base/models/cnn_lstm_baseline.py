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


class DailyCNNLSTMBaseline(nn.Module):
    """
    Hybrid temporal CNN + LSTM baseline for daily forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        target_dim: int,
        conv_channels: tuple[int, ...] = (64, 128),
        kernel_sizes: tuple[int, ...] = (3, 3),
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.15,
        activation: str = "gelu",
        head_hidden_dim: int = 128,
        use_batch_norm: bool = True,
        use_input_layer_norm: bool = True,
    ):
        super().__init__()
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one width.")
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError(
                f"conv_channels and kernel_sizes must have the same length, got "
                f"{len(conv_channels)} and {len(kernel_sizes)}."
            )
        if int(lstm_hidden_dim) <= 0:
            raise ValueError(f"lstm_hidden_dim must be positive, got {lstm_hidden_dim}")
        if int(lstm_layers) <= 0:
            raise ValueError(f"lstm_layers must be positive, got {lstm_layers}")
        if int(head_hidden_dim) <= 0:
            raise ValueError(f"head_hidden_dim must be positive, got {head_hidden_dim}")

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_dim = int(target_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.lstm_layers = int(lstm_layers)

        self.input_norm = nn.LayerNorm(self.input_dim) if use_input_layer_norm else nn.Identity()
        blocks: list[nn.Module] = []
        in_channels = self.input_dim
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            kernel_size = int(kernel_size)
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
            out_channels = int(out_channels)
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
                    build_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.temporal_cnn = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            dropout=float(dropout) if self.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        summary_dim = self.lstm_hidden_dim + in_channels * 2
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
        conv_features = self.temporal_cnn(x.transpose(1, 2)).transpose(1, 2)
        _, (hidden, _) = self.lstm(conv_features)
        lstm_summary = hidden[-1]
        avg_pool = torch.mean(conv_features, dim=1)
        max_pool = torch.amax(conv_features, dim=1)
        summary = torch.cat([lstm_summary, avg_pool, max_pool], dim=-1)
        output = self.head(summary)
        return output.view(x.size(0), self.pred_len, self.target_dim)
