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


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        activation: str,
        use_batch_norm: bool,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        self.conv = nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.norm = nn.BatchNorm1d(int(out_channels)) if use_batch_norm else nn.Identity()
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(float(dropout))
        self.residual = (
            nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)
            if int(in_channels) != int(out_channels)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class TemporalAdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, dropout: float):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(int(input_dim), int(attention_dim)),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(attention_dim), 1, bias=False),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        scores = self.score(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(sequence * weights, dim=1)


class DailyCLABaseline(nn.Module):
    """
    CNN-LSTM-Attention baseline for daily forecasting.
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
        lstm_num_layers: int = 1,
        attention_dim: int = 64,
        head_hidden_dim: int = 128,
        dropout: float = 0.15,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        use_input_layer_norm: bool = True,
    ):
        super().__init__()
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one channel width.")
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError(
                f"conv_channels and kernel_sizes must have the same length, got "
                f"{len(conv_channels)} and {len(kernel_sizes)}"
            )
        if int(lstm_hidden_dim) <= 0:
            raise ValueError(f"lstm_hidden_dim must be positive, got {lstm_hidden_dim}")
        if int(lstm_num_layers) <= 0:
            raise ValueError(f"lstm_num_layers must be positive, got {lstm_num_layers}")

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_dim = int(target_dim)
        self.conv_channels = tuple(int(value) for value in conv_channels)
        self.kernel_sizes = tuple(int(value) for value in kernel_sizes)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.lstm_num_layers = int(lstm_num_layers)

        self.input_norm = nn.LayerNorm(self.input_dim) if use_input_layer_norm else nn.Identity()
        blocks: list[nn.Module] = []
        in_channels = self.input_dim
        for out_channels, kernel_size in zip(self.conv_channels, self.kernel_sizes):
            blocks.append(
                TemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
            )
            in_channels = int(out_channels)
        self.conv_backbone = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            dropout=float(dropout) if self.lstm_num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = TemporalAdditiveAttention(
            input_dim=self.lstm_hidden_dim,
            attention_dim=int(attention_dim),
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.lstm_hidden_dim),
            nn.Linear(self.lstm_hidden_dim, int(head_hidden_dim)),
            build_activation(activation),
            nn.Dropout(float(dropout)),
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
        conv_input = x.transpose(1, 2)
        conv_features = self.conv_backbone(conv_input).transpose(1, 2)
        lstm_sequence, _ = self.lstm(conv_features)
        context = self.attention(lstm_sequence)
        output = self.head(context)
        return output.view(x.size(0), self.pred_len, self.target_dim)
