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
        dilation: int,
        dropout: float,
        activation: str,
        use_batch_norm: bool,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        dilation = int(dilation)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")

        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class DailyCNNBaseline(nn.Module):
    """
    A compact temporal CNN baseline for daily forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        target_dim: int,
        channels: tuple[int, ...] = (64, 128, 128),
        kernel_sizes: tuple[int, ...] = (3, 3, 3),
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.15,
        activation: str = "gelu",
        head_hidden_dim: int = 128,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if not channels:
            raise ValueError("channels must contain at least one block width.")
        if len(channels) != len(kernel_sizes) or len(channels) != len(dilations):
            raise ValueError(
                f"channels, kernel_sizes, dilations must have the same length, got "
                f"{len(channels)}, {len(kernel_sizes)}, {len(dilations)}"
            )
        if int(head_hidden_dim) <= 0:
            raise ValueError(f"head_hidden_dim must be positive, got {head_hidden_dim}")

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_dim = int(target_dim)
        self.channels = tuple(int(channel) for channel in channels)

        blocks: list[nn.Module] = []
        in_channels = self.input_dim
        for out_channels, kernel_size, dilation in zip(self.channels, kernel_sizes, dilations):
            blocks.append(
                TemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=int(out_channels),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=dropout,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                )
            )
            in_channels = int(out_channels)
        self.temporal_backbone = nn.Sequential(*blocks)

        pooled_dim = in_channels * 3
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, int(head_hidden_dim)),
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

        x = x.transpose(1, 2)
        features = self.temporal_backbone(x)
        avg_pool = torch.mean(features, dim=-1)
        max_pool = torch.amax(features, dim=-1)
        last_step = features[:, :, -1]
        pooled = torch.cat([avg_pool, max_pool, last_step], dim=-1)
        output = self.head(pooled)
        return output.view(x.size(0), self.pred_len, self.target_dim)
