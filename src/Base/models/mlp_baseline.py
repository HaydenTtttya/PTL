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


class DailyMLPBaseline(nn.Module):
    """
    A simple flattened-window MLP baseline for daily forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        target_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")
        if any(int(hidden_dim) <= 0 for hidden_dim in hidden_dims):
            raise ValueError(f"hidden_dims must be positive, got {hidden_dims}")

        self.input_dim = int(input_dim)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_dim = int(target_dim)
        self.hidden_dims = tuple(int(hidden_dim) for hidden_dim in hidden_dims)

        flat_input_dim = self.seq_len * self.input_dim
        flat_output_dim = self.pred_len * self.target_dim

        self.input_norm = nn.LayerNorm(flat_input_dim) if use_layer_norm else nn.Identity()

        layers: list[nn.Module] = []
        in_dim = flat_input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(build_activation(activation))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, flat_output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, input_dim], got shape {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        batch_size = x.size(0)
        flattened = x.reshape(batch_size, -1)
        flattened = self.input_norm(flattened)
        output = self.network(flattened)
        return output.view(batch_size, self.pred_len, self.target_dim)
