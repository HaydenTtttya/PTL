from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Base.models.cnn_baseline import DailyCNNBaseline  # noqa: E402
from Base.models.cnn_lstm_baseline import DailyCNNLSTMBaseline  # noqa: E402
from Base.models.lstm_baseline import (  # noqa: E402
    DailyBiLSTMBaseline,
    DailyLSTMBaseline,
)
from Base.models.mlp_baseline import DailyMLPBaseline  # noqa: E402


SUPPORTED_MODEL_AGNOSTIC_BACKBONES = (
    "mlp",
    "cnn",
    "lstm",
    "bilstm",
    "cnn_lstm",
)
SUPPORTED_MODEL_AGNOSTIC_INTERFACES = (
    "legacy",
    "feature_token_v1",
    "feature_token_residual_v2",
)


def normalize_backbone_name(name: str | None) -> str:
    normalized = str(name or "transformer").strip().lower().replace("-", "_")
    aliases = {
        "water_quality_transformer": "transformer",
        "wqt": "transformer",
        "conv": "cnn",
        "rnn": "lstm",
    }
    return aliases.get(normalized, normalized)


def normalize_model_agnostic_interface(name: str | None) -> str:
    normalized = str(name or "legacy").strip().lower().replace("-", "_")
    aliases = {
        "feature_token": "feature_token_v1",
        "unified_feature_token": "feature_token_v1",
        "feature_token_residual": "feature_token_residual_v2",
        "unified_feature_token_residual": "feature_token_residual_v2",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_MODEL_AGNOSTIC_INTERFACES:
        raise ValueError(
            f"Unsupported model-agnostic interface: {name}. "
            f"Expected one of {SUPPORTED_MODEL_AGNOSTIC_INTERFACES}."
        )
    return normalized


def _config_value(config, name, default):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def build_temporal_forecasting_backbone(
    backbone_name: str,
    input_dim: int,
    seq_len: int,
    pred_len: int,
    target_dim: int,
    config,
) -> nn.Module:
    backbone_name = normalize_backbone_name(backbone_name)
    if backbone_name == "cnn":
        return DailyCNNBaseline(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            target_dim=target_dim,
            channels=tuple(_config_value(config, "cnn_channels", (64, 128, 128))),
            kernel_sizes=tuple(_config_value(config, "cnn_kernel_sizes", (3, 3, 3))),
            dilations=tuple(_config_value(config, "cnn_dilations", (1, 2, 4))),
            dropout=float(_config_value(config, "backbone_dropout", 0.15)),
            activation=str(_config_value(config, "backbone_activation", "gelu")),
            head_hidden_dim=int(_config_value(config, "backbone_head_hidden_dim", 128)),
            use_batch_norm=bool(_config_value(config, "cnn_use_batch_norm", True)),
        )
    if backbone_name == "lstm":
        return DailyLSTMBaseline(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            target_dim=target_dim,
            hidden_dim=int(_config_value(config, "lstm_hidden_dim", 128)),
            num_layers=int(_config_value(config, "lstm_num_layers", 2)),
            dropout=float(_config_value(config, "backbone_dropout", 0.15)),
            activation=str(_config_value(config, "backbone_activation", "gelu")),
            head_hidden_dim=int(_config_value(config, "backbone_head_hidden_dim", 128)),
            bidirectional=False,
            use_input_layer_norm=bool(
                _config_value(config, "lstm_use_input_layer_norm", True)
            ),
        )
    if backbone_name == "bilstm":
        return DailyBiLSTMBaseline(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            target_dim=target_dim,
            hidden_dim=int(_config_value(config, "lstm_hidden_dim", 128)),
            num_layers=int(_config_value(config, "lstm_num_layers", 2)),
            dropout=float(_config_value(config, "backbone_dropout", 0.15)),
            activation=str(_config_value(config, "backbone_activation", "gelu")),
            head_hidden_dim=int(_config_value(config, "backbone_head_hidden_dim", 128)),
            use_input_layer_norm=bool(
                _config_value(config, "lstm_use_input_layer_norm", True)
            ),
        )
    if backbone_name == "cnn_lstm":
        return DailyCNNLSTMBaseline(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            target_dim=target_dim,
            conv_channels=tuple(
                _config_value(config, "cnn_lstm_conv_channels", (64, 128))
            ),
            kernel_sizes=tuple(
                _config_value(config, "cnn_lstm_kernel_sizes", (3, 3))
            ),
            lstm_hidden_dim=int(
                _config_value(config, "cnn_lstm_hidden_dim", 128)
            ),
            lstm_layers=int(_config_value(config, "cnn_lstm_layers", 1)),
            dropout=float(_config_value(config, "backbone_dropout", 0.15)),
            activation=str(_config_value(config, "backbone_activation", "gelu")),
            head_hidden_dim=int(_config_value(config, "backbone_head_hidden_dim", 128)),
            use_batch_norm=bool(
                _config_value(config, "cnn_lstm_use_batch_norm", True)
            ),
            use_input_layer_norm=bool(
                _config_value(config, "cnn_lstm_use_input_layer_norm", True)
            ),
        )
    if backbone_name == "mlp":
        return DailyMLPBaseline(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            target_dim=target_dim,
            hidden_dims=tuple(_config_value(config, "mlp_hidden_dims", (256, 128))),
            dropout=float(_config_value(config, "mlp_dropout", 0.1)),
            activation=str(_config_value(config, "backbone_activation", "gelu")),
            use_layer_norm=bool(_config_value(config, "mlp_use_layer_norm", True)),
        )
    raise ValueError(
        f"Unsupported model-agnostic backbone: {backbone_name}. "
        f"Expected one of {SUPPORTED_MODEL_AGNOSTIC_BACKBONES}."
    )


def _build_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class TemporalFeatureTokenForecastWrapper(nn.Module):
    """Use the same feature-token representation for pretraining and forecasting."""

    def __init__(
        self,
        forecast_model: nn.Module,
        backbone_name: str,
        target_dim: int,
        reconstruction_len: int,
        head_hidden_dim: int = 128,
        dropout: float = 0.15,
        activation: str = "gelu",
        residual_native_forecast: bool = False,
    ):
        super().__init__()
        self.forecast_model = forecast_model
        self.backbone_name = normalize_backbone_name(backbone_name)
        self.input_dim = int(forecast_model.input_dim)
        self.seq_len = int(forecast_model.seq_len)
        self.pred_len = int(forecast_model.pred_len)
        self.target_dim = int(target_dim)
        self.residual_native_forecast = bool(residual_native_forecast)

        if self.backbone_name == "cnn":
            self.d_model = int(self.forecast_model.channels[-1])
        elif self.backbone_name in {"lstm", "bilstm"}:
            self.d_model = int(
                self.forecast_model.hidden_dim * self.forecast_model.direction_count
            )
        elif self.backbone_name == "cnn_lstm":
            self.d_model = int(self.forecast_model.lstm_hidden_dim)
        elif self.backbone_name == "mlp":
            self.d_model = int(self.forecast_model.hidden_dims[-1])
        else:
            raise ValueError(f"Unsupported wrapper backbone: {self.backbone_name}")

        if not self.residual_native_forecast:
            # V1 uses only the shared feature-token path for forecasting.
            self.forecast_model.head = nn.Identity()
        self.feature_queries = nn.Parameter(torch.empty(self.target_dim, self.d_model))
        nn.init.normal_(self.feature_queries, mean=0.0, std=0.02)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, int(head_hidden_dim)),
            _build_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden_dim), self.pred_len),
        )
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, int(reconstruction_len)),
        )
        if self.residual_native_forecast:
            self.token_residual_logit = nn.Parameter(
                torch.full((self.target_dim,), -2.0)
            )
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        else:
            self.register_parameter("token_residual_logit", None)

    def encode_temporal_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, input_dim], got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        if self.backbone_name == "cnn":
            encoded = self.forecast_model.temporal_backbone(x.transpose(1, 2))
            return encoded.transpose(1, 2)

        if self.backbone_name in {"lstm", "bilstm"}:
            normalized = self.forecast_model.input_norm(x)
            encoded, _ = self.forecast_model.lstm(normalized)
            return encoded
        if self.backbone_name == "cnn_lstm":
            normalized = self.forecast_model.input_norm(x)
            conv_features = self.forecast_model.temporal_cnn(
                normalized.transpose(1, 2)
            ).transpose(1, 2)
            encoded, _ = self.forecast_model.lstm(conv_features)
            return encoded
        if self.backbone_name == "mlp":
            flattened = x.reshape(x.size(0), -1)
            encoded = self.forecast_model.input_norm(flattened)
            encoded = self.forecast_model.network[:-1](encoded)
            return encoded.unsqueeze(1)
        raise ValueError(f"Unsupported wrapper backbone: {self.backbone_name}")

    def encode_pretrain_tokens(self, x: torch.Tensor) -> torch.Tensor:
        temporal_sequence = self.encode_temporal_sequence(x)
        queries = self.feature_queries / math.sqrt(float(self.d_model))
        attention_logits = torch.einsum("bld,fd->bfl", temporal_sequence, queries)
        attention = torch.softmax(attention_logits, dim=-1)
        pooled = torch.einsum("bfl,bld->bfd", attention, temporal_sequence)
        return self.token_norm(pooled + self.feature_queries.unsqueeze(0))

    def decode_pretrain_tokens(
        self,
        tokens: torch.Tensor,
        decoder: nn.Module | None = None,
    ) -> torch.Tensor:
        decoder = decoder or self.reconstruction_head
        return decoder(tokens).permute(0, 2, 1)

    def build_cross_delta_decoder(self) -> nn.Module:
        reconstruction_len = int(self.reconstruction_head[-1].out_features)
        return nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, reconstruction_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encode_pretrain_tokens(x)
        token_forecast = self.head(tokens).permute(0, 2, 1)
        if not self.residual_native_forecast:
            return token_forecast
        native_forecast = self.forecast_model(x)
        residual_scale = torch.sigmoid(self.token_residual_logit).view(1, 1, -1)
        return native_forecast + residual_scale * token_forecast


def build_model_agnostic_forecaster(
    backbone_name: str,
    input_dim: int,
    seq_len: int,
    pred_len: int,
    target_dim: int,
    config,
    interface: str | None = None,
    reconstruction_len: int | None = None,
) -> nn.Module:
    forecast_model = build_temporal_forecasting_backbone(
        backbone_name=backbone_name,
        input_dim=input_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        target_dim=target_dim,
        config=config,
    )
    interface = normalize_model_agnostic_interface(
        interface or _config_value(config, "model_agnostic_interface", "legacy")
    )
    if interface == "legacy":
        return forecast_model
    return TemporalFeatureTokenForecastWrapper(
        forecast_model=forecast_model,
        backbone_name=backbone_name,
        target_dim=target_dim,
        reconstruction_len=int(reconstruction_len or seq_len),
        head_hidden_dim=int(_config_value(config, "backbone_head_hidden_dim", 128)),
        dropout=float(_config_value(config, "backbone_dropout", 0.15)),
        activation=str(_config_value(config, "backbone_activation", "gelu")),
        residual_native_forecast=interface == "feature_token_residual_v2",
    )


class TemporalFeatureTokenPretrainAdapter(nn.Module):
    """
    Exposes a common feature-token interface for temporal CNN and LSTM encoders.

    The forecasting model remains unchanged. The learned queries and reconstruction
    decoder are used only during cross-station pretraining and are not exported to
    target-station forecasting checkpoints.
    """

    def __init__(
        self,
        forecast_model: nn.Module,
        backbone_name: str,
        target_dim: int,
        reconstruction_len: int,
    ):
        super().__init__()
        self.forecast_model = forecast_model
        self.backbone_name = normalize_backbone_name(backbone_name)
        self.target_dim = int(target_dim)
        self.pred_len = int(reconstruction_len)

        if self.backbone_name == "cnn":
            self.d_model = int(self.forecast_model.channels[-1])
        elif self.backbone_name == "lstm":
            self.d_model = int(
                self.forecast_model.hidden_dim * self.forecast_model.direction_count
            )
        else:
            raise ValueError(f"Unsupported adapter backbone: {self.backbone_name}")

        self.feature_queries = nn.Parameter(torch.empty(self.target_dim, self.d_model))
        nn.init.normal_(self.feature_queries, mean=0.0, std=0.02)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.pred_len),
        )

        # The normal forecasting head is intentionally not part of the reconstruction
        # objective. It is initialized and trained by the first target-station stage.
        for parameter in self.forecast_model.head.parameters():
            parameter.requires_grad = False

    def encode_temporal_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, input_dim], got {tuple(x.shape)}")
        if x.shape[-1] != self.forecast_model.input_dim:
            raise ValueError(
                f"Expected input_dim={self.forecast_model.input_dim}, got {x.shape[-1]}"
            )

        if self.backbone_name == "cnn":
            encoded = self.forecast_model.temporal_backbone(x.transpose(1, 2))
            return encoded.transpose(1, 2)

        normalized = self.forecast_model.input_norm(x)
        encoded, _ = self.forecast_model.lstm(normalized)
        return encoded

    def encode_pretrain_tokens(self, x: torch.Tensor) -> torch.Tensor:
        temporal_sequence = self.encode_temporal_sequence(x)
        queries = self.feature_queries / math.sqrt(float(self.d_model))
        attention_logits = torch.einsum("bld,fd->bfl", temporal_sequence, queries)
        attention = torch.softmax(attention_logits, dim=-1)
        pooled = torch.einsum("bfl,bld->bfd", attention, temporal_sequence)
        return self.token_norm(pooled + self.feature_queries.unsqueeze(0))

    def decode_pretrain_tokens(
        self,
        tokens: torch.Tensor,
        decoder: nn.Module | None = None,
    ) -> torch.Tensor:
        decoder = decoder or self.reconstruction_head
        decoded = decoder(tokens)
        return decoded.permute(0, 2, 1)

    def build_cross_delta_decoder(self) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.pred_len),
        )

    def export_forecasting_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.forecast_model.state_dict().items()
            if not key.startswith("head.")
        }
