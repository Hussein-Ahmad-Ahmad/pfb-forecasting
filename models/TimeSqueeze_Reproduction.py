"""Paper-aligned TimeSqueeze tokenizer reproduction for in-distribution forecasting.

This module implements the lightweight forecasting experiment described in
Section 6 of TimeSqueeze (arXiv:2603.11352): relative-deviation dynamic
patching, boundary-token retention, a non-causal encoder-only Transformer,
and causal unpatching.  It is an independent implementation, not the authors'
117M/469M pretrained Time-MoE/Mamba foundation model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FullResolutionBlock(nn.Module):
    """Lightweight full-resolution residual block around the dynamic tokenizer."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=3, groups=d_model, bias=True
        )
        self.gate = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.norm(x)
        local = self.depthwise(F.pad(z.transpose(1, 2), (2, 0))).transpose(1, 2)
        z = torch.sigmoid(self.gate(z)) * torch.tanh(self.value(local))
        return residual + self.dropout(z)


class Model(nn.Module):
    """Dynamic-patching forecasting model aligned to TimeSqueeze Section 6."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("TimeSqueeze-Reproduction supports forecasting only")

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.tau = float(getattr(configs, "timesqueeze_tau", 0.3))
        self.power_window = int(getattr(configs, "timesqueeze_power_window", 8))
        self.max_patch_size = int(getattr(configs, "timesqueeze_max_patch", 8))
        if self.power_window < 1 or self.max_patch_size < 1:
            raise ValueError("TimeSqueeze window and maximum patch size must be positive")

        d_model = int(configs.d_model)
        n_heads = int(configs.n_heads)
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.input_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Embedding(self.seq_len, d_model)
        self.full_encoder = FullResolutionBlock(d_model, configs.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(configs.d_ff),
            dropout=float(configs.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_transformer = nn.TransformerEncoder(
            layer, num_layers=int(configs.e_layers), enable_nested_tensor=False
        )
        self.full_decoder = FullResolutionBlock(d_model, configs.dropout)
        self.output_projection = nn.Linear(d_model, 1)
        self.horizon_head = nn.Linear(self.seq_len, self.pred_len)

    def _boundaries(self, x: torch.Tensor) -> torch.Tensor:
        """Return relative-deviation boundaries for flattened univariate series."""
        n_series, length = x.shape
        square = x.square()
        prefix = F.pad(torch.cumsum(square, dim=1), (1, 0))
        end = torch.arange(length, device=x.device)
        start = (end - self.power_window).clamp_min(0)
        totals = prefix[:, end] - prefix[:, start]
        counts = (end - start).clamp_min(1).to(x.dtype)
        local_rms = torch.sqrt(totals / counts.unsqueeze(0) + 1e-8)

        boundary = torch.zeros(
            n_series, length, dtype=torch.bool, device=x.device
        )
        boundary[:, 0] = True
        if length > 1:
            deviation = (x[:, 1:] - x[:, :-1]).abs()
            boundary[:, 1:] = deviation > self.tau * local_rms[:, 1:]

        # The paper caps patches at eight samples. Regular safety boundaries
        # enforce that cap while retaining every deviation-driven boundary.
        boundary[:, :: self.max_patch_size] = True
        return boundary

    @staticmethod
    def _pack_boundary_tokens(
        encoded: torch.Tensor, boundary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack variable-count boundary tokens and preserve their source positions."""
        n_series, length, width = encoded.shape
        source_positions = torch.arange(length, device=encoded.device).expand(
            n_series, length
        )
        sortable = source_positions.masked_fill(~boundary, length)
        ordered, _ = sortable.sort(dim=1)
        token_count = boundary.sum(dim=1)
        max_tokens = int(token_count.max().item())
        positions = ordered[:, :max_tokens]
        valid = positions < length
        safe_positions = positions.clamp_max(length - 1)
        tokens = torch.gather(
            encoded, 1, safe_positions.unsqueeze(-1).expand(-1, -1, width)
        )
        return tokens, valid, safe_positions

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        normalized = centered / stdev

        batch, length, channels = normalized.shape
        if length != self.seq_len:
            raise ValueError(f"Expected lookback {self.seq_len}, received {length}")
        series = normalized.permute(0, 2, 1).reshape(batch * channels, length)

        full = self.input_projection(series.unsqueeze(-1))
        full = self.full_encoder(full)
        boundary = self._boundaries(series)
        tokens, valid, positions = self._pack_boundary_tokens(full, boundary)
        tokens = tokens + self.position_embedding(positions)
        tokens = self.global_transformer(tokens, src_key_padding_mask=~valid)

        # Every timestep receives its most recent processed boundary token.
        patch_ids = boundary.long().cumsum(dim=1) - 1
        expanded = torch.gather(
            tokens,
            1,
            patch_ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        decoded = self.full_decoder(full + expanded)
        decoded_series = self.output_projection(decoded).squeeze(-1)
        forecast = self.horizon_head(decoded_series)
        forecast = forecast.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
        return forecast * stdev[:, :1, :] + means[:, :1, :]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
