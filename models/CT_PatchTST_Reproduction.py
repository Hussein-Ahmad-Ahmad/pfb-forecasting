"""Independent paper-aligned implementation of CT-PatchTST.

The implementation follows arXiv:2501.08620: RevIN-style window
normalization, patch length 16 with 50% overlap, channel attention followed
by temporal attention, and four channel-time encoder blocks.  It is not an
official author release.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ChannelTimeBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        channel_heads: int,
        time_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.channel_attention = nn.MultiheadAttention(
            d_model, channel_heads, dropout=dropout, batch_first=True
        )
        self.time_attention = nn.MultiheadAttention(
            d_model, time_heads, dropout=dropout, batch_first=True
        )
        self.channel_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.time_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm_channel_attention = nn.LayerNorm(d_model)
        self.norm_channel_ffn = nn.LayerNorm(d_model)
        self.norm_time_attention = nn.LayerNorm(d_model)
        self.norm_time_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, patches, d_model]
        batch, channels, patches, width = x.shape

        by_patch = x.permute(0, 2, 1, 3).reshape(batch * patches, channels, width)
        attended, _ = self.channel_attention(
            by_patch, by_patch, by_patch, need_weights=False
        )
        by_patch = self.norm_channel_attention(by_patch + self.dropout(attended))
        by_patch = self.norm_channel_ffn(
            by_patch + self.dropout(self.channel_ffn(by_patch))
        )
        x = by_patch.reshape(batch, patches, channels, width).permute(0, 2, 1, 3)

        by_channel = x.reshape(batch * channels, patches, width)
        attended, _ = self.time_attention(
            by_channel, by_channel, by_channel, need_weights=False
        )
        by_channel = self.norm_time_attention(
            by_channel + self.dropout(attended)
        )
        by_channel = self.norm_time_ffn(
            by_channel + self.dropout(self.time_ffn(by_channel))
        )
        return by_channel.reshape(batch, channels, patches, width)


class Model(nn.Module):
    """CT-PatchTST channel-then-time patch encoder."""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("CT-PatchTST-Reproduction supports forecasting only")

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.patch_len = int(getattr(configs, "patch_len", 16))
        self.stride = int(getattr(configs, "stride", self.patch_len // 2))
        self.padding = self.stride
        self.patch_count = (
            (self.seq_len + self.padding - self.patch_len) // self.stride + 1
        )

        d_model = int(configs.d_model)
        time_heads = int(configs.n_heads)
        channel_heads = int(getattr(configs, "ct_channel_heads", 1))
        if d_model % time_heads or d_model % channel_heads:
            raise ValueError("d_model must be divisible by both attention head counts")

        self.patch_projection = nn.Linear(self.patch_len, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 1, self.patch_count, d_model)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.input_dropout = nn.Dropout(float(configs.dropout))
        self.blocks = nn.ModuleList(
            [
                ChannelTimeBlock(
                    d_model=d_model,
                    channel_heads=channel_heads,
                    time_heads=time_heads,
                    d_ff=int(configs.d_ff),
                    dropout=float(configs.dropout),
                )
                for _ in range(int(configs.e_layers))
            ]
        )
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Dropout(float(configs.dropout)),
            nn.Linear(self.patch_count * d_model, self.pred_len),
        )

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        normalized = centered / stdev

        x = normalized.permute(0, 2, 1)
        x = F.pad(x, (0, self.padding), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.shape[-2] != self.patch_count:
            raise RuntimeError(
                f"Expected {self.patch_count} patches, received {patches.shape[-2]}"
            )
        encoded = self.input_dropout(
            self.patch_projection(patches) + self.position_embedding
        )
        for block in self.blocks:
            encoded = block(encoded)

        forecast = self.head(encoded).permute(0, 2, 1)
        return forecast * stdev[:, :1, :] + means[:, :1, :]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc)
