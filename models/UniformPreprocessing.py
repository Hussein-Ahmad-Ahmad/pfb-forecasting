"""Common reversible window standardization for cross-family controls."""

from __future__ import annotations

import torch
import torch.nn as nn


class Wrapper(nn.Module):
    """Apply the same per-window, per-channel transform around any forecaster.

    The wrapper standardizes only the encoder window and reverses the transform
    on the forecast. Native layers inside each architecture remain unchanged.
    """

    def __init__(self, model: nn.Module, eps: float = 1e-5):
        super().__init__()
        self.model = model
        self.eps = eps

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        mean = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - mean
        scale = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + self.eps
        )
        normalized = centered / scale
        output = self.model(normalized, x_mark_enc, x_dec, x_mark_dec, mask=mask)
        return output * scale[:, :1, :] + mean[:, :1, :]
