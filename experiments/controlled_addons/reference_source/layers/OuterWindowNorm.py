"""A model-agnostic reversible outer window normalization wrapper.

This is intentionally outside individual model implementations so a common
preprocessing experiment can apply the same input transformation to every
model family.  The wrapper restores the forecast to the input window's scale
before the loss or metrics are computed.
"""

import torch
import torch.nn as nn


class ReversibleOuterWindowNorm(nn.Module):
    def __init__(self, model: nn.Module, eps: float = 1e-5):
        super().__init__()
        self.model = model
        self.eps = eps

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, *args, **kwargs):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + self.eps)
        normalized_x = centered / stdev
        output = self.model(normalized_x, x_mark_enc, x_dec, x_mark_dec, *args, **kwargs)
        return output * stdev + means
