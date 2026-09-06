"""Shared reversible window preprocessing used by controlled comparisons."""

from __future__ import annotations

import torch


def normalize_window(x: torch.Tensor, enabled: bool, eps: float = 1e-5):
    if not enabled:
        mean = torch.zeros_like(x[:, :1, ...])
        scale = torch.ones_like(x[:, :1, ...])
        return x, mean, scale
    mean = x.mean(dim=1, keepdim=True).detach()
    centered = x - mean
    scale = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + eps)
    return centered / scale, mean, scale
