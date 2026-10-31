"""Opt-in controls. Historical model files and their forward paths are untouched."""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn

from models import DLinear, PatchTST, PFB_Direct, PFB_Projected, TiDE, TimeXer, iTransformer
from utils.window_preprocessing import normalize_window


VARIANTS = (
    "concat34", "linear_d", "gelu_d", "linear_d_reg", "projected34",
    "linear_d_lift", "gelu_d_lift", "linear_2d", "gelu_2d",
)
FAMILIES = {
    "PatchTST": PatchTST, "PFB-Direct": PFB_Direct,
    "PFB-Projected": PFB_Projected, "DLinear": DLinear,
    "iTransformer": iTransformer, "TiDE": TiDE, "TimeXer": TimeXer,
}


def configuration(dataset, params=None):
    from .design import DATASETS
    info = DATASETS[dataset]
    values = dict(
        task_name="long_term_forecast", seq_len=336, label_len=96, pred_len=192,
        enc_in=info["channels"], dec_in=info["channels"], c_out=info["channels"],
        d_model=128, n_heads=8, e_layers=3, d_layers=1, d_ff=512,
        factor=1, dropout=0.1, activation="gelu", embed="timeF", freq=info["freq"],
        features="M", patch_len=16, stride=8, pfb_k=0, use_norm=1,
        moving_avg=25, individual=False, learning_rate=0.0001,
    )
    values.update(params or {})
    return SimpleNamespace(**values)


class MechanismControl(nn.Module):
    """All variants share Projected's 3+4 encoders and original token/head order.

    'reg' retains Dropout+LayerNorm; the pure-linear controls do not. Width-d
    lifting duplicates z/sqrt(2) without learned parameters, fixing the head
    interface at 2d for the width x GELU contrasts. Compression changes capacity;
    it is not described as a parameter-isolated intervention.
    """
    def __init__(self, cfg, variant):
        super().__init__()
        if variant not in VARIANTS or cfg.e_layers != 3 or cfg.pfb_k != 0:
            raise ValueError("Mechanism controls require a named variant and fixed 3+4 depths")
        self.variant = variant
        self.base = PFB_Projected.Model(cfg)
        self.lift = variant.endswith("_lift")
        out_width = 2 * cfg.d_model if variant in {"concat34", "linear_2d", "gelu_2d"} else cfg.d_model
        head_width = 2 * cfg.d_model if self.lift else out_width
        # Construct the common wide head BEFORE any variant-specific RNG draws.
        if head_width != cfg.d_model:
            n_patches = (cfg.seq_len - cfg.patch_len) // cfg.stride + 2
            self.base.head = PFB_Projected.FlattenHead(
                cfg.enc_in, head_width * n_patches, cfg.pred_len, cfg.dropout)
        linear, _, dropout, norm = self.base.fusion_projection
        if variant == "concat34":
            fusion = nn.Identity()
        elif variant == "projected34":
            fusion = self.base.fusion_projection
        elif variant == "linear_d_reg":
            fusion = nn.Sequential(linear, nn.Identity(), dropout, norm)
        else:
            if out_width == 2 * cfg.d_model:
                linear = nn.Linear(2 * cfg.d_model, out_width)
            fusion = nn.Sequential(linear, nn.GELU() if variant.startswith("gelu") else nn.Identity())
        self.base.fusion_projection = fusion
        self.head_width = head_width

    def forward(self, x, x_mark=None, x_dec=None, y_mark=None, mask=None):
        x, mean, scale = normalize_window(x, True)
        tokens, channels = self.base.patch_embedding(x.permute(0, 2, 1))
        first, _ = self.base.patch_encoder(tokens)
        second = self.base.secondary_encoder(tokens)
        z = self.base.fusion_projection(torch.cat([first, second], dim=-1))
        if self.lift:
            z = torch.cat([z, z], dim=-1) / math.sqrt(2)
        z = z.reshape(-1, channels, z.shape[-2], z.shape[-1])
        result = self.base.head(z).permute(0, 2, 1)
        return result * scale + mean


class NormalizationControl(nn.Module):
    """Exactly one reversible window standardization, or none.

    Replays the original four models' encoder/head paths, bypassing ONLY their
    hard-coded window standardization. Learned LayerNorm/BatchNorm stay active.
    ON/OFF receive identical model initialization under the same seed.
    """
    def __init__(self, cfg, family, enabled):
        super().__init__()
        if family not in {"DLinear", "PatchTST", "PFB-Direct", "PFB-Projected"}:
            raise ValueError("Unsupported normalization control")
        if cfg.pfb_k:
            raise ValueError("Refinement-depth diagnostics are outside this design")
        self.base = FAMILIES[family].Model(cfg)
        self.family, self.enabled = family, enabled

    def forward(self, x, x_mark=None, x_dec=None, y_mark=None, mask=None):
        x, mean, scale = normalize_window(x, self.enabled)
        if self.family == "DLinear":
            result = self.base.encoder(x)
        else:
            tokens, channels = self.base.patch_embedding(x.permute(0, 2, 1))
            if self.family == "PatchTST":
                z, _ = self.base.encoder(tokens)
            else:
                first, _ = self.base.patch_encoder(tokens)
                second = self.base.secondary_encoder(tokens)
                z = torch.cat([first, second], dim=-1)
                if self.family == "PFB-Projected":
                    z = self.base.fusion_projection(z)
            z = z.reshape(-1, channels, z.shape[-2], z.shape[-1])
            if self.family == "PatchTST":
                z = z.permute(0, 1, 3, 2)
            result = self.base.head(z).permute(0, 2, 1)
        return result * scale + mean


def build_model(spec):
    cfg = configuration(spec["dataset"], spec["params"])
    if spec["study"] == "projected" and spec["variant"] in VARIANTS:
        return MechanismControl(cfg, spec["variant"])
    if spec["study"] == "preprocessing":
        return NormalizationControl(cfg, spec["family"], spec["variant"] == "norm_on")
    if spec["family"] == "DLinear":
        # The historical constructor ignores configs.individual unless passed explicitly.
        return DLinear.Model(cfg, individual=cfg.individual)
    return FAMILIES[spec["family"]].Model(cfg)


def patch_budget(params, channels=1):
    """Exact trainable parameters; dense/attention MACs per full sample.

    MAC convention excludes bias additions, normalization, activations, pooling,
    positional additions and other elementwise operations. NOT measured latency.
    """
    cfg = configuration("ETTh2", params)
    d, f, e = cfg.d_model, cfg.d_ff, cfg.e_layers
    n = (cfg.seq_len - cfg.patch_len) // cfg.stride + 2
    layer_params = 4*d*d + 2*d*f + 9*d + f
    parameters = cfg.patch_len*d + e*layer_params + 2*d + n*d*cfg.pred_len + cfg.pred_len
    layer_macs = 4*n*d*d + 2*n*d*f + 2*n*n*d
    macs = channels * (n*cfg.patch_len*d + e*layer_macs + n*d*cfg.pred_len)
    return {"parameters": parameters, "macs_per_sample": macs}


def profile_model(spec):
    """CPU hooks count all executed Linear/Conv1d and both attention contractions."""
    from layers.SelfAttention_Family import FullAttention
    cfg = configuration(spec["dataset"], spec["params"])
    model = build_model(spec).eval()
    total, handles = [0], []

    def count(module, inputs, output):
        if isinstance(module, nn.Linear):
            total[0] += output.numel() * module.in_features
        elif isinstance(module, nn.Conv1d):
            total[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
        elif isinstance(module, FullAttention):
            q, k, v = inputs[:3]
            total[0] += q.shape[0]*q.shape[2]*q.shape[1]*k.shape[1]*(q.shape[3]+v.shape[3])
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, FullAttention)):
            handles.append(module.register_forward_hook(count))
    try:
        with torch.inference_mode():
            x = torch.zeros(1, cfg.seq_len, cfg.enc_in)
            result = model(x, None, None, None)
        if result.shape != (1, cfg.pred_len, cfg.enc_in) or not torch.isfinite(result).all():
            raise ValueError("Profile failed output shape/finite check")
        return {"parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "macs_per_sample": total[0], "head_width": getattr(model, "head_width", cfg.d_model)}
    finally:
        for handle in handles:
            handle.remove()


def match_patch_budget(target, tolerance=0.01):
    """Search genuine PatchTST configurations; no unused parameter padding."""
    candidates = []
    for d in range(64, 257, 8):
        for depth in range(1, 15):
            for ff in range(64, 2049, 8):
                params = dict(d_model=d, n_heads=8, e_layers=depth, d_ff=ff)
                measured = patch_budget(params)
                dp = measured["parameters"] / target["parameters"] - 1
                dm = measured["macs_per_sample"] / target["macs_per_sample"] - 1
                score = max(abs(dp), abs(dm))
                if score <= tolerance:
                    candidates.append((score, abs(dp)+abs(dm), d, depth, ff, dp, dm))
    if not candidates:
        raise ValueError("No simultaneous parameter/MAC match within tolerance; do not launch or claim matching")
    score, _, d, depth, ff, dp, dm = min(candidates)
    return {"params": dict(d_model=d, n_heads=8, e_layers=depth, d_ff=ff),
            "parameter_difference_pct": 100*dp, "mac_difference_pct": 100*dm,
            "tolerance_pct": 100*tolerance, "matched": True}
