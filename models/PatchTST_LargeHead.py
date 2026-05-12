"""
PatchTST_LargeHead — Ablation variant for the PatchFusionBERT study.

Identical backbone to PatchTST (same patch embedding + encoder) but replaces
the single-layer FlattenHead with a 2-layer MLP head:
    Flatten → Linear(nf, d_ff_head) → GELU → Dropout → Linear(d_ff_head, pred_len)

Purpose: test whether adding capacity to the output head (rather than through
BERT refinement and adaptive fusion as in PFBv0) can recover similar gains.

Head size: nf=5376 with d_ff_head=512 gives ~2.85M head parameters vs PatchTST
base head of ~1.03M, bringing total params to ~3.45M (slightly above PFBv0's 3.26M).
This is the "capacity-matched at the head" control for the ablation chain.
"""

import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class MLPHead(nn.Module):
    """2-layer MLP prediction head shared across variables."""

    def __init__(self, nf, target_window, d_ff_head=512, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.fc1 = nn.Linear(nf, d_ff_head)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(head_dropout)
        self.fc2 = nn.Linear(d_ff_head, target_window)

    def forward(self, x):  # x: [bs × nvars × d_model × patch_num]
        x = self.flatten(x)      # [bs × nvars × nf]
        x = self.fc1(x)          # [bs × nvars × d_ff_head]
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)          # [bs × nvars × target_window]
        return x


class Model(nn.Module):
    """
    PatchTST backbone + 2-layer MLP output head.
    Same configs as standard PatchTST; uses d_ff as the MLP head hidden dim
    (configs.d_ff_head if set, otherwise falls back to configs.d_ff).
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        patch_len = getattr(configs, 'patch_len', patch_len)
        stride = getattr(configs, 'stride', stride)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder (identical to PatchTST)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2),
            ),
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        d_ff_head = getattr(configs, 'd_ff_head', configs.d_ff)

        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            self.head = MLPHead(self.head_nf, configs.pred_len,
                                d_ff_head=d_ff_head,
                                head_dropout=configs.dropout)
        else:
            raise NotImplementedError(
                f"PatchTST_LargeHead only supports forecasting tasks, got: {self.task_name}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Instance normalisation
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)

        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)           # [bs × nvars × pred_len]
        dec_out = dec_out.permute(0, 2, 1)

        # De-normalise
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise NotImplementedError
