"""
PatchFusionBERT - refinement-only variant for component ablation.

Chain interpretation:
- PatchOnly: backbone only
- RefineOnly: backbone + serial refinement without fusion
- v0 (pfb_k=0): backbone + fusion without deeper refinement
- v0 (pfb_k>0): full model with fusion and deeper refinement
"""

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class RefinementEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, attention_dropout=dropout, output_attention=False),
                    d_model,
                    n_heads,
                ),
                d_model,
                d_ff,
                dropout=dropout,
                activation='gelu',
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, _ = layer(x, attn_mask=attn_mask)
        return self.norm(x)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.pfb_k = int(getattr(configs, 'pfb_k', 3))

        patch_len = getattr(configs, 'patch_len', patch_len)
        stride = getattr(configs, 'stride', stride)
        padding = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.patch_encoder = Encoder(
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
                    activation='gelu'
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.refinement_encoder = RefinementEncoder(
            configs.d_model,
            configs.n_heads,
            configs.d_ff,
            max(self.pfb_k, 1),
            configs.dropout,
        )

        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(
            configs.enc_in,
            self.head_nf,
            configs.pred_len,
            head_dropout=configs.dropout,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_patches, n_vars = self.patch_embedding(x_enc.permute(0, 2, 1))
        patch_enc_out, _ = self.patch_encoder(enc_patches)
        refined_out = self.refinement_encoder(patch_enc_out)

        refined_out = torch.reshape(
            refined_out, (-1, n_vars, refined_out.shape[-2], refined_out.shape[-1]))
        dec_out = self.head(refined_out)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None