"""PFB cross-variate diagnostic.

This model retains the PFB-Projected temporal paths and inserts one gated
cross-variate attention block after temporal fusion.  It is an experimental
control for testing whether explicit channel interaction changes the
PatchTST-family comparison; it is not one of the two main PFB variants.
"""

import torch
import torch.nn as nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class SecondaryPatchEncoder(nn.Module):
    """Independently parameterized encoder over the shared patch tokens."""

    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, _ = layer(x, attn_mask=attn_mask)
        return self.norm(x)


class GatedCrossVariateBlock(nn.Module):
    """Mix channels independently at every temporal-patch position."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch, patches, variables, d_model]
        batch, patches, variables, width = x.shape
        tokens = x.reshape(batch * patches, variables, width)
        mixed, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        gate = torch.sigmoid(self.gate(torch.cat([tokens, mixed], dim=-1)))
        tokens = self.norm(tokens + self.dropout(gate * mixed))
        return tokens.reshape(batch, patches, variables, width)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class Model(nn.Module):
    """Projected parallel PFB with a post-fusion cross-variate block."""

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        patch_len = getattr(configs, "patch_len", patch_len)
        stride = getattr(configs, "stride", stride)

        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            patch_len,
            stride,
            stride,
            configs.dropout,
        )
        self.patch_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation="gelu",
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        secondary_layers = max(configs.e_layers + 1, 3)
        self.secondary_encoder = SecondaryPatchEncoder(
            configs.d_model,
            configs.n_heads,
            configs.d_ff,
            secondary_layers,
            configs.dropout,
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(2 * configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.LayerNorm(configs.d_model),
        )
        self.cross_variate = GatedCrossVariateBlock(
            configs.d_model,
            configs.n_heads,
            configs.dropout,
        )
        patch_count = int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(
            configs.enc_in,
            configs.d_model * patch_count,
            configs.pred_len,
            head_dropout=configs.dropout,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        normalized = x_enc - means
        stdev = torch.sqrt(
            torch.var(normalized, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        normalized = normalized / stdev

        patches, n_vars = self.patch_embedding(normalized.permute(0, 2, 1))
        primary, _ = self.patch_encoder(patches)
        secondary = self.secondary_encoder(patches)
        fused = self.fusion_projection(torch.cat([primary, secondary], dim=-1))

        batch = x_enc.shape[0]
        fused = fused.reshape(batch, n_vars, fused.shape[-2], fused.shape[-1])
        fused = fused.permute(0, 2, 1, 3)
        fused = self.cross_variate(fused)
        fused = fused.permute(0, 2, 1, 3)

        output = self.head(fused).permute(0, 2, 1)
        output = output * stdev[:, 0, :].unsqueeze(1)
        output = output + means[:, 0, :].unsqueeze(1)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            output = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return output[:, -self.pred_len :, :]
        return None
