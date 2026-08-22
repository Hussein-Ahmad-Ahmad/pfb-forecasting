"""
PatchFusionBERT v2: projected parallel patch-encoder fusion.

Dual-Stream Architecture:
1. PatchTranEnc Path: Standard Transformer Encoder for local temporal patterns
2. Secondary Path: independently parameterized Transformer encoder over the same patches
3. Fusion Strategy: Concatenate representations and predict via linear head
"""

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class SecondaryPatchEncoder(nn.Module):
    """Secondary Transformer encoder operating on the shared patch tokens."""
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout):
        super().__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, attention_dropout=dropout, output_attention=False),
                    d_model, n_heads
                ),
                d_model,
                d_ff,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, attn_mask=None):
        # x: [B*C, N, d_model]
        for layer in self.layers:
            x, _ = layer(x, attn_mask=attn_mask)
        return self.norm(x)


class FlattenHead(nn.Module):
    """Flattening head for forecasting"""
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
    """
    PatchFusionBERT v2: projected parallel patch-encoder fusion.
    
    Architecture:
    - Stream 1 (PatchTranEnc): Captures local patterns with standard attention
    - Stream 2 (secondary patch encoder): parallel transformation of shared patch tokens
    - Fusion: Concatenate both streams and project to forecast
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # pfb_k controls the depth of the secondary stream.
        # 0 means use default (max(e_layers+1, 3)); >0 means use pfb_k layers directly.
        self.pfb_k = int(getattr(configs, 'pfb_k', 0))
        # Use values from configs if available, otherwise use defaults
        patch_len = getattr(configs, 'patch_len', patch_len)
        stride = getattr(configs, 'stride', stride)
        padding = stride

        # Shared patch embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Stream 1: PatchTranEnc Path (Standard Transformer)
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

        # Stream 2: secondary parallel patch encoder.
        # Use pfb_k as secondary depth if explicitly set; otherwise use the default heuristic.
        if self.pfb_k > 0:
            secondary_layers = self.pfb_k
        else:
            secondary_layers = max(configs.e_layers + 1, 3)
        # Attribute name retained for compatibility with existing checkpoints and hooks.
        self.bert_encoder = SecondaryPatchEncoder(
            configs.d_model, 
            configs.n_heads, 
            configs.d_ff, 
            secondary_layers,
            configs.dropout
        )

        # Fusion layer: Projects concatenated features
        # After concatenation: 2 * d_model dimensions
        self.fusion_projection = nn.Sequential(
            nn.Linear(2 * configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.LayerNorm(configs.d_model)
        )

        # Prediction head
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, 
                               head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Shared patch embedding
        # Input: [B, L, C] -> Permute to [B, C, L]
        enc_patches, n_vars = self.patch_embedding(x_enc.permute(0, 2, 1))
        # enc_patches: [B*C, N, d_model]
        
        # Stream 1: PatchTranEnc Path
        # Captures local temporal patterns and short-term dependencies
        patch_enc_out, _ = self.patch_encoder(enc_patches)
        # patch_enc_out: [B*C, N, d_model]
        
        # Stream 2: independently parameterized secondary patch encoder
        secondary_enc_out = self.bert_encoder(enc_patches)
        # secondary_enc_out: [B*C, N, d_model]
        
        # Fusion: Concatenate both streams
        fused = torch.cat([patch_enc_out, secondary_enc_out], dim=-1)
        # fused: [B*C, N, 2*d_model]
        
        # Project fused features
        fused = self.fusion_projection(fused)
        # fused: [B*C, N, d_model]
        
        # Reshape for prediction head
        fused = torch.reshape(
            fused, (-1, n_vars, fused.shape[-2], fused.shape[-1]))
        # fused: [B, C, N, d_model]
        
        # Generate forecast
        dec_out = self.head(fused)
        # dec_out: [B, C, pred_len]
        
        dec_out = dec_out.permute(0, 2, 1)
        # dec_out: [B, pred_len, C]
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
