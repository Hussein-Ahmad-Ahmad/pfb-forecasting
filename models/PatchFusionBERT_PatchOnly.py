"""
PatchFusionBERT - Patch-Only Variant (Ablation Study)
Only uses the Patch Transformer Encoder path without BERT
"""

import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Model(nn.Module):
    """
    Ablation: Patch-Only baseline (no BERT path)
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Patching parameters
        self.patch_len = configs.patch_len if hasattr(configs, 'patch_len') else 16
        self.stride = configs.stride if hasattr(configs, 'stride') else 8
        self.padding = self.stride
        
        # Calculate number of patches
        self.num_patches = int((configs.seq_len - self.patch_len) / self.stride + 2)
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride,
            self.padding, configs.dropout
        )
        
        # ONLY Patch Transformer Encoder (NO BERT)
        self.patch_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Projection head (only from patch encoder)
        self.head_nf = configs.d_model * self.num_patches
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = nn.Linear(self.head_nf, configs.pred_len * configs.c_out)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = nn.Linear(self.head_nf, configs.seq_len * configs.c_out)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)
    
    def forecast(self, x_enc):
        B, L, M = x_enc.shape

        # Patch embedding
        enc_out, n_vars = self.patch_embedding(x_enc.permute(0, 2, 1))  # [B*M, num_patches, d_model]
        
        # Patch Transformer encoding
        patch_out, _ = self.patch_encoder(enc_out)  # [B*M, num_patches, d_model]
        
        # Flatten and project
        patch_out = patch_out.reshape(B * M, -1)  # [B*M, num_patches * d_model]
        dec_out = self.head(patch_out)  # [B*M, pred_len]
        dec_out = dec_out.reshape(B, M, -1).permute(0, 2, 1).contiguous()  # [B, pred_len, M]
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        return None
