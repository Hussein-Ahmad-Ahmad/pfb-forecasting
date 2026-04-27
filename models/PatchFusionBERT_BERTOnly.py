"""
PatchFusionBERT - BERT-Only Variant (Ablation Study)
Only uses the BERT path without Patch Transformer Encoder
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class Model(nn.Module):
    """
    Ablation: BERT-Only baseline (no Patch Transformer path)
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Patching parameters
        self.patch_len = configs.patch_len if hasattr(configs, 'patch_len') else 16
        self.stride = configs.stride if hasattr(configs, 'stride') else 8
        
        # Calculate number of patches
        self.num_patches = (max(configs.seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        
        # Input projection to BERT hidden size
        self.input_projection = nn.Linear(self.patch_len, configs.d_model)
        
        # ONLY BERT Encoder (NO Patch Transformer)
        bert_config = BertConfig(
            hidden_size=configs.d_model,
            num_hidden_layers=configs.e_layers,
            num_attention_heads=configs.n_heads,
            intermediate_size=configs.d_ff,
            hidden_dropout_prob=configs.dropout,
            attention_probs_dropout_prob=configs.dropout,
            max_position_embeddings=512
        )
        self.bert_encoder = BertModel(bert_config)
        
        # Projection head (only from BERT encoder)
        self.head_nf = configs.d_model * self.num_patches
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = nn.Linear(self.head_nf, configs.pred_len * configs.c_out)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = nn.Linear(self.head_nf, configs.seq_len * configs.c_out)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)
    
    def create_patches(self, x):
        """Create patches from time series"""
        B, L, M = x.shape
        x = x.permute(0, 2, 1).contiguous()  # [B, M, L]
        x = x.reshape(B * M, L)  # [B*M, L]
        
        # Create patches
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B*M, num_patches, patch_len]
        return patches, B, M
    
    def forecast(self, x_enc):
        # Create patches
        patches, B, M = self.create_patches(x_enc)  # [B*M, num_patches, patch_len]
        
        # Project patches to BERT hidden size
        bert_input = self.input_projection(patches)  # [B*M, num_patches, d_model]
        
        # BERT encoding
        bert_output = self.bert_encoder(inputs_embeds=bert_input).last_hidden_state  # [B*M, num_patches, d_model]
        
        # Flatten and project
        bert_output = bert_output.reshape(B * M, -1)  # [B*M, num_patches * d_model]
        dec_out = self.head(bert_output)  # [B*M, pred_len]
        dec_out = dec_out.reshape(B, M, -1).permute(0, 2, 1).contiguous()  # [B, pred_len, M]
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        return None
