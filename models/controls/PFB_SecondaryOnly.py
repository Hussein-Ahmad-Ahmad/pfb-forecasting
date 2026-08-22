"""
PFB-SecondaryOnly: secondary-encoder-only component control.

The encoder is initialized from configuration and trained on time-series patch
tokens.
"""

import torch
import torch.nn as nn
from models.PFB_Direct import SecondaryPatchEncoder


class Model(nn.Module):
    """
    Secondary-encoder-only component control.
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
        
        # Input projection to the encoder hidden size
        self.input_projection = nn.Linear(self.patch_len, configs.d_model)
        
        # Secondary encoder
        self.secondary_encoder = SecondaryPatchEncoder(
            configs.d_model,
            configs.n_heads,
            configs.d_ff,
            configs.e_layers,
            configs.dropout,
        )
        
        # Projection head
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
        
        # Project patches to the encoder hidden size
        secondary_input = self.input_projection(patches)  # [B*M, num_patches, d_model]
        
        # Secondary encoding
        secondary_output = self.secondary_encoder(secondary_input)  # [B*M, num_patches, d_model]
        
        # Flatten and project
        secondary_output = secondary_output.reshape(B * M, -1)  # [B*M, num_patches * d_model]
        dec_out = self.head(secondary_output)  # [B*M, pred_len]
        dec_out = dec_out.reshape(B, M, -1).permute(0, 2, 1).contiguous()  # [B, pred_len, M]
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        return None
