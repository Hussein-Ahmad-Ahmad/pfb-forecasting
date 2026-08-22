import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class Model(nn.Module):
    """
    Single-path serial control for PFB-Direct.

    It uses one PatchTST-style encoder path with 2 * e_layers Transformer blocks and
    the same 2d flattened head interface as PFB-Direct. This keeps the Transformer
    block budget and head width close while replacing dual-parallel processing with a
    single serial path.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        patch_len = getattr(configs, "patch_len", patch_len)
        stride = getattr(configs, "stride", stride)
        padding = stride
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )
        serial_layers = int(getattr(configs, "single_serial_layers", 2 * configs.e_layers))
        self.encoder = Encoder(
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
                for _ in range(serial_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.to_fused_width = nn.Linear(configs.d_model, 2 * configs.d_model)
        self.head_nf = 2 * configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(self.head_nf, configs.pred_len, head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev

        enc_out, n_vars = self.patch_embedding(x_norm.permute(0, 2, 1))
        enc_out, _ = self.encoder(enc_out)
        enc_out = self.to_fused_width(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        dec_out = self.head(enc_out).permute(0, 2, 1)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
