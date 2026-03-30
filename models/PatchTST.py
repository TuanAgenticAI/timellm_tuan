import math

import torch
import torch.nn as nn

from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize


class Model(nn.Module):
    """
    PatchTST-style baseline for time series forecasting.

    This implementation uses:
    - patching along the time dimension (with replication padding),
    - a Transformer encoder over patch tokens,
    - a simple flatten head to predict `pred_len` for each variable independently.

    Interface compatibility:
    forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) -> (B, pred_len, C)
    """

    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.enc_in = getattr(configs, "enc_in", 1)
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = getattr(configs, "e_layers", 2)
        self.d_ff = configs.d_ff
        self.dropout_p = getattr(configs, "dropout", 0.1)

        self.patch_len = configs.patch_len
        self.stride = configs.stride

        self.normalize_layers = Normalize(self.enc_in, affine=False)

        # PatchEmbedding in this repo pads by (0, stride) before unfolding.
        # Effective length: seq_len + stride
        # num_patches = floor((L_eff - patch_len)/stride) + 1
        l_eff = self.seq_len + self.stride
        num_patches = (l_eff - self.patch_len) // self.stride + 1
        self.num_patches = max(1, int(num_patches))

        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=self.dropout_p,
        )

        # Learnable positional embedding over patches.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)

        self.head = nn.Linear(self.num_patches * self.d_model, self.pred_len)
        self.dropout = nn.Dropout(self.dropout_p)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc: (B, seq_len, C)
        Returns:
            (B, pred_len, C)
        """
        x_enc = self.normalize_layers(x_enc, "norm")
        bsz, seq_len, c_in = x_enc.shape

        # PatchTST is applied per variable:
        # (B, T, C) -> (B, C, T)
        x = x_enc.permute(0, 2, 1).contiguous()

        # tokens: (B*C, num_patches, d_model)
        tokens, n_vars = self.patch_embedding(x)
        # n_vars should equal c_in
        if n_vars != c_in:
            # Defensive: reshape assumes the same c_in dimension.
            # If this happens, it likely means patch_embedding behavior changed.
            raise RuntimeError(f"Unexpected n_vars from patch_embedding: {n_vars} != {c_in}")

        # Add positional embedding (truncate in case of edge rounding).
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :].to(tokens.device, tokens.dtype)
        tokens = self.dropout(tokens)

        enc = self.encoder(tokens)  # (B*C, num_patches, d_model)

        # Flatten across patches and apply head
        enc = enc.reshape(enc.size(0), -1)  # (B*C, num_patches*d_model)
        pred = self.head(enc)  # (B*C, pred_len)

        pred = pred.reshape(bsz, c_in, self.pred_len).permute(0, 2, 1).contiguous()  # (B, pred_len, C)
        pred = self.normalize_layers(pred, "denorm")
        return pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc)
        return None

