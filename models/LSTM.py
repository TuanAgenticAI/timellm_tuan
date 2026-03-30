import torch
import torch.nn as nn

from layers.StandardNorm import Normalize


class Model(nn.Module):
    """
    Simple LSTM baseline for time series forecasting.

    Notes:
    - This implementation follows the repo's model interface:
      forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) -> (B, pred_len, C)
    - We run LSTM channel-independently (each feature/variable is modeled separately)
      to remain compatible with the repo's datasets/slicing logic.
    """

    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        hidden_size = configs.d_model
        num_layers = getattr(configs, "e_layers", 1)
        dropout = getattr(configs, "dropout", 0.0)

        # RevIN-style normalization (affine disabled to avoid parameter count mismatch).
        # This normalize layer infers statistics from the actual tensor shape at runtime.
        self.normalize_layers = Normalize(getattr(configs, "enc_in", 1), affine=False)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, self.pred_len)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc: (B, seq_len, C)
        Returns:
            (B, pred_len, C)
        """
        x_enc = self.normalize_layers(x_enc, "norm")
        bsz, seq_len, c_in = x_enc.shape

        # Channel-independent modeling:
        # (B, T, C) -> (B*C, T, 1)
        x = x_enc.permute(0, 2, 1).contiguous().reshape(bsz * c_in, seq_len, 1)

        out, _ = self.lstm(x)  # (B*C, T, H)
        last = out[:, -1, :]  # (B*C, H)
        last = self.dropout(last)
        pred = self.head(last)  # (B*C, pred_len)

        # (B*C, pred_len) -> (B, pred_len, C)
        pred = pred.reshape(bsz, c_in, self.pred_len).permute(0, 2, 1).contiguous()
        pred = self.normalize_layers(pred, "denorm")
        return pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc)

        # Other tasks are not implemented for this repo baseline.
        return None

