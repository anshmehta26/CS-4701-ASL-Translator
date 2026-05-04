"""Bidirectional LSTM classifier for variable-length landmark sequences.

The model:
  1. Projects per-frame landmark vectors into a hidden space (with input
     dropout, since landmark coordinates can be noisy).
  2. Runs a multi-layer biLSTM with packed sequences (so padding is
     completely ignored).
  3. Uses a length-aware masked mean-pool over time as the clip embedding.
  4. Maps the embedding to class logits.

This model also exposes a ``forward_logits_per_step`` method so it can be
trained with CTC loss when ``training.ctc_mode = true``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.4,
        input_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.input_dropout = nn.Dropout(input_dropout)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.input_act = nn.GELU()

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_size, num_classes)

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Returns the per-timestep hidden states (B, T, H*)."""
        x = self.input_dropout(x)
        x = self.input_act(self.input_proj(x))

        # Packing requires lengths on CPU and sequences sorted by length, but
        # PyTorch handles the sort internally with enforce_sorted=False.
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return out  # (B, T, H*)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Clip-level classification. Returns logits of shape (B, num_classes)."""
        out = self._encode(x, lengths)
        # Masked mean over time so padding never contributes to the embedding.
        T = out.size(1)
        mask = (torch.arange(T, device=out.device).unsqueeze(0)
                < lengths.to(out.device).unsqueeze(1))            # (B, T)
        mask_f = mask.float().unsqueeze(-1)
        summed = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def forward_logits_per_step(
        self, x: torch.Tensor, lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For CTC training: returns (per-step log-probs, lengths).

        Output shape is (T, B, C) with log-softmax applied, ready for
        ``nn.CTCLoss``.
        """
        out = self._encode(x, lengths)              # (B, T, H*)
        out = self.dropout(out)
        logits = self.classifier(out)               # (B, T, C)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.transpose(0, 1), lengths   # (T, B, C), (B,)
