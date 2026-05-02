"""Transformer encoder classifier for landmark sequences.

This is the comparison architecture mentioned in the proposal. It uses:
  * A linear projection from landmark_dim -> d_model.
  * Sinusoidal position encodings (cheap and length-flexible).
  * An ``nn.TransformerEncoder`` with ``batch_first=True`` and proper
    ``src_key_padding_mask`` so padded positions are ignored.
  * Masked mean-pool over time -> classification head.

It also implements ``forward_logits_per_step`` for optional CTC training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 192,
        nhead: int = 6,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_position_embeddings: int = 128,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = _SinusoidalPE(d_model, max_len=max_position_embeddings * 8)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,        # pre-norm trains more stably
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    # ------------------------------------------------------------------
    def _build_pad_mask(self, T: int, lengths: torch.Tensor,
                       device: torch.device) -> torch.Tensor:
        """Returns a (B, T) bool tensor where True marks positions to ignore."""
        ar = torch.arange(T, device=device).unsqueeze(0)
        return ar >= lengths.to(device).unsqueeze(1)

    def _encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pe(x)
        pad_mask = self._build_pad_mask(x.size(1), lengths, x.device)
        # If a row would be entirely masked (e.g. length 0), Transformer can
        # produce NaNs. Our dataset enforces a minimum length, so pad_mask
        # never has all-True rows in practice.
        return self.encoder(x, src_key_padding_mask=pad_mask)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        out = self._encode(x, lengths)                    # (B, T, d_model)
        T = out.size(1)
        mask = (torch.arange(T, device=out.device).unsqueeze(0)
                < lengths.to(out.device).unsqueeze(1))    # (B, T)
        mask_f = mask.float().unsqueeze(-1)
        summed = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom
        return self.classifier(self.dropout(pooled))

    def forward_logits_per_step(
        self, x: torch.Tensor, lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._encode(x, lengths)
        logits = self.classifier(self.dropout(out))       # (B, T, C)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.transpose(0, 1), lengths
