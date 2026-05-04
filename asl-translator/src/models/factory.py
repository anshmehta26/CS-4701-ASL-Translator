"""Factory: build a model by name from a config object."""

from __future__ import annotations

import torch.nn as nn

from .lstm import LSTMClassifier
from .transformer import TransformerClassifier


def build_model(name: str, input_dim: int, num_classes: int, cfg) -> nn.Module:
    """Build either the LSTM or Transformer model.

    Args:
        name: "lstm" or "transformer"
        input_dim: dimensionality of one frame's landmark vector
        num_classes: number of glosses
        cfg: top-level config (dot-accessible). Reads ``cfg.lstm`` or
             ``cfg.transformer`` depending on ``name``.
    """
    name = name.lower()
    if name == "lstm":
        m = LSTMClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_size=int(cfg.lstm.hidden_size),
            num_layers=int(cfg.lstm.num_layers),
            bidirectional=bool(cfg.lstm.bidirectional),
            dropout=float(cfg.lstm.dropout),
            input_dropout=float(cfg.lstm.input_dropout),
        )
    elif name == "transformer":
        m = TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=int(cfg.transformer.d_model),
            nhead=int(cfg.transformer.nhead),
            num_encoder_layers=int(cfg.transformer.num_encoder_layers),
            dim_feedforward=int(cfg.transformer.dim_feedforward),
            dropout=float(cfg.transformer.dropout),
            max_position_embeddings=int(cfg.transformer.max_position_embeddings),
        )
    else:
        raise ValueError(f"Unknown model name: {name!r} (use 'lstm' or 'transformer')")
    return m


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
