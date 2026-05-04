"""Tests that the models forward correctly on variable-length input."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import LSTMClassifier, TransformerClassifier


def test_lstm_forward_shapes():
    B, T, D, C = 4, 12, 126, 5
    model = LSTMClassifier(input_dim=D, num_classes=C,
                           hidden_size=32, num_layers=2)
    x = torch.randn(B, T, D)
    lengths = torch.tensor([12, 8, 10, 5])
    out = model(x, lengths)
    assert out.shape == (B, C)
    assert torch.isfinite(out).all()


def test_lstm_forward_per_step():
    B, T, D, C = 3, 10, 126, 5
    model = LSTMClassifier(input_dim=D, num_classes=C, hidden_size=32, num_layers=1)
    x = torch.randn(B, T, D)
    lengths = torch.tensor([10, 7, 4])
    log_probs, lens = model.forward_logits_per_step(x, lengths)
    # CTC expects (T, B, C)
    assert log_probs.shape == (T, B, C)
    assert torch.allclose(log_probs.exp().sum(dim=-1),
                          torch.ones_like(log_probs.sum(dim=-1)), atol=1e-4)
    assert lens.tolist() == [10, 7, 4]


def test_transformer_forward_shapes():
    B, T, D, C = 4, 16, 126, 7
    model = TransformerClassifier(
        input_dim=D, num_classes=C,
        d_model=64, nhead=4, num_encoder_layers=2,
        dim_feedforward=128, max_position_embeddings=64,
    )
    x = torch.randn(B, T, D)
    lengths = torch.tensor([16, 16, 8, 4])
    out = model(x, lengths)
    assert out.shape == (B, C)
    assert torch.isfinite(out).all()


def test_transformer_handles_short_sequence():
    """A length-2 sequence shouldn't crash the encoder."""
    B, D, C = 2, 126, 4
    model = TransformerClassifier(
        input_dim=D, num_classes=C,
        d_model=32, nhead=4, num_encoder_layers=1,
        dim_feedforward=64, max_position_embeddings=32,
    )
    x = torch.randn(B, 4, D)
    lengths = torch.tensor([2, 4])
    out = model(x, lengths)
    assert out.shape == (B, C)


if __name__ == "__main__":
    test_lstm_forward_shapes()
    test_lstm_forward_per_step()
    test_transformer_forward_shapes()
    test_transformer_handles_short_sequence()
    print("All model tests passed.")
