"""Lightweight metric utilities used during training and evaluation."""

from __future__ import annotations

import numpy as np
import torch


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                  ks: tuple[int, ...] = (1, 5)) -> dict[int, float]:
    """Returns ``{k: accuracy}`` for each k in ``ks``."""
    if logits.numel() == 0:
        return {k: 0.0 for k in ks}
    max_k = min(max(ks), logits.size(1))
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)  # (B, max_k)
    correct = pred.eq(labels.view(-1, 1))                            # (B, max_k)
    out: dict[int, float] = {}
    for k in ks:
        kk = min(k, max_k)
        out[k] = float(correct[:, :kk].any(dim=1).float().mean().item())
    return out


def confusion_matrix(preds: np.ndarray, labels: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """Standard CxC confusion matrix; rows are true labels, cols are predictions."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


class RunningAverage:
    """Track a running mean of a scalar across a training run."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def value(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
