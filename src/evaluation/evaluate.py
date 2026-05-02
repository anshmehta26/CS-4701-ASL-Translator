"""Evaluate a trained checkpoint on the held-out test split.

Computes:
  * Top-1 / Top-5 accuracy
  * Per-class accuracy
  * Full confusion matrix (saved as PNG and CSV)
  * The 10 most-confused class pairs (printed)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import LandmarkSequenceDataset, pad_collate
from src.models import build_model
from src.training.metrics import confusion_matrix, topk_accuracy
from src.utils import get_logger

log = get_logger(__name__)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(checkpoint: Path, cfg, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model(
        ckpt["model_name"], int(ckpt["input_dim"]), int(ckpt["num_classes"]), cfg,
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def _save_confusion_matrix_png(cm: np.ndarray, class_names: list[str],
                               out_path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping confusion matrix figure.")
        return
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * n), max(6, 0.3 * n)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix to %s", out_path)


def _save_confusion_matrix_csv(cm: np.ndarray, class_names: list[str],
                               out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, row in enumerate(cm):
            w.writerow([class_names[i]] + row.tolist())
    log.info("Saved confusion matrix CSV to %s", out_path)


def _top_confused_pairs(cm: np.ndarray, class_names: list[str], k: int = 10) -> list[dict]:
    n = cm.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append({"true": class_names[i], "pred": class_names[j],
                          "count": int(cm[i, j])})
    pairs.sort(key=lambda r: r["count"], reverse=True)
    return pairs[:k]


@torch.no_grad()
def evaluate(checkpoint_path: str | Path, cfg, split: str = "test") -> dict:
    device = _device()
    checkpoint_path = Path(checkpoint_path)
    model, ckpt = _load_model(checkpoint_path, cfg, device)

    label_map: dict[str, int] = ckpt["label_map"]
    class_names = [g for g, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    num_classes = len(class_names)

    manifest = Path(cfg.paths.processed) / "manifest.csv"
    landmarks = Path(cfg.paths.landmarks)
    ds = LandmarkSequenceDataset(
        manifest, landmarks, split,
        max_seq_len=int(cfg.dataset.max_seq_len),
        min_seq_len=int(cfg.dataset.min_seq_len),
        augmenter=None,
    )
    loader = DataLoader(
        ds, batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.num_workers), collate_fn=pad_collate,
    )

    all_logits, all_labels = [], []
    for x, lengths, labels, _ in loader:
        logits = model(x.to(device), lengths.to(device))
        all_logits.append(logits.cpu()); all_labels.append(labels)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    accs = topk_accuracy(logits, labels, ks=(1, 5))
    preds = logits.argmax(dim=1).numpy()
    labels_np = labels.numpy()

    cm = confusion_matrix(preds, labels_np, num_classes=num_classes)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    fig_dir = Path(cfg.paths.figures); fig_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.paths.logs); log_dir.mkdir(parents=True, exist_ok=True)
    name = checkpoint_path.stem
    _save_confusion_matrix_png(cm, class_names,
                               fig_dir / f"{name}_confusion_{split}.png",
                               f"{name} confusion matrix ({split})")
    _save_confusion_matrix_csv(cm, class_names,
                               log_dir / f"{name}_confusion_{split}.csv")

    top_confused = _top_confused_pairs(cm, class_names, k=10)
    log.info("=== Evaluation Results: %s on %s split ===", name, split)
    log.info("Top-1 accuracy: %.4f", accs[1])
    log.info("Top-5 accuracy: %.4f", accs[5])
    log.info("Top-10 most confused pairs:")
    for r in top_confused:
        log.info("  %s -> %s : %d", r["true"], r["pred"], r["count"])

    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "num_samples": int(labels.numel()),
        "top1": float(accs[1]),
        "top5": float(accs[5]),
        "per_class_accuracy": {class_names[i]: float(per_class_acc[i]) for i in range(num_classes)},
        "top_confused_pairs": top_confused,
    }
    summary_path = log_dir / f"{name}_eval_{split}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved evaluation summary to %s", summary_path)
    return summary
