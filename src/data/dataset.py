"""PyTorch Dataset for cached landmark sequences.

Each sample on disk is an ``.npz`` file containing a single key ``landmarks``
of shape ``(T, D)``. A manifest CSV defines the train/val/test splits and
labels.

A custom ``pad_collate`` function batches variable-length sequences with
padding and returns a per-sample length tensor so models can build masks.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augmentation import SequenceAugmenter


class LandmarkSequenceDataset(Dataset):
    """In-memory dataset of cached landmark sequences."""

    def __init__(
        self,
        manifest_csv: str | Path,
        landmarks_dir: str | Path,
        split: str,
        max_seq_len: int,
        min_seq_len: int = 4,
        augmenter: SequenceAugmenter | None = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.landmarks_dir = Path(landmarks_dir)
        self.split = split
        self.max_seq_len = int(max_seq_len)
        self.min_seq_len = int(min_seq_len)
        self.augmenter = augmenter

        self.samples: list[dict] = []
        with self.manifest_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                npz = self.landmarks_dir / f"{row['video_id']}.npz"
                if not npz.exists():
                    continue
                self.samples.append({
                    "video_id": row["video_id"],
                    "label": int(row["label"]),
                    "gloss": row["gloss"],
                    "npz": npz,
                })
        if not self.samples:
            raise RuntimeError(
                f"No samples found for split={split!r} in {self.manifest_csv}. "
                "Run scripts/extract_landmarks.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_seq(self, npz_path: Path) -> np.ndarray:
        with np.load(npz_path) as data:
            seq = data["landmarks"].astype(np.float32)
        return seq

    def _trim(self, seq: np.ndarray) -> np.ndarray:
        T = seq.shape[0]
        if T <= self.max_seq_len:
            return seq
        # Center crop. Random crop could be added as augmentation, but center
        # is more deterministic and works well in practice.
        start = (T - self.max_seq_len) // 2
        return seq[start:start + self.max_seq_len]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        meta = self.samples[idx]
        seq = self._load_seq(meta["npz"])
        if self.augmenter is not None:
            seq = self.augmenter(seq)
        seq = self._trim(seq)
        if seq.shape[0] < self.min_seq_len:
            # pad in time so the model still sees a usable clip
            pad = self.min_seq_len - seq.shape[0]
            seq = np.concatenate(
                [seq, np.repeat(seq[-1:], pad, axis=0)], axis=0,
            )
        return torch.from_numpy(seq), int(meta["label"]), meta["video_id"]


def pad_collate(batch: list[tuple[torch.Tensor, int, str]]):
    """Right-pads sequences to the longest in the batch.

    Returns:
        sequences: (B, T_max, D) float tensor
        lengths:   (B,) int tensor of true lengths
        labels:    (B,) long tensor
        ids:       list[str]
    """
    sequences, labels, ids = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    T_max = int(lengths.max().item())
    D = sequences[0].shape[1]
    out = torch.zeros((len(sequences), T_max, D), dtype=torch.float32)
    for i, s in enumerate(sequences):
        out[i, : s.shape[0]] = s
    return out, lengths, torch.tensor(labels, dtype=torch.long), list(ids)
