"""Tests for the dataset and pad_collate function."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import LandmarkSequenceDataset, pad_collate


def _build_tiny_dataset(tmp_root: Path):
    landmarks_dir = tmp_root / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(6):
        T = 10 + i * 2
        seq = np.random.RandomState(i).randn(T, 12).astype(np.float32)
        np.savez_compressed(landmarks_dir / f"vid{i}.npz", landmarks=seq)
        rows.append({
            "video_id": f"vid{i}", "gloss": f"g{i % 3}",
            "label": i % 3, "split": "train" if i < 4 else "val",
            "signer_id": -1, "bbox": "",
            "frame_start": -1, "frame_end": -1, "video_path": "x",
        })
    manifest = tmp_root / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with (tmp_root / "label_map.json").open("w") as f:
        json.dump({"g0": 0, "g1": 1, "g2": 2}, f)
    return manifest, landmarks_dir


def test_dataset_loads_split(tmp_path):
    manifest, landmarks_dir = _build_tiny_dataset(tmp_path)
    ds = LandmarkSequenceDataset(
        manifest, landmarks_dir, "train", max_seq_len=32, min_seq_len=2,
    )
    assert len(ds) == 4
    seq, label, vid = ds[0]
    assert isinstance(seq, torch.Tensor)
    assert seq.dim() == 2 and seq.shape[1] == 12
    assert isinstance(label, int)
    assert vid.startswith("vid")


def test_pad_collate_pads_and_returns_lengths(tmp_path):
    manifest, landmarks_dir = _build_tiny_dataset(tmp_path)
    ds = LandmarkSequenceDataset(manifest, landmarks_dir, "train",
                                 max_seq_len=32, min_seq_len=2)
    batch = [ds[i] for i in range(len(ds))]
    seqs, lengths, labels, ids = pad_collate(batch)
    assert seqs.shape[0] == len(batch)
    assert seqs.shape[2] == 12
    assert seqs.shape[1] == int(lengths.max().item())
    assert labels.shape == (len(batch),)
    assert len(ids) == len(batch)


def test_dataset_trims_long_sequences(tmp_path):
    manifest, landmarks_dir = _build_tiny_dataset(tmp_path)
    ds = LandmarkSequenceDataset(manifest, landmarks_dir, "train",
                                 max_seq_len=8, min_seq_len=2)
    seq, _, _ = ds[0]
    assert seq.shape[0] <= 8


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        tp = Path(t)
        test_dataset_loads_split(tp)
    with tempfile.TemporaryDirectory() as t:
        test_pad_collate_pads_and_returns_lengths(Path(t))
    with tempfile.TemporaryDirectory() as t:
        test_dataset_trims_long_sequences(Path(t))
    print("All dataset tests passed.")
