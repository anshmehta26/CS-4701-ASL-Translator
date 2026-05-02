"""Generate synthetic landmark data so the rest of the pipeline can be smoke-
tested without needing the full WLASL video corpus.

Use this when you want to verify the training/eval/realtime code paths but
either don't yet have the video files downloaded or want a quick sanity run.

For each "fake" gloss we generate ``samples_per_class`` clips of random length.
Each clip has a class-specific signature embedded in the landmark coordinates
so the model can actually learn to discriminate. This is *only* a development
aid — real results should always use real WLASL data.

Usage::

    python scripts/make_synthetic_data.py --classes 10 --per-class 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, seed_everything   # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--min-T", type=int, default=20)
    parser.add_argument("--max-T", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.training.seed))
    rng = np.random.default_rng(int(cfg.training.seed))

    num_hands = int(cfg.preprocessing.num_hands)
    D = num_hands * 21 * 3

    out_root = Path(cfg.paths.processed)
    landmarks_dir = Path(cfg.paths.landmarks)
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    # Each class gets its own "signature" — a fixed direction in feature space
    # added to every clip of that class. The model must learn to detect it
    # despite added noise / random base motion.
    signatures = rng.normal(0.0, 0.5, size=(args.classes, D)).astype(np.float32)

    glosses = [f"sign_{i:02d}" for i in range(args.classes)]
    label_map = {g: i for i, g in enumerate(glosses)}

    rows = []
    for cls_id, gloss in enumerate(glosses):
        for k in range(args.per_class):
            T = int(rng.integers(args.min_T, args.max_T + 1))
            base = rng.normal(0.0, 0.05, size=(T, D)).astype(np.float32)
            # Fade the class signature in/out across the clip so the model
            # has to integrate over time, not just look at one frame.
            ramp = np.sin(np.linspace(0, np.pi, T)).astype(np.float32)[:, None]
            seq = base + ramp * signatures[cls_id]
            vid_id = f"syn_{cls_id:02d}_{k:03d}"
            np.savez_compressed(landmarks_dir / f"{vid_id}.npz", landmarks=seq)
            # Stratified train/val/test by index modulo
            split = "train" if k % 5 < 3 else ("val" if k % 5 == 3 else "test")
            rows.append({
                "video_id": vid_id, "gloss": gloss, "label": cls_id,
                "split": split, "signer_id": -1, "bbox": "",
                "frame_start": -1, "frame_end": -1,
                "video_path": "synthetic",
            })

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with (out_root / "label_map.json").open("w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Generated synthetic dataset: {args.classes} classes, "
          f"{args.classes * args.per_class} clips. Manifest at {out_root}/manifest.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
