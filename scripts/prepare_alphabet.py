"""Build a manifest from the Kaggle ASL Alphabet dataset AND extract
MediaPipe landmarks for every image, all in one go.

Unlike WLASL (where extraction is slow and worth caching separately), each
alphabet sample is a single image — so we just walk the image tree, run
MediaPipe per image, and save the resulting 1-frame landmark vectors.

After this finishes, ``data/processed/manifest.csv`` and
``data/processed/landmarks/{image_id}.npz`` will be ready for training:

    python scripts/prepare_alphabet.py
    python scripts/train.py --model lstm

Useful flags:
    --cap-per-class 500        keep only N images per letter (default: 500)
    --no-special               skip "space", "del", "nothing" classes
    --train-root path/to/dir   override location of the alphabet folders
    --limit N                  smoke test: only process N total samples

The Kaggle ZIP unpacks to a nested structure where the train images live in
either ``asl_alphabet_train/asl_alphabet_train/`` or just
``asl_alphabet_train/``. We auto-detect and unwrap one extra level.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.alphabet import build_alphabet_manifest               # noqa: E402
from src.data.landmarks import LandmarkExtractor                    # noqa: E402
from src.utils import get_logger, load_config, seed_everything      # noqa: E402

log = get_logger(__name__)


def _resolve_train_root(default_root: Path) -> Path:
    """Handle the doubly-nested layout of the Kaggle ZIP.

    If ``default_root`` itself contains class folders (A, B, C, ...), use it.
    Otherwise look one level deeper for ``asl_alphabet_train``.
    """
    if not default_root.exists():
        # Try common alternatives
        for alt in (
            default_root.parent / "asl_alphabet_train" / "asl_alphabet_train",
            default_root.parent / "asl_alphabet_train",
            default_root / "asl_alphabet_train",
        ):
            if alt.exists() and any(c.is_dir() for c in alt.iterdir()):
                return alt
        return default_root  # let the loader raise a clean error

    # If the given root contains exactly one folder named asl_alphabet_train,
    # descend into it.
    children = [c for c in default_root.iterdir() if c.is_dir()]
    if (len(children) == 1 and children[0].name.lower().startswith("asl_alphabet_train")
            and any(g.is_dir() for g in children[0].iterdir())):
        return children[0]
    return default_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--train-root", default=None,
        help="Path to the alphabet training folder (overrides config).",
    )
    parser.add_argument("--cap-per-class", type=int, default=500,
                        help="Max images per letter (default: 500).")
    parser.add_argument("--no-special", action="store_true",
                        help="Skip the space / del / nothing classes.")
    parser.add_argument("--limit", type=int, default=None,
                        help="For smoke tests: process only N samples total.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.training.seed))

    if args.train_root:
        train_root = Path(args.train_root)
    else:
        # Default location for the Kaggle dataset after extraction.
        train_root = Path(cfg.paths.data_root) / "raw" / "asl_alphabet_train"
    train_root = _resolve_train_root(train_root)
    log.info("Reading alphabet images from: %s", train_root)

    samples, label_map = build_alphabet_manifest(
        train_root=train_root,
        cap_per_class=args.cap_per_class,
        train_ratio=float(cfg.dataset.train_ratio),
        val_ratio=float(cfg.dataset.val_ratio),
        test_ratio=float(cfg.dataset.test_ratio),
        seed=int(cfg.training.seed),
        include_special=not args.no_special,
    )
    if args.limit:
        samples = samples[: args.limit]
        log.info("Truncated to %d samples for smoke test.", len(samples))

    processed = Path(cfg.paths.processed)
    landmarks_dir = Path(cfg.paths.landmarks)
    processed.mkdir(parents=True, exist_ok=True)
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run MediaPipe over every image ----
    counts = {"train": 0, "val": 0, "test": 0}
    n_done = n_skip = n_fail = 0
    kept_samples = []
    t0 = time.time()
    with LandmarkExtractor(cfg.mediapipe, cfg.preprocessing) as extractor:
        for i, s in enumerate(samples, start=1):
            out_path = landmarks_dir / f"{s.image_id}.npz"
            if out_path.exists() and not args.overwrite:
                kept_samples.append(s); counts[s.split] += 1
                n_skip += 1
                continue
            try:
                result = extractor.extract_image(s.image_path)
            except Exception as e:                          # noqa: BLE001
                log.warning("Failed on %s: %s", s.image_path, e)
                result = None
            if result is None:
                n_fail += 1
                continue
            np.savez_compressed(out_path, landmarks=result.landmarks)
            kept_samples.append(s)
            counts[s.split] += 1
            n_done += 1

            if i % 500 == 0 or i == len(samples):
                dt = time.time() - t0
                log.info(
                    "[%d/%d] kept=%d skipped=%d failed=%d (%.1f img/s)",
                    i, len(samples), n_done + n_skip, n_skip, n_fail,
                    i / max(dt, 1e-6),
                )

    log.info("MediaPipe extraction done: kept=%d failed=%d (%.0f%% success).",
             n_done + n_skip, n_fail,
             100.0 * (n_done + n_skip) / max(1, n_done + n_skip + n_fail))
    log.info("Split counts (after dropping failed detections): %s", counts)

    if not kept_samples:
        log.error("No samples kept. Check dataset path and MediaPipe install.")
        return 1

    # Re-index labels in case some classes were entirely dropped (unlikely
    # but possible if MediaPipe failed on every image of some class).
    used_glosses = sorted({s.gloss for s in kept_samples})
    new_label_map = {g: i for i, g in enumerate(used_glosses)}
    if len(new_label_map) != len(label_map):
        log.warning(
            "Re-indexed labels: %d -> %d classes (some classes had no "
            "successful detections).",
            len(label_map), len(new_label_map),
        )
    for s in kept_samples:
        s.label = new_label_map[s.gloss]

    # Write manifest + label map (same schema as WLASL pipeline)
    fields = [
        "video_id", "gloss", "label", "split", "signer_id",
        "bbox", "frame_start", "frame_end", "video_path",
    ]
    with (processed / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in kept_samples:
            w.writerow(s.to_row())
    with (processed / "label_map.json").open("w") as f:
        json.dump(new_label_map, f, indent=2)
    log.info(
        "Wrote manifest (%d samples) and label_map (%d classes) to %s",
        len(kept_samples), len(new_label_map), processed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
