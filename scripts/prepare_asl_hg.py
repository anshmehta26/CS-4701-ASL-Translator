"""Build manifest + extract MediaPipe landmarks from ASL-HG.

ASL-HG contains 10 signers (P1-P10), each contributing 100 images per class.
This script splits them by signer (configurable via CLI flags) and runs
MediaPipe over every image, caching the resulting 1-frame landmark vectors.

Default split (mirrors what most multi-signer ASL papers use):
  train: P1-P6 (6 signers)
  val:   P7-P8 (2 signers)
  test:  P9-P10 (2 signers)

Run:
    python scripts/prepare_asl_hg.py --config configs/asl_hg.yaml
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

from src.data.asl_hg import build_asl_hg_manifest                  # noqa: E402
from src.data.landmarks import LandmarkExtractor                   # noqa: E402
from src.utils import get_logger, load_config, seed_everything     # noqa: E402

log = get_logger(__name__)


def _parse_signer_list(arg: str) -> list[str]:
    return [s.strip().upper() for s in arg.split(",") if s.strip()]


def _resolve_dataset_root(default_root: Path) -> Path:
    """ASL-HG unzips into ugly nested folders. Find ``asl_dataset`` for the user."""
    candidates = [
        default_root,
        default_root / "asl_dataset",
    ]
    # Recursive search up to 4 levels deep for a directory literally named asl_dataset.
    for d in default_root.rglob("asl_dataset"):
        if d.is_dir() and any(c.is_dir() for c in d.iterdir()):
            candidates.insert(0, d)
            break
    for c in candidates:
        if c.exists() and c.is_dir():
            children = [p for p in c.iterdir() if p.is_dir()]
            if any(p.name.upper() in {"A", "B", "C", "D", "E", "F"} for p in children):
                return c
    return default_root  # let the loader raise a clean error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/asl_hg.yaml")
    parser.add_argument(
        "--dataset-root", default=None,
        help="Path to the asl_dataset folder (overrides config).",
    )
    parser.add_argument(
        "--train-signers", default="P1,P2,P3,P4,P5,P6",
        help="Comma-separated list of signers for the train split.",
    )
    parser.add_argument(
        "--val-signers", default="P7,P8",
        help="Comma-separated list of signers for the validation split.",
    )
    parser.add_argument(
        "--test-signers", default="P9,P10",
        help="Comma-separated list of signers for the test split.",
    )
    parser.add_argument("--include-digits", action="store_true",
                        help="Include 0-9 classes (default: letters only).")
    parser.add_argument("--limit", type=int, default=None,
                        help="For smoke tests: process only N samples total.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.training.seed))

    # ---- locate the dataset ----
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = Path(cfg.paths.data_root) / "raw" / "asl-hg"
    dataset_root = _resolve_dataset_root(dataset_root)
    log.info("Reading ASL-HG images from: %s", dataset_root)

    train_s = _parse_signer_list(args.train_signers)
    val_s = _parse_signer_list(args.val_signers)
    test_s = _parse_signer_list(args.test_signers)
    log.info("Signer split: train=%s val=%s test=%s", train_s, val_s, test_s)

    samples, label_map = build_asl_hg_manifest(
        dataset_root=dataset_root,
        train_signers=train_s,
        val_signers=val_s,
        test_signers=test_s,
        include_digits=bool(args.include_digits),
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
                    "[%d/%d] kept=%d failed=%d (%.1f img/s)",
                    i, len(samples), n_done + n_skip, n_fail,
                    i / max(dt, 1e-6),
                )

    log.info("Per-split counts after extraction: %s", counts)
    if not kept_samples:
        log.error("No samples kept. Check dataset path and MediaPipe install.")
        return 1

    # ---- Write manifest + label map ----
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
        json.dump(label_map, f, indent=2)
    log.info(
        "Wrote manifest (%d samples) + label_map (%d classes) to %s",
        len(kept_samples), len(label_map), processed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
