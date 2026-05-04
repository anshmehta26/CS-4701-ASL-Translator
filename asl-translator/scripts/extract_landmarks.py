"""Run MediaPipe across every video in the manifest and cache landmark
sequences as ``.npz`` files under ``data/processed/landmarks/``.

This is the slowest preprocessing stage. It only needs to run once.

Usage::

    python scripts/extract_landmarks.py [--config configs/config.yaml] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.landmarks import LandmarkExtractor    # noqa: E402
from src.utils import get_logger, load_config       # noqa: E402

log = get_logger(__name__)


def _parse_bbox(s: str) -> tuple[int, int, int, int] | None:
    if not s:
        return None
    parts = s.split(",")
    if len(parts) != 4:
        return None
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N samples (for smoke tests).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if output .npz already exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = Path(cfg.paths.processed) / "manifest.csv"
    if not manifest.exists():
        log.error("Manifest not found: %s. Run scripts/prepare_dataset.py first.",
                  manifest)
        return 1

    out_dir = Path(cfg.paths.landmarks); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    log.info("Extracting landmarks for %d videos -> %s", len(rows), out_dir)

    n_done = n_skip = n_fail = 0
    t0 = time.time()
    with LandmarkExtractor(cfg.mediapipe, cfg.preprocessing) as extractor:
        for i, row in enumerate(rows, start=1):
            vid = row["video_id"]
            out_path = out_dir / f"{vid}.npz"
            if out_path.exists() and not args.overwrite:
                n_skip += 1
                continue
            try:
                bbox = _parse_bbox(row.get("bbox", "") or "")
                result = extractor.extract_video(
                    row["video_path"],
                    frame_start=int(row["frame_start"]),
                    frame_end=int(row["frame_end"]),
                    bbox=bbox,
                )
                if result is None:
                    n_fail += 1
                    continue
                np.savez_compressed(out_path, landmarks=result.landmarks)
                n_done += 1
            except Exception as e:                      # noqa: BLE001
                log.warning("Failed on video %s: %s", vid, e)
                n_fail += 1

            if i % 25 == 0 or i == len(rows):
                dt = time.time() - t0
                log.info(
                    "[%d/%d] done=%d skipped=%d failed=%d (%.1f vid/s)",
                    i, len(rows), n_done, n_skip, n_fail, i / max(dt, 1e-6),
                )

    log.info("Finished. done=%d skipped=%d failed=%d", n_done, n_skip, n_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
