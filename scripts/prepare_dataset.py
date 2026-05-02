"""Build the project's video manifest from WLASL_v0.3.json.

Usage::

    python scripts/prepare_dataset.py [--config configs/config.yaml]

This:
  1. Reads the WLASL JSON.
  2. Picks the top-N most frequent glosses with enough instances.
  3. Filters to videos that actually exist on disk.
  4. Reassigns splits so each split is non-empty even on small subsets
     (we *replace* WLASL's split tags using a stratified random split).
  5. Writes ``data/processed/manifest.csv`` and ``label_map.json``.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Add the project root to sys.path so ``import src`` works when this script is
# run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.wlasl import build_manifest, save_label_map, save_manifest  # noqa: E402
from src.utils import get_logger, load_config, seed_everything             # noqa: E402

log = get_logger(__name__)


def stratified_split(samples, train_r: float, val_r: float, test_r: float, seed: int):
    """Stratify on label so each split contains every class (when possible)."""
    if abs((train_r + val_r + test_r) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    rng = random.Random(seed)
    by_label: dict[int, list] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    for s in samples:
        s.split = "train"  # default; overwritten below

    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        if n < 3:
            # Not enough to split; keep all in train and warn.
            log.warning(
                "Class label=%d has only %d samples; assigning all to train.",
                label, n,
            )
            for s in items:
                s.split = "train"
            continue
        n_test = max(1, int(round(n * test_r)))
        n_val = max(1, int(round(n * val_r)))
        n_train = n - n_val - n_test
        if n_train < 1:
            # Pull one back into train if rounding squeezed it out.
            n_train = 1
            n_val = max(1, n - n_train - n_test)
        for s in items[:n_train]:
            s.split = "train"
        for s in items[n_train:n_train + n_val]:
            s.split = "val"
        for s in items[n_train + n_val:]:
            s.split = "test"
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.training.seed))

    wlasl_json = Path(cfg.paths.wlasl_json)
    raw_videos = Path(cfg.paths.raw_videos)
    processed = Path(cfg.paths.processed)

    samples, label_map = build_manifest(
        wlasl_json=wlasl_json,
        raw_videos=raw_videos,
        num_classes=int(cfg.dataset.num_classes),
        min_videos=int(cfg.dataset.min_videos_per_class),
    )
    if not samples:
        log.error(
            "No samples could be built. Place videos under %s and ensure "
            "%s exists.", raw_videos, wlasl_json,
        )
        return 1

    samples = stratified_split(
        samples,
        train_r=float(cfg.dataset.train_ratio),
        val_r=float(cfg.dataset.val_ratio),
        test_r=float(cfg.dataset.test_ratio),
        seed=int(cfg.training.seed),
    )

    counts = {"train": 0, "val": 0, "test": 0}
    for s in samples:
        counts[s.split] += 1
    log.info("Split counts: %s", counts)

    save_manifest(samples, processed / "manifest.csv")
    save_label_map(label_map, processed / "label_map.json")
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
