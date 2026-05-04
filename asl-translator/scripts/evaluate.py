"""Evaluate a trained checkpoint on the test split.

Usage::

    python scripts/evaluate.py --checkpoint checkpoints/lstm_best.pt
    python scripts/evaluate.py --checkpoint checkpoints/transformer_best.pt --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import evaluate                          # noqa: E402
from src.utils import load_config, get_logger                # noqa: E402

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(args.checkpoint, cfg, split=args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
