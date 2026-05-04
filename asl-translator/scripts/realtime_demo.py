"""Run the real-time webcam demo.

Usage::

    python scripts/realtime_demo.py --checkpoint checkpoints/lstm_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import run_webcam     # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    run_webcam(args.checkpoint, args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
