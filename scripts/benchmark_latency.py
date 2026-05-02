"""Benchmark inference latency.

Usage::

    python scripts/benchmark_latency.py --checkpoint checkpoints/lstm_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import benchmark                         # noqa: E402
from src.utils import load_config                            # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmark(args.checkpoint, cfg, num_runs=args.num_runs, warmup=args.warmup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
