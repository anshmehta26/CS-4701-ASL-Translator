"""Train a model.

Usage::

    python scripts/train.py --model lstm
    python scripts/train.py --model transformer --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import Trainer                # noqa: E402
from src.utils import load_config, get_logger    # noqa: E402

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--ctc", action="store_true",
                        help="Train with CTC loss instead of cross-entropy.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ctc:
        cfg.training.ctc_mode = True
        log.info("CTC mode enabled via CLI.")

    trainer = Trainer(cfg, model_name=args.model)
    result = trainer.run()
    log.info("Training complete. Best val top-1: %.4f", result["best_val_top1"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
