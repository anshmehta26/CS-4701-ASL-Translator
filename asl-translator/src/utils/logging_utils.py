"""Lightweight logging setup. Use ``get_logger(__name__)`` everywhere."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_INITIALIZED = False


def _init_root(level: int = logging.INFO, log_file: str | None = None) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _INITIALIZED = True


def get_logger(name: str, level: int = logging.INFO,
               log_file: str | None = None) -> logging.Logger:
    _init_root(level=level, log_file=log_file)
    return logging.getLogger(name)
