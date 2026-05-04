"""Config loading utilities.

Loads YAML configs into a dot-accessible namespace so call sites can write
``cfg.training.batch_size`` instead of ``cfg["training"]["batch_size"]``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


class _DotDict(SimpleNamespace):
    """SimpleNamespace that also supports dict-style access."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return _ns_to_dict(self)


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return _DotDict(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _ns_to_dict(obj: Any) -> Any:
    if isinstance(obj, _DotDict):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_ns_to_dict(v) for v in obj]
    return obj


def load_config(path: str | Path = "configs/config.yaml") -> _DotDict:
    """Load a YAML config and return it as a dot-accessible object."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)
