"""Benchmark per-prediction latency of a trained checkpoint.

Times only the forward pass (not webcam I/O or MediaPipe). Synthetic input is
used so the benchmark can run on any machine. Results are printed and saved
to ``logs/{ckpt_name}_latency.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.models import build_model
from src.utils import get_logger

log = get_logger(__name__)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def benchmark(checkpoint_path: str | Path, cfg, num_runs: int = 100,
              warmup: int = 10) -> dict:
    device = _device()
    ckpt = torch.load(Path(checkpoint_path), map_location=device)
    model = build_model(
        ckpt["model_name"], int(ckpt["input_dim"]), int(ckpt["num_classes"]), cfg,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    T = int(cfg.realtime.window_size)
    D = int(ckpt["input_dim"])
    x = torch.randn(1, T, D, dtype=torch.float32, device=device)
    lengths = torch.tensor([T], dtype=torch.long, device=device)

    # Warmup — first forward passes are slower on most backends.
    for _ in range(warmup):
        _ = model(x, lengths)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model(x, lengths)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    summary = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "model": ckpt["model_name"],
        "window_size": T,
        "feature_dim": D,
        "num_runs": num_runs,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "target_ms": float(cfg.realtime.target_latency_ms),
        "meets_target": bool(np.percentile(arr, 95) <= float(cfg.realtime.target_latency_ms)),
    }
    log.info("Latency for %s on %s:", Path(checkpoint_path).name, device)
    log.info("  mean   %.2f ms", summary["mean_ms"])
    log.info("  median %.2f ms", summary["median_ms"])
    log.info("  p95    %.2f ms (target %.0f ms)", summary["p95_ms"], summary["target_ms"])
    log.info("  meets <%.0fms target: %s", summary["target_ms"], summary["meets_target"])

    log_dir = Path(cfg.paths.logs); log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{Path(checkpoint_path).stem}_latency.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved latency report to %s", out)
    return summary
