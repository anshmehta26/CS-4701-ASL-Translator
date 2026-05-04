"""Real-time webcam inference.

Pipeline per frame:
  1. Read a frame from the webcam.
  2. Run MediaPipe Hands once (cached — same as the training pipeline).
  3. Append the resulting landmark vector to a fixed-size deque (sliding window).
  4. Every ``stride`` frames, run the model on the current window contents,
     apply exponential smoothing on the predicted probabilities for stability,
     and overlay top-K predictions on the frame.

Latency is measured per inference step (model forward time + the slice of
preprocessing it triggered) so we can compare against the proposal's <300ms
budget.
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Deque

import numpy as np
import torch

from src.data.landmarks import LandmarkExtractor, NUM_LANDMARKS, COORDS_PER_LANDMARK
from src.data.landmarks import (
    _interpolate_missing as interpolate_missing,
    _moving_average as moving_average,
    _wrist_relative_normalize as wrist_relative_normalize,
)
from src.models import build_model
from src.utils import get_logger, load_config

log = get_logger(__name__)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RealtimePredictor:
    """Holds the model + a sliding window of landmark vectors."""

    def __init__(self, checkpoint_path: str | Path, cfg) -> None:
        self.cfg = cfg
        self.device = _device()
        ckpt = torch.load(Path(checkpoint_path), map_location=self.device)

        self.label_map: dict[str, int] = ckpt["label_map"]
        self.idx_to_gloss = {int(v): k for k, v in self.label_map.items()}
        self.num_classes = int(ckpt["num_classes"])
        self.input_dim = int(ckpt["input_dim"])
        model_name = str(ckpt["model_name"])

        self.model = build_model(model_name, self.input_dim, self.num_classes, cfg)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        self.window_size = int(cfg.realtime.window_size)
        self.stride = int(cfg.realtime.stride)
        self.alpha = float(cfg.realtime.smoothing_alpha)
        self.threshold = float(cfg.realtime.confidence_threshold)
        self.top_k = int(cfg.realtime.display_top_k)
        self.num_hands = int(cfg.preprocessing.num_hands)

        self.window: Deque[np.ndarray] = collections.deque(maxlen=self.window_size)
        self.frame_counter = 0
        self.smoothed_probs: np.ndarray | None = None
        self.last_latency_ms = 0.0

    # ------------------------------------------------------------------
    def _preprocess_window(self) -> np.ndarray:
        """Apply the same per-clip preprocessing as the training pipeline,
        but only to the current sliding window (cheap)."""
        seq = np.stack(self.window, axis=0)            # (T, D)

        if bool(self.cfg.preprocessing.interpolate_missing):
            seq = interpolate_missing(seq)
        else:
            seq = np.nan_to_num(seq, nan=0.0)

        if bool(self.cfg.preprocessing.wrist_relative):
            seq = np.stack(
                [wrist_relative_normalize(f, self.num_hands) for f in seq],
                axis=0,
            )

        seq = moving_average(seq, int(self.cfg.preprocessing.smoothing_window))
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        return seq.astype(np.float32)

    @torch.no_grad()
    def _infer(self, seq: np.ndarray) -> np.ndarray:
        """Returns class probabilities of shape ``(num_classes,)``."""
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)            # (1, T, D)
        lengths = torch.tensor([x.size(1)], dtype=torch.long, device=self.device)
        logits = self.model(x, lengths)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
        return probs

    # ------------------------------------------------------------------
    def push_landmarks(self, frame_features: np.ndarray) -> None:
        """Append one frame's landmark vector to the sliding window."""
        if frame_features.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected feature dim {self.input_dim}, got {frame_features.shape[0]}",
            )
        self.window.append(frame_features.astype(np.float32))
        self.frame_counter += 1

    def maybe_predict(self) -> tuple[list[tuple[str, float]], float] | None:
        """Run the model if conditions are met; otherwise return None.

        Returns ``(top_k_predictions, latency_ms)`` if a prediction was run,
        else ``None``.
        """
        if len(self.window) < max(8, self.window_size // 2):
            return None
        if self.frame_counter % self.stride != 0:
            return None

        t0 = time.perf_counter()
        seq = self._preprocess_window()
        probs = self._infer(seq)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Exponential smoothing of probabilities (stabilizes the displayed label
        # so it doesn't flicker from frame to frame).
        if self.smoothed_probs is None or self.smoothed_probs.shape != probs.shape:
            self.smoothed_probs = probs
        else:
            self.smoothed_probs = (
                self.alpha * self.smoothed_probs + (1.0 - self.alpha) * probs
            )

        top_idx = np.argsort(-self.smoothed_probs)[: self.top_k]
        top = [(self.idx_to_gloss[int(i)], float(self.smoothed_probs[int(i)]))
               for i in top_idx]
        return top, self.last_latency_ms


# ----------------------------------------------------------------------
# Webcam loop
# ----------------------------------------------------------------------
def run_webcam(checkpoint: str | Path, config_path: str | Path = "configs/config.yaml") -> None:
    """Open the webcam, run live inference, and display predictions."""
    import cv2

    cfg = load_config(config_path)
    extractor = LandmarkExtractor(cfg.mediapipe, cfg.preprocessing)
    extractor._ensure_open()                                 # noqa: SLF001
    predictor = RealtimePredictor(checkpoint, cfg)

    cam_idx = int(cfg.realtime.camera_index)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise IOError(f"Could not open webcam at index {cam_idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.realtime.display_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.realtime.display_height))

    log.info("Starting webcam loop. Press 'q' to quit.")
    fps_avg, fps_last_t = 0.0, time.perf_counter()
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                log.error("Failed to read from webcam.")
                break
            frame_bgr = cv2.flip(frame_bgr, 1)               # mirror for natural feel
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            feat, _ = extractor._frame_to_features(rgb)      # noqa: SLF001
            predictor.push_landmarks(feat)

            result = predictor.maybe_predict()

            # ---- overlay -------------------------------------------------
            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, now - fps_last_t)
            fps_avg = 0.9 * fps_avg + 0.1 * inst_fps if fps_avg > 0 else inst_fps
            fps_last_t = now

            cv2.putText(frame_bgr, f"FPS: {fps_avg:5.1f}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Latency: {predictor.last_latency_ms:5.1f} ms",
                        (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if result is not None:
                top, _lat = result
                y = 100
                for i, (gloss, prob) in enumerate(top):
                    color = (0, 255, 0) if i == 0 and prob >= predictor.threshold else (200, 200, 200)
                    cv2.putText(
                        frame_bgr,
                        f"{i+1}. {gloss}  {prob*100:5.1f}%",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    )
                    y += 32

            cv2.imshow("ASL Real-Time", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
