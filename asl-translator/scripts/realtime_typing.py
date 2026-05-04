"""Real-time fingerspelling-to-text demo.

Builds a typed string by watching the webcam:
  * Commit a letter to the typed string when the same prediction has been the
    top-1 for ``commit_frames`` consecutive predictions AND its smoothed
    confidence stays above ``confidence_threshold``.
  * "nothing" (no hand visible) acts as a letter separator — the user shows
    a letter, drops their hand briefly, then forms the next letter. This
    avoids accidentally typing "AAAAA" while holding one letter too long.
  * "space" gesture inserts a space; "del" removes the last character.
  * Press ``c`` to clear the buffer, ``q`` to quit.

This is built on top of ``RealtimePredictor`` so it inherits sliding-window
inference, exponential probability smoothing, and latency measurement.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.landmarks import LandmarkExtractor                 # noqa: E402
from src.inference.realtime import RealtimePredictor              # noqa: E402
from src.utils import load_config, get_logger                     # noqa: E402

log = get_logger(__name__)


class TypingState:
    """Tracks the typed buffer and the rolling history of top-1 predictions."""

    def __init__(self, commit_frames: int, threshold: float,
                 cooldown_predictions: int) -> None:
        self.commit_frames = commit_frames
        self.threshold = threshold
        # Don't commit two letters in quick succession — wait for either a
        # "nothing" gesture (preferred separator) or this many predictions to
        # pass after the last commit.
        self.cooldown_predictions = cooldown_predictions
        self.recent: deque[str] = deque(maxlen=commit_frames)
        self.text: str = ""
        self.preds_since_commit: int = 0
        self.last_committed: str | None = None

    def push(self, label: str, confidence: float) -> str | None:
        """Add a new top-1 prediction. Returns the letter just committed, or None."""
        self.preds_since_commit += 1
        # We require the model to be confident before we even consider this
        # prediction a "vote" — low-confidence frames just clear the buffer.
        if confidence < self.threshold:
            self.recent.clear()
            return None
        self.recent.append(label)

        # "nothing" resets state and acts as a separator (clears the cooldown
        # so the next confident letter can be committed immediately).
        if label == "nothing":
            self.recent.clear()
            self.preds_since_commit = self.cooldown_predictions
            self.last_committed = None
            return None

        if len(self.recent) < self.commit_frames:
            return None
        if any(r != label for r in self.recent):
            return None
        # Avoid double-committing the same letter unless we've cooled down or
        # seen a separator.
        if (label == self.last_committed
                and self.preds_since_commit < self.cooldown_predictions):
            return None

        # ---- commit ----
        self._apply(label)
        self.last_committed = label
        self.preds_since_commit = 0
        self.recent.clear()
        return label

    def _apply(self, label: str) -> None:
        if label == "del":
            self.text = self.text[:-1]
        elif label == "space":
            self.text = self.text + " "
        else:
            self.text = self.text + label

    def clear(self) -> None:
        self.text = ""
        self.recent.clear()
        self.preds_since_commit = 0
        self.last_committed = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/alphabet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--commit-frames", type=int, default=4,
                        help="Predictions in a row required to commit a letter.")
    parser.add_argument("--cooldown", type=int, default=8,
                        help="Predictions to wait before committing the same "
                             "letter again.")
    args = parser.parse_args()

    import cv2

    cfg = load_config(args.config)
    extractor = LandmarkExtractor(cfg.mediapipe, cfg.preprocessing)
    extractor._ensure_open()                                       # noqa: SLF001
    predictor = RealtimePredictor(args.checkpoint, cfg)
    typer = TypingState(
        commit_frames=args.commit_frames,
        threshold=float(cfg.realtime.confidence_threshold),
        cooldown_predictions=args.cooldown,
    )

    cam_idx = int(cfg.realtime.camera_index)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise IOError(f"Could not open webcam at index {cam_idx}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.realtime.display_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.realtime.display_height))

    log.info("Fingerspelling demo running. Keys: q=quit, c=clear, b=backspace.")
    fps_avg, fps_last_t = 0.0, time.perf_counter()
    last_top: list[tuple[str, float]] = []

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_bgr = cv2.flip(frame_bgr, 1)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            feat, det_mask = extractor._frame_to_features(rgb)     # noqa: SLF001
            # If MediaPipe sees nothing, we still want to feed *something* so
            # the model can predict "nothing" (the dataset has that class).
            # Wrist-relative normalize the (zeroed) vector for consistency.
            if not det_mask.any():
                feat = np.zeros_like(feat)
            predictor.push_landmarks(feat)

            result = predictor.maybe_predict()
            if result is not None:
                last_top, _lat = result
                top_label, top_conf = last_top[0]
                committed = typer.push(top_label, top_conf)
                if committed is not None:
                    log.info("Committed: %r  ->  %r", committed, typer.text)

            # ---- HUD ----
            now = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, now - fps_last_t)
            fps_avg = 0.9 * fps_avg + 0.1 * inst_fps if fps_avg > 0 else inst_fps
            fps_last_t = now

            cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 84),
                          (0, 0, 0), -1)
            cv2.putText(frame_bgr, f"Typed: {typer.text}",
                        (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2)
            cv2.putText(frame_bgr,
                        f"FPS: {fps_avg:5.1f}   "
                        f"Latency: {predictor.last_latency_ms:5.1f} ms   "
                        f"q=quit  c=clear  b=backspace",
                        (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (180, 180, 180), 1)

            y = 110
            for i, (label, prob) in enumerate(last_top):
                color = (0, 255, 0) if i == 0 and prob >= float(
                    cfg.realtime.confidence_threshold) else (200, 200, 200)
                cv2.putText(frame_bgr, f"{i+1}. {label}  {prob*100:5.1f}%",
                            (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                y += 32

            cv2.imshow("ASL Fingerspelling", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                typer.clear()
            if key == ord("b"):
                typer.text = typer.text[:-1]
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
