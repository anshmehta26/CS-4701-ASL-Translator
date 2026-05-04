"""MediaPipe-based landmark extraction.

For each input video, this module produces a numpy array of shape
``(T, num_hands * 21 * 3)`` where ``T`` is the number of valid frames after
trimming and dropping over-occluded videos.

Preprocessing performed per-clip (in order):
  1. Read frames with OpenCV, optionally crop to bbox.
  2. Run MediaPipe Hands on each frame (RGB).
  3. For frames with no detection, leave NaNs and interpolate later.
  4. Drop the clip if fraction of missing frames > max_missing_frame_ratio.
  5. Wrist-relative translation: subtract joint 0 from all joints in the hand.
  6. Scale normalization: divide by max-pairwise-distance within the hand.
  7. Temporal smoothing: centered moving average over ``smoothing_window``
     frames (odd-sized).

This file is import-safe even without MediaPipe installed; we lazy-import it
inside the extractor class so that downstream modules (training, model defs)
can be imported on machines without a MediaPipe build.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from src.utils import get_logger

log = get_logger(__name__)

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z


@dataclass
class ExtractionResult:
    landmarks: np.ndarray   # (T, num_hands * 21 * 3)
    detection_mask: np.ndarray  # (T, num_hands) bool: True if hand detected
    fps: float
    width: int
    height: int


def _hand_feature_size(num_hands: int) -> int:
    return num_hands * NUM_LANDMARKS * COORDS_PER_LANDMARK


def _empty_frame(num_hands: int) -> np.ndarray:
    """Per-frame feature vector filled with NaN (used when no hand detected).

    NaNs are turned into real values later by interpolation; if interpolation
    fails (e.g., entire video has no detection) the clip is dropped.
    """
    return np.full(_hand_feature_size(num_hands), np.nan, dtype=np.float32)


def _wrist_relative_normalize(frame: np.ndarray, num_hands: int) -> np.ndarray:
    """Translate each hand so its wrist (joint 0) sits at the origin, then
    scale by the largest within-hand pairwise distance."""
    out = frame.copy()
    per_hand = NUM_LANDMARKS * COORDS_PER_LANDMARK
    for h in range(num_hands):
        s, e = h * per_hand, (h + 1) * per_hand
        if np.isnan(out[s:e]).all():
            continue
        hand = out[s:e].reshape(NUM_LANDMARKS, COORDS_PER_LANDMARK)
        wrist = hand[0].copy()
        hand = hand - wrist
        # Scale by the hand's bounding-box diagonal in xyz so distance to the
        # camera doesn't change feature magnitudes.
        if not np.isnan(hand).any():
            spread = float(np.linalg.norm(hand.max(axis=0) - hand.min(axis=0)))
            if spread > 1e-6:
                hand = hand / spread
        out[s:e] = hand.reshape(-1)
    return out


def _interpolate_missing(seq: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values along the time axis (per-feature)."""
    seq = seq.copy()
    T, D = seq.shape
    for d in range(D):
        col = seq[:, d]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        if nan_mask.all():
            seq[:, d] = 0.0
            continue
        valid_idx = np.flatnonzero(~nan_mask)
        # np.interp gracefully edge-pads with the first/last valid value
        seq[:, d] = np.interp(np.arange(T), valid_idx, col[valid_idx])
    return seq


def _moving_average(seq: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average smoothing along the time axis."""
    if window <= 1:
        return seq
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(seq, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.empty_like(seq)
    for d in range(seq.shape[1]):
        smoothed[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return smoothed


class LandmarkExtractor:
    """Wrapper around MediaPipe Hands.

    Designed to be reused across many videos: instantiate once, call
    ``extract_video`` repeatedly, then ``close()`` when done.
    """

    def __init__(self, mp_cfg, prep_cfg) -> None:
        self.mp_cfg = mp_cfg
        self.prep_cfg = prep_cfg
        self.num_hands = int(prep_cfg.num_hands)
        self._hands = None  # lazy

    # --- lifecycle ----------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._hands is not None:
            return
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:  # ImportError or DLL load issues
            raise RuntimeError(
                "MediaPipe is required for landmark extraction. "
                "Install it with `pip install mediapipe`."
            ) from exc
        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=bool(self.mp_cfg.static_image_mode),
            max_num_hands=self.num_hands,
            min_detection_confidence=float(self.mp_cfg.min_detection_confidence),
            min_tracking_confidence=float(self.mp_cfg.min_tracking_confidence),
        )

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def __enter__(self) -> "LandmarkExtractor":
        self._ensure_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- core extraction ---------------------------------------------------
    def _frames_from_video(
        self, video_path: Path, frame_start: int, frame_end: int,
        bbox: tuple[int, int, int, int] | None,
    ) -> Iterator[np.ndarray]:
        import cv2  # local import keeps top-level imports cheap
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        try:
            i = 0
            # WLASL frame indices are 1-based; -1 means "use entire video".
            start = 0 if frame_start == -1 else max(0, frame_start - 1)
            end_exclusive = None if frame_end == -1 else frame_end
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                i = start
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if end_exclusive is not None and i >= end_exclusive:
                    break
                if bbox is not None:
                    x, y, w, h = bbox
                    h_img, w_img = frame.shape[:2]
                    x = max(0, x); y = max(0, y)
                    w = min(w, w_img - x); h = min(h, h_img - y)
                    if w > 0 and h > 0:
                        frame = frame[y:y + h, x:x + w]
                # OpenCV reads BGR; MediaPipe wants RGB
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                i += 1
        finally:
            cap.release()

    def _frame_to_features(self, rgb_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run MediaPipe on one frame; return (feature_vector, mask).

        ``mask`` is a length-``num_hands`` boolean of which slots got a
        detection. The feature vector has NaNs in the slots that didn't.
        """
        feat = _empty_frame(self.num_hands)
        mask = np.zeros(self.num_hands, dtype=bool)
        results = self._hands.process(rgb_frame)  # type: ignore[union-attr]
        if not results.multi_hand_landmarks:
            return feat, mask
        # Sort detected hands by handedness (Right first) so feature ordering
        # is consistent across frames. Falls back to detection order.
        hands = list(zip(
            results.multi_hand_landmarks,
            results.multi_handedness or [],
        ))
        def _key(item):
            _, handed = item
            if handed is None:
                return 1
            label = handed.classification[0].label
            return 0 if label == "Right" else 1
        hands.sort(key=_key)

        per_hand = NUM_LANDMARKS * COORDS_PER_LANDMARK
        for h_idx, (lms, _) in enumerate(hands[: self.num_hands]):
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in lms.landmark],
                dtype=np.float32,
            ).reshape(-1)
            feat[h_idx * per_hand:(h_idx + 1) * per_hand] = arr
            mask[h_idx] = True
        return feat, mask

    def extract_video(
        self, video_path: str | Path,
        frame_start: int = -1, frame_end: int = -1,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> ExtractionResult | None:
        self._ensure_open()
        video_path = Path(video_path)

        feats: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for rgb in self._frames_from_video(video_path, frame_start, frame_end, bbox):
            f, m = self._frame_to_features(rgb)
            feats.append(f)
            masks.append(m)

        if len(feats) == 0:
            return None

        seq = np.stack(feats, axis=0)             # (T, D)
        det_mask = np.stack(masks, axis=0)        # (T, num_hands)

        # Drop clip if too many missing frames in the *primary* (first) hand.
        miss_ratio = 1.0 - det_mask[:, 0].mean()
        if miss_ratio > float(self.prep_cfg.max_missing_frame_ratio):
            log.warning(
                "Dropping %s: %.0f%% of frames missing a hand detection.",
                video_path.name, 100 * miss_ratio,
            )
            return None

        if bool(self.prep_cfg.interpolate_missing):
            seq = _interpolate_missing(seq)
        else:
            seq = np.nan_to_num(seq, nan=0.0)

        if bool(self.prep_cfg.wrist_relative):
            seq = np.stack(
                [_wrist_relative_normalize(f, self.num_hands) for f in seq],
                axis=0,
            )

        seq = _moving_average(seq, int(self.prep_cfg.smoothing_window))

        # Sanity: kill any residual NaN/Inf
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        return ExtractionResult(
            landmarks=seq.astype(np.float32),
            detection_mask=det_mask,
            fps=0.0, width=0, height=0,
        )

    def extract_image(self, image_path: str | Path) -> ExtractionResult | None:
        """Run landmark extraction on a single still image.

        The returned ``landmarks`` array has shape ``(1, D)`` so it can flow
        through the same Dataset / model pipeline as a 1-frame "sequence".
        Returns ``None`` if MediaPipe failed to detect any hand in the image
        (we drop these because there's no temporal context to interpolate
        from).
        """
        self._ensure_open()
        import cv2
        image_path = Path(image_path)
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        feat, mask = self._frame_to_features(rgb)
        if not mask.any():
            return None  # no hand detected -> drop this sample

        # Apply the same per-frame normalization used by extract_video,
        # but skip temporal smoothing (only one frame).
        if bool(self.prep_cfg.wrist_relative):
            feat = _wrist_relative_normalize(feat, self.num_hands)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        return ExtractionResult(
            landmarks=feat.reshape(1, -1).astype(np.float32),
            detection_mask=mask.reshape(1, -1),
            fps=0.0, width=0, height=0,
        )
