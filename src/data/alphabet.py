"""ASL Alphabet (Kaggle) loader.

Dataset layout we expect (the Kaggle ZIP unpacks like this)::

    data/raw/asl_alphabet_train/
        A/A1.jpg, A/A2.jpg, ...
        B/B1.jpg, ...
        ...
        Z/...
        space/...
        nothing/...
        del/...

Each class folder contains thousands of single images. For our purposes one
image is a 1-frame "sequence". We optionally cap per-class so MediaPipe
doesn't have to process all 87k images (which adds little after a few
hundred per class).

This module produces a Sample list and label_map analogous to the WLASL
loader, so the existing manifest / extraction / training code paths can
consume it without modification.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from src.utils import get_logger

log = get_logger(__name__)

# Common variations of class names we may encounter
_DEFAULT_INCLUDE_SPECIAL = {"space", "del", "nothing"}


@dataclass
class AlphabetSample:
    image_id: str         # e.g. "A_0001"
    gloss: str            # "A", "B", ..., "space", "del", "nothing"
    label: int
    split: str            # train / val / test
    image_path: str

    def to_row(self) -> dict:
        # Mirror the WLASL manifest schema so the dataset code is identical.
        return {
            "video_id": self.image_id,
            "gloss": self.gloss,
            "label": self.label,
            "split": self.split,
            "signer_id": -1,
            "bbox": "",
            "frame_start": -1,
            "frame_end": -1,
            "video_path": self.image_path,
        }


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_class_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)


def _normalize_class_name(name: str) -> str:
    # Kaggle's folder names are exactly "A".."Z", "space", "del", "nothing".
    # Some mirrors lowercase the letters; normalize letters to uppercase but
    # keep the special tokens lowercase.
    if name.lower() in _DEFAULT_INCLUDE_SPECIAL:
        return name.lower()
    return name.upper()


def build_alphabet_manifest(
    train_root: Path,
    cap_per_class: int | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    include_special: bool = True,
) -> tuple[list[AlphabetSample], dict[str, int]]:
    if not train_root.exists():
        raise FileNotFoundError(
            f"Alphabet dataset not found at {train_root}. Download from "
            "https://www.kaggle.com/datasets/grassknoted/asl-alphabet and "
            "extract under data/raw/."
        )
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = random.Random(seed)

    class_dirs = sorted(p for p in train_root.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found inside {train_root}.")

    glosses: list[str] = []
    for d in class_dirs:
        g = _normalize_class_name(d.name)
        if g in _DEFAULT_INCLUDE_SPECIAL and not include_special:
            continue
        glosses.append(g)
    glosses = sorted(set(glosses))
    label_map = {g: i for i, g in enumerate(glosses)}
    log.info(
        "Found %d alphabet classes: %s",
        len(glosses), ", ".join(glosses[:8]) + ("..." if len(glosses) > 8 else ""),
    )

    samples: list[AlphabetSample] = []
    for d in class_dirs:
        gloss = _normalize_class_name(d.name)
        if gloss not in label_map:
            continue
        images = _list_class_images(d)
        if not images:
            log.warning("Class %s has no images; skipping.", gloss)
            continue
        rng.shuffle(images)
        if cap_per_class is not None:
            images = images[:cap_per_class]

        n = len(images)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_train = max(1, n - n_val - n_test)
        # Re-balance if rounding pushed n_train negative
        if n_train + n_val + n_test > n:
            n_test = max(1, n - n_train - n_val)

        for i, img_path in enumerate(images):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            samples.append(AlphabetSample(
                image_id=f"{gloss}_{i:05d}",
                gloss=gloss,
                label=label_map[gloss],
                split=split,
                image_path=str(img_path),
            ))

    log.info(
        "Built alphabet manifest: %d samples across %d classes (capped at %s/class).",
        len(samples), len(glosses),
        "all" if cap_per_class is None else cap_per_class,
    )
    return samples, label_map
