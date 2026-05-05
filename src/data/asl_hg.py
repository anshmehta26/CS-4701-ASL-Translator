"""ASL-HG (Mendeley) dataset loader.

The dataset is laid out as::

    asl_dataset/
        A/  P1_A_1.jpg, P1_A_2.jpg, ..., P10_A_100.jpg
        B/  P1_B_1.jpg, ...
        ...
        Z/
        0/, 1/, ..., 9/

Each filename starts with ``Pn_`` where ``n`` is the signer index (1-10).
There are 10 distinct signers contributing 100 images per class.

Critical design decision: we split **by signer**, never by image. The same
person never appears in two splits. This is the whole point of using this
dataset — without by-signer splits, the test number wouldn't measure what
we want it to measure (cross-signer generalization).

We restrict to letter classes (A-Z) by default, since our project task is
fingerspelling, not digit recognition. Digits can be optionally included.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.utils import get_logger

log = get_logger(__name__)


_FILENAME_RE = re.compile(r"^(P\d+)_([A-Z0-9])_\d+\.(?:jpg|jpeg|png)$",
                          re.IGNORECASE)
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class HGSample:
    image_id: str        # e.g. "P3_A_42"
    gloss: str           # "A".."Z" (and optionally "0".."9")
    label: int
    split: str           # train / val / test
    signer_id: str       # "P1".."P10"
    image_path: str

    def to_row(self) -> dict:
        # Mirror the WLASL/Alphabet manifest schema for downstream code reuse.
        return {
            "video_id": self.image_id,
            "gloss": self.gloss,
            "label": self.label,
            "split": self.split,
            "signer_id": self.signer_id,
            "bbox": "",
            "frame_start": -1,
            "frame_end": -1,
            "video_path": self.image_path,
        }


def _parse_filename(name: str) -> tuple[str, str] | None:
    """Returns (signer_id, gloss) or None if the filename doesn't match."""
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return m.group(1).upper(), m.group(2).upper()


def build_asl_hg_manifest(
    dataset_root: Path,
    train_signers: list[str],
    val_signers: list[str],
    test_signers: list[str],
    include_digits: bool = False,
) -> tuple[list[HGSample], dict[str, int]]:
    """Walk ASL-HG and tag each image with split based on signer.

    Args:
        dataset_root: path to the ``asl_dataset`` folder (the one with A, B, ...).
        train_signers / val_signers / test_signers: disjoint lists of "P1".."P10".
        include_digits: if False (default), drop the 0-9 class folders.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"ASL-HG dataset root not found: {dataset_root}\n"
            "Expected layout: <root>/{A,B,...,Z}/Pn_X_idx.jpg"
        )
    all_signers = set(train_signers) | set(val_signers) | set(test_signers)
    if any(s in val_signers for s in train_signers) or \
       any(s in test_signers for s in train_signers) or \
       any(s in test_signers for s in val_signers):
        raise ValueError("Signer lists must be disjoint across splits.")

    split_of = {s: "train" for s in train_signers}
    split_of.update({s: "val" for s in val_signers})
    split_of.update({s: "test" for s in test_signers})

    class_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    glosses: list[str] = []
    for d in class_dirs:
        n = d.name
        if not include_digits and n.isdigit():
            continue
        if n.isalpha() and len(n) == 1 and n.isascii():
            glosses.append(n.upper())
        elif include_digits and n.isdigit() and len(n) == 1:
            glosses.append(n)
    glosses = sorted(set(glosses))
    label_map = {g: i for i, g in enumerate(glosses)}
    log.info(
        "ASL-HG: %d classes (digits=%s): %s",
        len(glosses), include_digits, ", ".join(glosses[:10])
        + ("..." if len(glosses) > 10 else ""),
    )

    samples: list[HGSample] = []
    seen_signers: dict[str, int] = {}
    skipped_unknown_signer = 0
    skipped_unparseable = 0

    for d in class_dirs:
        gloss = d.name.upper() if d.name.isalpha() else d.name
        if gloss not in label_map:
            continue
        for img in d.iterdir():
            if not img.is_file() or img.suffix.lower() not in _IMAGE_EXTS:
                continue
            parsed = _parse_filename(img.name)
            if parsed is None:
                skipped_unparseable += 1
                continue
            signer_id, file_gloss = parsed
            if file_gloss != gloss:
                # Filename and folder disagree -> trust the folder (label).
                file_gloss = gloss
            if signer_id not in split_of:
                skipped_unknown_signer += 1
                continue
            seen_signers[signer_id] = seen_signers.get(signer_id, 0) + 1
            samples.append(HGSample(
                image_id=f"{signer_id}_{gloss}_{img.stem.split('_')[-1]}",
                gloss=gloss,
                label=label_map[gloss],
                split=split_of[signer_id],
                signer_id=signer_id,
                image_path=str(img),
            ))

    log.info(
        "Built ASL-HG manifest: %d samples (unparseable=%d, unknown_signer=%d).",
        len(samples), skipped_unparseable, skipped_unknown_signer,
    )
    log.info("Per-signer counts: %s", dict(sorted(seen_signers.items())))
    counts = {"train": 0, "val": 0, "test": 0}
    for s in samples:
        counts[s.split] += 1
    log.info("Per-split counts: %s", counts)
    return samples, label_map
