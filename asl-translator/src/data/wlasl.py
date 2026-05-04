"""Parse the WLASL dataset metadata and produce a flat sample manifest.

The WLASL JSON has the structure::

    [
      {
        "gloss": "book",
        "instances": [
          {"video_id": "12345", "split": "train", "signer_id": 7, ...},
          ...
        ]
      },
      ...
    ]

Real videos must be downloaded separately (the official repo provides scrapers).
Our pipeline only requires that ``{paths.raw_videos}/{video_id}.mp4`` exists for
each sample we want to use; missing files are skipped with a warning.

This module's job is:
  1. Load WLASL_v0.3.json
  2. Pick the top-N most frequent glosses (subset training)
  3. Build a manifest of samples that actually have a video file on disk
  4. Save the manifest as a CSV for downstream stages
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class Sample:
    video_id: str
    gloss: str
    label: int
    split: str           # "train" | "val" | "test"
    signer_id: int       # -1 if unknown
    bbox: tuple[int, int, int, int] | None  # (x, y, w, h) or None
    frame_start: int     # 1-indexed inclusive (-1 = whole video)
    frame_end: int       # 1-indexed inclusive (-1 = whole video)
    video_path: str

    def to_row(self) -> dict:
        bbox_str = ",".join(map(str, self.bbox)) if self.bbox else ""
        return {
            "video_id": self.video_id,
            "gloss": self.gloss,
            "label": self.label,
            "split": self.split,
            "signer_id": self.signer_id,
            "bbox": bbox_str,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "video_path": self.video_path,
        }


def _resolve_video_path(video_id: str, raw_dir: Path) -> Path | None:
    """WLASL videos may be .mp4 or .swf. Prefer mp4."""
    for ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        p = raw_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    return None


def select_top_glosses(entries: list[dict], num_classes: int,
                       min_videos: int) -> list[str]:
    """Pick the ``num_classes`` glosses with the most instances, requiring at
    least ``min_videos`` instances each."""
    counts = Counter()
    for e in entries:
        counts[e["gloss"]] = len(e.get("instances", []))
    eligible = [g for g, c in counts.items() if c >= min_videos]
    eligible.sort(key=lambda g: counts[g], reverse=True)
    chosen = eligible[:num_classes]
    log.info(
        "Selected %d glosses (requested %d, min videos %d). Top-5: %s",
        len(chosen), num_classes, min_videos, chosen[:5],
    )
    return chosen


def build_manifest(wlasl_json: Path, raw_videos: Path, num_classes: int,
                   min_videos: int) -> tuple[list[Sample], dict[str, int]]:
    if not wlasl_json.exists():
        raise FileNotFoundError(
            f"WLASL JSON not found at {wlasl_json}. Download it from "
            "https://github.com/dxli94/WLASL"
        )
    with wlasl_json.open() as f:
        entries = json.load(f)

    chosen = select_top_glosses(entries, num_classes, min_videos)
    label_map = {g: i for i, g in enumerate(sorted(chosen))}
    chosen_set = set(chosen)

    samples: list[Sample] = []
    missing = 0
    for e in entries:
        gloss = e["gloss"]
        if gloss not in chosen_set:
            continue
        for inst in e.get("instances", []):
            vid = str(inst["video_id"])
            vpath = _resolve_video_path(vid, raw_videos)
            if vpath is None:
                missing += 1
                continue
            bbox = tuple(inst["bbox"]) if "bbox" in inst else None
            samples.append(Sample(
                video_id=vid,
                gloss=gloss,
                label=label_map[gloss],
                split=inst.get("split", "train"),
                signer_id=int(inst.get("signer_id", -1)),
                bbox=bbox,                             # type: ignore[arg-type]
                frame_start=int(inst.get("frame_start", -1)),
                frame_end=int(inst.get("frame_end", -1)),
                video_path=str(vpath),
            ))
    log.info(
        "Built manifest: %d samples available, %d missing video files.",
        len(samples), missing,
    )
    return samples, label_map


def save_manifest(samples: Iterable[Sample], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id", "gloss", "label", "split", "signer_id",
        "bbox", "frame_start", "frame_end", "video_path",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            w.writerow(s.to_row())
    log.info("Wrote manifest to %s", out_csv)


def save_label_map(label_map: dict[str, int], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(label_map, f, indent=2)
    log.info("Wrote label map (%d classes) to %s", len(label_map), out_json)
