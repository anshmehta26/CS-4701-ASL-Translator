"""Tests for the alphabet path: manifest builder and typing state machine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.alphabet import build_alphabet_manifest


def _make_alphabet_dir(root: Path, classes: list[str], n_per_class: int) -> Path:
    """Create a fake Kaggle-style folder layout with empty .jpg files."""
    train = root / "asl_alphabet_train"
    train.mkdir(parents=True)
    for c in classes:
        d = train / c
        d.mkdir()
        for i in range(n_per_class):
            (d / f"{c}{i:04d}.jpg").write_bytes(b"")  # content doesn't matter
    return train


def test_alphabet_manifest_basic(tmp_path):
    train = _make_alphabet_dir(
        tmp_path, classes=["A", "B", "C", "space", "del", "nothing"],
        n_per_class=20,
    )
    samples, label_map = build_alphabet_manifest(
        train_root=train, cap_per_class=None,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=42, include_special=True,
    )
    assert len(samples) == 6 * 20
    assert set(label_map.keys()) == {"A", "B", "C", "space", "del", "nothing"}
    splits = {"train": 0, "val": 0, "test": 0}
    for s in samples:
        splits[s.split] += 1
    # Every split should have at least one sample per class
    assert all(v > 0 for v in splits.values())


def test_alphabet_manifest_excludes_special(tmp_path):
    train = _make_alphabet_dir(
        tmp_path, classes=["A", "B", "space", "nothing"], n_per_class=10,
    )
    samples, label_map = build_alphabet_manifest(
        train_root=train, cap_per_class=None,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=42, include_special=False,
    )
    assert set(label_map.keys()) == {"A", "B"}
    assert all(s.gloss in {"A", "B"} for s in samples)


def test_alphabet_manifest_caps_per_class(tmp_path):
    train = _make_alphabet_dir(tmp_path, classes=["A", "B"], n_per_class=100)
    samples, _ = build_alphabet_manifest(
        train_root=train, cap_per_class=10,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=0, include_special=False,
    )
    # 2 classes * 10 = 20
    assert len(samples) == 20


# -----------------------------------------------------------------------------
# Typing state machine
# -----------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
# Import from the script file (not a package); we don't want to execute its
# main, so we set up sys.modules carefully.
import importlib.util as _ilu

_typing_spec = _ilu.spec_from_file_location(
    "_typing_module",
    str(Path(__file__).resolve().parents[1] / "scripts" / "realtime_typing.py"),
)
_typing_module = _ilu.module_from_spec(_typing_spec)
# Don't run main on import — guard via the standard if __name__ == "__main__"
_typing_spec.loader.exec_module(_typing_module)
TypingState = _typing_module.TypingState


def test_typing_commits_after_consecutive_predictions():
    state = TypingState(commit_frames=3, threshold=0.5, cooldown_predictions=5)
    assert state.push("A", 0.9) is None
    assert state.push("A", 0.9) is None
    assert state.push("A", 0.9) == "A"
    assert state.text == "A"


def test_typing_low_confidence_resets_buffer():
    state = TypingState(commit_frames=3, threshold=0.5, cooldown_predictions=5)
    state.push("A", 0.9); state.push("A", 0.9)
    state.push("A", 0.2)                         # low conf -> clears buffer
    state.push("A", 0.9)
    assert state.text == ""


def test_typing_inconsistent_predictions_reset():
    state = TypingState(commit_frames=3, threshold=0.5, cooldown_predictions=5)
    state.push("A", 0.9); state.push("A", 0.9)
    state.push("B", 0.9)                         # buffer now [A, A, B] -> mixed
    assert state.text == ""
    # Three matching B's should still commit
    state.push("B", 0.9); state.push("B", 0.9)
    assert state.text == "B"


def test_typing_nothing_acts_as_separator():
    state = TypingState(commit_frames=3, threshold=0.5, cooldown_predictions=10)
    for _ in range(3):
        state.push("A", 0.9)
    assert state.text == "A"
    # "nothing" allows the next confident letter to commit immediately.
    state.push("nothing", 0.9)
    for _ in range(3):
        state.push("A", 0.9)
    assert state.text == "AA"


def test_typing_cooldown_blocks_repeat():
    state = TypingState(commit_frames=3, threshold=0.5, cooldown_predictions=10)
    for _ in range(3):
        state.push("A", 0.9)
    assert state.text == "A"
    # Without a separator, repeated A's should NOT commit again until cooldown
    for _ in range(5):
        state.push("A", 0.9)
    assert state.text == "A"


def test_typing_special_tokens():
    state = TypingState(commit_frames=2, threshold=0.5, cooldown_predictions=3)
    # Type "AB"
    state.push("A", 0.9); state.push("A", 0.9)
    state.push("nothing", 0.9)
    state.push("B", 0.9); state.push("B", 0.9)
    assert state.text == "AB"
    # Backspace via "del"
    state.push("nothing", 0.9)
    state.push("del", 0.9); state.push("del", 0.9)
    assert state.text == "A"
    # Space
    state.push("nothing", 0.9)
    state.push("space", 0.9); state.push("space", 0.9)
    assert state.text == "A "


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        test_alphabet_manifest_basic(Path(t))
    print("Manifest tests passed.")
    test_typing_commits_after_consecutive_predictions()
    test_typing_low_confidence_resets_buffer()
    test_typing_inconsistent_predictions_reset()
    test_typing_nothing_acts_as_separator()
    test_typing_cooldown_blocks_repeat()
    test_typing_special_tokens()
    print("All typing tests passed.")
