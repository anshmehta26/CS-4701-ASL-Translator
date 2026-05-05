"""Tests for the ASL-HG signer-level loader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.asl_hg import build_asl_hg_manifest


def _make_asl_hg_dir(root: Path, classes: list[str], signers: list[str],
                     n_per_signer: int) -> Path:
    """Create a fake ASL-HG layout with empty .jpg files."""
    ds = root / "asl_dataset"
    ds.mkdir(parents=True)
    for c in classes:
        d = ds / c
        d.mkdir()
        for sig in signers:
            for i in range(1, n_per_signer + 1):
                (d / f"{sig}_{c}_{i}.jpg").write_bytes(b"")
    return ds


def test_asl_hg_basic_split(tmp_path):
    classes = ["A", "B", "C"]
    signers = [f"P{i}" for i in range(1, 11)]
    ds = _make_asl_hg_dir(tmp_path, classes, signers, n_per_signer=5)

    samples, label_map = build_asl_hg_manifest(
        dataset_root=ds,
        train_signers=["P1", "P2", "P3", "P4", "P5", "P6"],
        val_signers=["P7", "P8"],
        test_signers=["P9", "P10"],
    )
    assert len(samples) == 3 * 10 * 5
    assert set(label_map.keys()) == {"A", "B", "C"}

    # By-signer split correctness: no signer appears in two splits.
    by_split = {"train": set(), "val": set(), "test": set()}
    for s in samples:
        by_split[s.split].add(s.signer_id)
    assert by_split["train"] == {"P1", "P2", "P3", "P4", "P5", "P6"}
    assert by_split["val"] == {"P7", "P8"}
    assert by_split["test"] == {"P9", "P10"}
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])


def test_asl_hg_excludes_digits_by_default(tmp_path):
    ds = _make_asl_hg_dir(
        tmp_path, classes=["A", "B", "0", "9"],
        signers=["P1", "P2", "P3"], n_per_signer=4,
    )
    samples, label_map = build_asl_hg_manifest(
        dataset_root=ds,
        train_signers=["P1"], val_signers=["P2"], test_signers=["P3"],
    )
    assert set(label_map.keys()) == {"A", "B"}
    assert all(s.gloss in {"A", "B"} for s in samples)


def test_asl_hg_includes_digits_when_asked(tmp_path):
    ds = _make_asl_hg_dir(
        tmp_path, classes=["A", "0", "1"],
        signers=["P1", "P2", "P3"], n_per_signer=4,
    )
    samples, label_map = build_asl_hg_manifest(
        dataset_root=ds,
        train_signers=["P1"], val_signers=["P2"], test_signers=["P3"],
        include_digits=True,
    )
    assert set(label_map.keys()) == {"A", "0", "1"}


def test_asl_hg_disjoint_signer_lists_required(tmp_path):
    ds = _make_asl_hg_dir(tmp_path, ["A"], ["P1", "P2"], 1)
    import pytest
    with pytest.raises(ValueError):
        build_asl_hg_manifest(
            dataset_root=ds,
            train_signers=["P1", "P2"],
            val_signers=["P2"],         # overlap with train
            test_signers=[],
        )


def test_asl_hg_unknown_signer_skipped(tmp_path):
    classes = ["A"]
    signers = ["P1", "P2", "P3"]
    ds = _make_asl_hg_dir(tmp_path, classes, signers, n_per_signer=2)
    # Only P1 is in the split lists -> P2 and P3 should be silently dropped
    samples, _ = build_asl_hg_manifest(
        dataset_root=ds,
        train_signers=["P1"],
        val_signers=[],
        test_signers=[],
    )
    assert len(samples) == 2  # 1 class * 1 signer * 2 images
    assert all(s.signer_id == "P1" for s in samples)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        test_asl_hg_basic_split(Path(t))
    with tempfile.TemporaryDirectory() as t:
        test_asl_hg_excludes_digits_by_default(Path(t))
    with tempfile.TemporaryDirectory() as t:
        test_asl_hg_includes_digits_when_asked(Path(t))
    with tempfile.TemporaryDirectory() as t:
        test_asl_hg_disjoint_signer_lists_required(Path(t))
    with tempfile.TemporaryDirectory() as t:
        test_asl_hg_unknown_signer_skipped(Path(t))
    print("All ASL-HG tests passed.")
