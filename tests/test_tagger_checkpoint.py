"""``TaggerCheckpoint.from_dir`` — one read of a checkpoint dir.

Five CLIs opened ``config.json`` / ``vocab.json`` / ``dataset.json`` by hand,
each with its own spelling of the missing-file exit and its own idea of which
files were optional. These pin the parts they disagreed on.
"""

from __future__ import annotations

import json

import pytest

from anime_tools.tagger.data import TaggerCheckpoint, TaggerManifest

VOCAB = {
    "tags": [
        {"name": "1girl", "index": 0, "category": "count"},
        {"name": "blue hair", "index": 1, "category": "general"},
    ],
    "ratings": ["general"],
}
DATASET = {
    "stems": ["a", "b"],
    "image_paths": ["img/a.webp", "img/b.webp"],
    "tag_indices": [[0], [0, 1]],
    "rating_indices": [0, 0],
    "n_tags": 2,
    "n_ratings": 1,
    "split": {"train": ["a"], "val": ["b"]},
}


def _ckpt(tmp_path, *, config=None, vocab=VOCAB, dataset=None):
    for name, payload in (
        ("config.json", config),
        ("vocab.json", vocab),
        ("dataset.json", dataset),
    ):
        if payload is not None:
            (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_reads_every_present_file_even_when_only_one_is_required(tmp_path):
    _ckpt(tmp_path, config={"backend": "dbv4"}, dataset=DATASET)
    ckpt = TaggerCheckpoint.from_dir(tmp_path)
    assert ckpt.vocab == VOCAB
    assert ckpt.dataset == DATASET
    assert ckpt.config == {"backend": "dbv4"}


def test_an_unrequired_absent_file_is_none_not_an_exit(tmp_path):
    """`derive_groups` merely *prefers* the manifest — no manifest, caption scan."""
    _ckpt(tmp_path)
    ckpt = TaggerCheckpoint.from_dir(tmp_path, require=("vocab",))
    assert ckpt.dataset is None and ckpt.config is None


def test_missing_required_files_are_named_together(tmp_path):
    _ckpt(tmp_path, vocab=None)
    with pytest.raises(SystemExit) as e:
        TaggerCheckpoint.from_dir(tmp_path, require=("vocab", "dataset"))
    msg = str(e.value)
    assert "vocab.json" in msg and "dataset.json" in msg
    assert "run --mode build_vocab first" in msg


def test_backend_check_implies_reading_the_config(tmp_path):
    _ckpt(tmp_path, config={"backend": "pe"})
    with pytest.raises(SystemExit, match="is not a dbv4-backed checkpoint"):
        TaggerCheckpoint.from_dir(tmp_path, backend="dbv4")
    # …and a checkpoint with no config.json at all fails on the file, not a
    # None-deref inside the backend test.
    (tmp_path / "bare").mkdir()
    _ckpt(tmp_path / "bare", vocab=VOCAB)
    with pytest.raises(SystemExit, match="config.json"):
        TaggerCheckpoint.from_dir(tmp_path / "bare", backend="dbv4")


def test_unknown_require_key_is_a_programming_error(tmp_path):
    _ckpt(tmp_path)
    with pytest.raises(ValueError, match="unknown checkpoint file"):
        TaggerCheckpoint.from_dir(tmp_path, require=("rules",))


def test_idx_to_name_and_n_tags(tmp_path):
    ckpt = TaggerCheckpoint.from_dir(_ckpt(tmp_path))
    assert ckpt.idx_to_name() == {0: "1girl", 1: "blue hair"}
    assert ckpt.n_tags == 2


def test_manifest_revives_the_already_read_dataset(tmp_path):
    ckpt = TaggerCheckpoint.from_dir(_ckpt(tmp_path, dataset=DATASET))
    m = ckpt.manifest()
    assert isinstance(m, TaggerManifest)
    assert m.val_stems == ["b"]
    assert m.stem_index() == {"a": 0, "b": 1}
    # Same object as the path-based reader, which now delegates to from_dict.
    (tmp_path / "dataset.json").write_text(json.dumps(DATASET), encoding="utf-8")
    assert TaggerManifest.from_path(tmp_path / "dataset.json") == m


def test_manifest_without_one_still_points_at_build_vocab(tmp_path):
    ckpt = TaggerCheckpoint.from_dir(_ckpt(tmp_path))
    with pytest.raises(SystemExit, match="run --mode build_vocab first"):
        ckpt.manifest()
