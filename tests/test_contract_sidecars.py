"""A ``{stem}.*.txt`` sidecar beside a caption is never read as a caption.

Every sidecar that ends in ``.txt`` is in ``contract.CAPTION_SIDECAR_SUFFIXES``,
and the walks that collect captions by suffix skip it."""

from __future__ import annotations

from pathlib import Path

from anime_tools.captions.history import HISTORY_SIDECAR_SUFFIX
from anime_tools.captions.ocr_sidecar import OCR_SIDECAR_SUFFIX
from anime_tools.captions.variants import VARIANTS_SIDECAR_SUFFIX
from anime_tools.contract import (
    CAPTION_SIDECAR_SUFFIXES,
    LAYOUT_JSON_SUFFIX,
    is_caption_sidecar,
)

SIDECARS = {
    VARIANTS_SIDECAR_SUFFIX: "# generated\nv0\t1girl, solo\n",
    HISTORY_SIDECAR_SUFFIX: "# history\n",
    OCR_SIDECAR_SUFFIX: "1\t0,0,10,10\t0.9\t文字\n",
    LAYOUT_JSON_SUFFIX: '{"summary":{"caption":"1girl"}}',
}


def test_the_tuple_is_every_owners_suffix():
    assert set(CAPTION_SIDECAR_SUFFIXES) == set(SIDECARS)
    assert LAYOUT_JSON_SUFFIX == ".layout.txt"
    assert all(s.endswith(".txt") for s in CAPTION_SIDECAR_SUFFIXES)
    assert is_caption_sidecar("a.b.layout.txt") and is_caption_sidecar("A.LAYOUT.TXT")
    assert not is_caption_sidecar("a.txt") and not is_caption_sidecar("layout.txt")


def _tree(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "a.txt").write_text("1girl, solo", encoding="utf-8")
    for suffix, text in SIDECARS.items():
        (root / f"a{suffix}").write_text(text, encoding="utf-8")


def test_the_caption_index_walk_skips_sidecars(tmp_path):
    from anime_tools.captions.index import _iter_captions

    _tree(tmp_path / "src")
    assert [rel for _, rel, _ in _iter_captions(tmp_path / "src")] == ["a.txt"]


def test_the_tagger_vocab_walk_skips_sidecars(tmp_path):
    from anime_tools.tagger.cli.vocab import find_caption_files

    _tree(tmp_path / "src")
    assert find_caption_files([tmp_path / "src"]) == [tmp_path / "src" / "a.txt"]


def test_export_publishes_the_caption_not_the_layout_sidecar(tmp_path):
    """A `.layout.txt` in the workspace tree is not a caption either: the
    caption row is the image's `{stem}.txt`."""
    from conftest import write_png

    from anime_tools.stages.export_workspace import ExportPaths, plan_export

    resized = tmp_path / "workspace" / "resized"
    _tree(resized)
    write_png(resized / "a.png")
    paths = ExportPaths(
        resized=resized,
        masks=tmp_path / "workspace" / "masks",
        master=tmp_path / "workspace" / "master",
        index=tmp_path / "workspace" / "captions" / "caption_index.json",
        src=tmp_path / "image_dataset",
        out=tmp_path / "out",
    )
    (caption,) = [r for r in plan_export(paths) if r.kind == "caption"]
    assert Path(caption.src) == resized / "a.txt"
