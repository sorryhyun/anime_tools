"""Export: which artifacts it finds, when it skips, what it refuses to overwrite
blind, and that publishing twice publishes nothing the second time."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from anime_tools.stages.export_workspace import (
    ExportPaths,
    plan_export,
    revert_export,
    rows_from_report,
    run_export,
)


def _png(path: Path, colour: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (colour, colour, colour)).save(path)


def _txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def ws(tmp_path):
    """``a`` has pixels, caption, variants sidecar, mask and revised master;
    ``sub/b`` has pixels only."""
    p = ExportPaths(
        resized=tmp_path / "workspace" / "resized",
        masks=tmp_path / "workspace" / "masks",
        master=tmp_path / "workspace" / "master",
        index=tmp_path / "workspace" / "captions" / "caption_index.json",
        src=tmp_path / "image_dataset",
        out=tmp_path / "post_image_dataset",
    )
    _png(p.resized / "a.png")
    _txt(p.resized / "a.txt", "1girl, solo. On the left, cat.")
    _txt(p.resized / "a.variants.txt", "# generated\nv0\t1girl, solo\n")
    _png(p.masks / "a_mask.png", 200)
    _txt(p.master / "a.txt", "1girl, solo, revised")
    _png(p.resized / "sub" / "b.png")
    _txt(p.index, json.dumps({"image_meta": {}}))
    # The input tree the master publishes back over.
    _txt(p.src / "a.txt", "1girl, solo")
    return p


def _by(rows, kind):
    return [r for r in rows if r.kind == kind]


# ---- the plan -----------------------------------------------------------


def test_plan_finds_every_artifact_and_invents_none(ws):
    rows = plan_export(ws)
    assert {r.kind for r in rows} == {
        "image",
        "caption",
        "variants",
        "mask",
        "master",
        "index",
    }
    # Only `a` has the other four; `sub/b` contributes its pixels alone.
    assert sorted(r.rel for r in _by(rows, "image")) == ["a.png", "sub/b.png"]
    assert [r.rel for r in _by(rows, "caption")] == ["a.png"]
    # Every row is a create except the revised master, which lands over an
    # existing caption.
    assert {r.status for r in rows} == {"would-create", "would-overwrite"}
    assert [r.kind for r in rows if r.status == "would-overwrite"] == ["master"]


def test_the_master_publishes_back_over_the_input_tree(ws):
    """The master row writes outside `--out`, into `image_dataset/`."""
    (master,) = _by(plan_export(ws), "master")
    assert Path(master.dst) == ws.src / "a.txt"
    # It can overwrite hand-written text, so `before` is kept for the revert.
    assert master.status == "would-overwrite" and master.before == "1girl, solo"


def test_an_unrevised_master_is_not_a_row(ws):
    """No overlay means nothing to publish, not an empty write over the input."""
    (ws.master / "a.txt").unlink()
    assert _by(plan_export(ws), "master") == []


def test_a_flat_legacy_mask_still_publishes(ws):
    """A flat `masks/{stem}_mask.png` is still a mask."""
    _png(ws.masks / "b_mask.png", 180)
    rels = {r.rel for r in _by(plan_export(ws), "mask")}
    assert rels == {"a.png", "sub/b.png"}


def test_the_pattern_narrows_to_one_image(ws):
    rows = plan_export(ws, path_pattern="sub/b.*")
    # The index is dataset-wide, so it is not something a scope excludes.
    assert {r.rel for r in rows} == {"sub/b.png", "caption_index.json"}


# ---- the copy -----------------------------------------------------------


def test_a_dry_run_writes_nothing(ws):
    rows, stats = run_export(ws, apply=False)
    assert not ws.out.exists()
    assert stats.created == 0 and all(r.status.startswith("would-") for r in rows)


def test_apply_publishes_the_contract_paths(ws):
    _, stats = run_export(ws, apply=True)
    assert (ws.out / "resized" / "a.png").is_file()
    assert (
        (ws.out / "resized" / "a.txt").read_text(encoding="utf-8").startswith("1girl")
    )
    assert (ws.out / "resized" / "a.variants.txt").is_file()
    assert (ws.out / "resized" / "sub" / "b.png").is_file()
    assert (ws.out / "masks" / "a_mask.png").is_file()
    assert (ws.out / "captions" / "caption_index.json").is_file()
    # The master went to the input tree, not under --out.
    assert (ws.src / "a.txt").read_text(encoding="utf-8") == "1girl, solo, revised"
    assert not (ws.out / "a.txt").exists()
    assert stats.created == 6 and stats.overwrote == 1


def test_exporting_twice_publishes_nothing_the_second_time(ws):
    """`copy2` preserves mtime, so an unchanged tree compares equal next time."""
    run_export(ws, apply=True)
    rows, stats = run_export(ws, apply=True)
    assert stats.created == 0 and stats.overwrote == 0
    assert stats.skipped["identical"] == len(rows)


def test_a_changed_caption_republishes_without_touching_the_pixels(ws):
    run_export(ws, apply=True)
    _txt(ws.resized / "a.txt", "1girl, solo, night")
    rows, stats = run_export(ws, apply=True)
    assert stats.overwrote == 1 and stats.created == 0
    (caption,) = [r for r in rows if r.status == "overwrote"]
    assert caption.kind == "caption"
    assert (ws.out / "resized" / "a.txt").read_text(encoding="utf-8").endswith("night")


def test_a_destination_edited_since_the_plan_is_decided_again_at_write_time(ws):
    """The copy re-reads disk: a row planned as a create that now exists is
    reported as an overwrite."""
    rows = plan_export(ws)
    (image,) = [r for r in _by(rows, "image") if r.rel == "a.png"]
    assert image.status == "would-create"
    _png(ws.out / "resized" / "a.png", 99)
    _, stats = run_export(ws, apply=True)
    assert stats.overwrote == 2  # the master, and now the pre-existing image


def test_an_empty_workspace_yields_no_rows(tmp_path):
    p = ExportPaths(
        resized=tmp_path / "resized",
        masks=tmp_path / "masks",
        master=tmp_path / "master",
        index=tmp_path / "index.json",
        src=tmp_path / "src",
        out=tmp_path / "out",
    )
    p.resized.mkdir()
    assert plan_export(p) == []


# ---- putting it back ----------------------------------------------------


def test_revert_removes_what_the_export_created(ws):
    rows, _ = run_export(ws, apply=True)
    back, stats = revert_export(rows, apply=True)
    assert not (ws.out / "resized" / "a.png").exists()
    assert not (ws.out / "captions" / "caption_index.json").exists()
    assert stats.removed == 6
    assert {r.status for r in back} == {"removed", "restored"}


def test_revert_puts_an_overwritten_master_back(ws):
    rows, _ = run_export(ws, apply=True)
    assert (ws.src / "a.txt").read_text(encoding="utf-8") == "1girl, solo, revised"
    _, stats = revert_export(rows, apply=True)
    assert (ws.src / "a.txt").read_text(encoding="utf-8") == "1girl, solo"
    assert stats.restored == 1


def test_revert_leaves_a_file_edited_since_the_export_alone(ws):
    rows, _ = run_export(ws, apply=True)
    _txt(ws.src / "a.txt", "hand-edited after the export")
    _txt(ws.out / "resized" / "a.txt", "also hand-edited")
    _, stats = revert_export(rows, apply=True)
    assert (ws.src / "a.txt").read_text(
        encoding="utf-8"
    ) == "hand-edited after the export"
    assert (ws.out / "resized" / "a.txt").is_file()
    assert stats.skipped["drifted"] == 2


def test_revert_cannot_put_back_overwritten_pixels(ws):
    """No pixel snapshot is kept; re-exporting is the way back."""
    _png(ws.out / "resized" / "a.png", 99)
    rows, _ = run_export(ws, apply=True)
    _, stats = revert_export(rows, apply=True)
    assert stats.skipped["not-undoable"] == 1
    assert (ws.out / "resized" / "a.png").is_file()


def test_revert_of_a_dry_run_has_nothing_to_undo(ws):
    rows, _ = run_export(ws, apply=False)
    _, stats = revert_export(rows, apply=True)
    assert stats.removed == 0 and stats.restored == 0
    assert stats.skipped["nothing-to-undo"] == len(rows)


def test_rows_survive_the_round_trip_through_a_report(ws):
    """Undo reads rows back out of the report's JSON, `before` included."""
    rows, _ = run_export(ws, apply=True)
    report = {"rows": [r.to_dict() for r in rows]}
    again = rows_from_report(json.loads(json.dumps(report)))
    assert [r.to_dict() for r in again] == [r.to_dict() for r in rows]


# ---- combining OCR ---------------------------------------------------------


JA = 'Japanese text reads as "大丈夫", "本当に".'


@pytest.fixture
def ocr(ws, tmp_path):
    """The workspace's OCR tree: ``a`` has two lines, ``sub/b`` none."""
    from anime_tools.captions.ocr_sidecar import OcrLine, write_ocr_for

    ocr_dir = tmp_path / "workspace" / "ocr"
    write_ocr_for(
        ocr_dir,
        Path("a.txt"),
        [
            OcrLine(seq=1, box=(0, 0, 10, 10), score=0.9, text="大丈夫"),
            OcrLine(seq=2, box=(0, 20, 10, 30), score=0.9, text="本当に"),
        ],
    )
    return ExportPaths(**{**vars(ws), "ocr": ocr_dir})


def test_combine_attaches_the_clause_to_the_caption_and_every_variant(ocr):
    rows, stats = run_export(ocr, apply=True)
    caption = (ocr.out / "resized" / "a.txt").read_text(encoding="utf-8")
    assert caption == f"1girl, solo. On the left, cat. {JA}"
    variants = (ocr.out / "resized" / "a.variants.txt").read_text(encoding="utf-8")
    assert variants.splitlines()[0].startswith("#")
    assert variants.splitlines()[1] == f"v0\t1girl, solo. {JA}"
    assert stats.combined == 2
    assert {r.kind for r in rows if r.combined} == {"caption", "variants"}
    # The workspace copies are what they were: the combine is a property of
    # the published tree.
    assert (ocr.resized / "a.txt").read_text(encoding="utf-8").endswith("cat.")


def test_combine_leaves_the_master_and_an_image_with_no_sidecar_alone(ocr):
    _txt(ocr.resized / "sub" / "b.txt", "2girls")
    rows, stats = run_export(ocr, apply=True)
    assert (ocr.src / "a.txt").read_text(encoding="utf-8") == "1girl, solo, revised"
    assert (ocr.out / "resized" / "sub" / "b.txt").read_text(
        encoding="utf-8"
    ) == "2girls"
    (b,) = [r for r in rows if r.kind == "caption" and r.rel == "sub/b.png"]
    assert not b.combined and not b.text
    assert stats.combined == 2


def test_combining_twice_publishes_nothing_the_second_time(ocr):
    run_export(ocr, apply=True)
    rows, stats = run_export(ocr, apply=True)
    assert stats.created == 0 and stats.overwrote == 0 and stats.combined == 0
    assert stats.skipped["identical"] == len(rows)


def test_exporting_without_the_combine_takes_the_clause_back(ocr, ws):
    run_export(ocr, apply=True)
    rows, stats = run_export(ws, apply=True)
    assert stats.overwrote == 2
    assert {r.kind for r in rows if r.status == "overwrote"} == {"caption", "variants"}
    assert (ws.out / "resized" / "a.txt").read_text(encoding="utf-8").endswith("cat.")


def test_a_sidecar_gone_since_the_plan_publishes_the_caption_bare(ocr):
    """The text is re-derived at write time: the OCR pass deletes a sidecar
    when a re-run finds nothing, and the export must not keep the old claim."""
    rows = plan_export(ocr)
    (caption,) = [r for r in rows if r.kind == "caption"]
    assert caption.combined and caption.text.endswith(JA)
    Path(caption.ocr).unlink()
    from anime_tools.stages.export_workspace import _run

    _run(rows, apply=True)
    assert (ocr.out / "resized" / "a.txt").read_text(encoding="utf-8").endswith("cat.")


def test_a_combined_row_survives_the_report_and_reverts(ocr):
    rows, _ = run_export(ocr, apply=True)
    again = rows_from_report({"rows": [r.to_dict() for r in rows]})
    assert [(r.ocr, r.text) for r in again] == [(r.ocr, r.text) for r in rows]
    assert sum(r.combined for r in again) == 2
    back, stats = revert_export(again, apply=True)
    assert stats.removed == 6 and not (ocr.out / "resized" / "a.txt").exists()
    assert {r.status for r in back} == {"removed", "restored"}


def test_revert_checks_a_combined_row_against_what_it_recorded(ocr):
    """A sidecar rewritten after the export changes what a fresh combine would
    say, but the destination still holds what was published, so it reverts."""
    from anime_tools.captions.ocr_sidecar import OcrLine, write_ocr_for

    rows, _ = run_export(ocr, apply=True)
    write_ocr_for(
        ocr.ocr, Path("a.txt"), [OcrLine(seq=1, box=(0, 0, 1, 1), score=1, text="別")]
    )
    _, stats = revert_export(rows, apply=True)
    assert stats.skipped.get("drifted", 0) == 0
    assert not (ocr.out / "resized" / "a.txt").exists()
