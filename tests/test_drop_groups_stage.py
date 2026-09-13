"""The Tag groups stage (GH #95): whole groups of tags cut from the revised
captions, and nothing else about the caption moved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from anime_tools.captions.correction import (
    CaptionCorrectionOptions,
    drop_caption_groups,
    load_tag_knowledge_base,
)
from anime_tools.captions.history import history_sidecar_path
from anime_tools.captions.variants import variants_sidecar_path
from anime_tools.contract import REPLAY_SHAPES
from anime_tools.stages.captions import write_corrected_preprocess_captions
from anime_tools.stages.drop_groups import drop_tag_groups, run_drop_groups
from anime_tools.stages.requests import DropGroupRequest


@pytest.fixture(scope="module")
def kb_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("kb") / "danbooru_tags_classified.csv"
    path.write_text(
        """name,category,post_count,description
1girl,0,10,"[인물 > 인원수] count"
sincos,1,10,"[아티스트 > 개인] artist"
long_hair,0,10,"[머리카락 > 머리 길이] hair"
school_uniform,0,10,"[의상 > 상의] clothing"
backlighting,0,10,"[효과/연출 > 조명] lighting"
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def kb(kb_csv):
    return load_tag_knowledge_base(kb_csv)


def _dataset(
    tmp_path: Path,
    master: dict[str, str],
    revised: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """``(resized_dir, source_dir)``."""
    resized = tmp_path / "resized"
    source = tmp_path / "master"
    resized.mkdir()
    source.mkdir()
    for stem, caption in master.items():
        Image.new("RGB", (8, 8), (10, 20, 30)).save(resized / f"{stem}.png")
        (source / f"{stem}.txt").write_text(caption, encoding="utf-8")
    for stem, caption in (revised or {}).items():
        (resized / f"{stem}.txt").write_text(caption, encoding="utf-8")
    return resized, source


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


# ---- the cut ------------------------------------------------------------------


def test_the_cut_leaves_every_other_tag_where_it_was(kb):
    """No bucket order: Correct would put ``1girl`` first, the cut does not."""
    out = drop_caption_groups(
        "long_hair, @sincos, 1girl, backlighting, mystery", kb, ("artist", "lighting")
    )
    assert out.text == "long_hair, 1girl, mystery"
    assert out.dropped_tags == ("@sincos", "backlighting")


def test_nothing_to_drop_is_the_caption_as_it_arrived(kb):
    text = "1girl,long_hair ,  school uniform"
    assert drop_caption_groups(text, kb, ("artist",)).text == text
    assert drop_caption_groups(text, kb, ()).text == text


def test_a_clause_loses_its_tags_and_an_emptied_clause_goes(kb):
    text = (
        "1girl, long hair. On the left, backlighting. "
        "On the right, school uniform, backlighting."
    )
    out = drop_caption_groups(text, kb, ("lighting",))
    assert out.text == "1girl, long hair. On the right, school uniform."


def test_keep_protects_a_trigger_word_that_is_itself_an_artist(kb):
    out = drop_caption_groups("@trig, @sincos, 1girl", kb, ("artist",), keep=("@trig",))
    assert out.text == "@trig, 1girl"


def test_a_tag_the_kb_does_not_know_is_never_dropped(kb):
    text = "mystery, 1girl"
    assert drop_caption_groups(text, kb, ("clothing", "hair", "effect")).text == text


def test_no_correct_still_drops_the_groups(tmp_path, kb):
    """``--no_correct`` turns the reorder off, not the drop — it used to mirror
    the raw caption with ``--caption_drop_groups`` silently ignored."""
    resized, source = _dataset(tmp_path, {"a": "long_hair, @sincos, 1girl"})
    write_corrected_preprocess_captions(
        source,
        resized,
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=False, drop_groups=("artist",)
        ),
        correct=False,
    )
    assert _read(resized / "a.txt") == "long_hair, 1girl"


# ---- the stage ----------------------------------------------------------------


def test_a_run_cuts_the_revised_caption_and_leaves_the_master(tmp_path, kb):
    resized, source = _dataset(
        tmp_path,
        {"a": "1girl, @sincos, long_hair", "b": "1girl, long_hair"},
        revised={"a": "1girl, @sincos, long_hair, backlighting"},
    )
    variants_sidecar_path(resized / "a.txt").write_text("v0\t1girl\n", encoding="utf-8")

    rows, stats = drop_tag_groups(source, resized, kb, ("artist",))

    assert _read(resized / "a.txt") == "1girl, long_hair, backlighting"
    assert _read(source / "a.txt") == "1girl, @sincos, long_hair"
    # The replaced text is a version, and the sidecar drawn from it is gone.
    assert history_sidecar_path(resized / "a.txt").exists()
    assert not variants_sidecar_path(resized / "a.txt").exists()
    # Nothing to cut in b: no revised copy of its master appears.
    assert not (resized / "b.txt").exists()

    by = {r.image: r for r in rows}
    assert by["a.png"].status == "ok" and by["a.png"].dropped == ["@sincos"]
    assert by["b.png"].status == "skip:unchanged"
    assert (stats.written, stats.unchanged, stats.from_master) == (1, 1, 0)
    assert stats.by_group == {"artist": 1}


def test_a_cut_of_a_master_creates_the_revised_caption(tmp_path, kb):
    """The row whose ``target_before`` is empty is the one Undo deletes."""
    resized, source = _dataset(tmp_path, {"a": "1girl, backlighting"})
    rows, stats = drop_tag_groups(source, resized, kb, ("lighting",))
    assert _read(resized / "a.txt") == "1girl"
    assert rows[0].target_before == "" and stats.from_master == 1


def test_a_dry_run_reports_the_cut_and_writes_nothing(tmp_path, kb):
    resized, source = _dataset(tmp_path, {"a": "1girl, backlighting"})
    rows, stats = drop_tag_groups(source, resized, kb, ("lighting",), apply=False)
    assert rows[0].proposed == "1girl" and stats.written == 1
    assert not (resized / "a.txt").exists()


def test_the_report_is_the_shape_undo_reads(tmp_path, kb_csv, capsys):
    resized, source = _dataset(tmp_path, {"a": "1girl, @sincos"})
    report_dir = tmp_path / "rep"
    run_drop_groups(
        DropGroupRequest(
            src=str(source),
            dst=str(resized),
            groups=("artist",),
            tag_csv=str(kb_csv),
            report_dir=str(report_dir),
            apply=True,
        )
    )
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    spec = REPLAY_SHAPES["drop_groups"]
    (row,) = report[spec.rows_key]
    assert row["status"] == spec.ok_status
    assert (row[spec.before_field], row[spec.after_field]) == ("", "1girl")
    assert report[spec.stats_key]["by_group"] == {"artist": 1}
    assert "artist: 1 tag" in capsys.readouterr().out


def test_a_run_with_nothing_picked_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Nothing to drop"):
        run_drop_groups(DropGroupRequest(src=str(tmp_path), dst=str(tmp_path)))


def test_groups_are_one_flag_of_known_slugs():
    req = DropGroupRequest(
        groups=("artist", "lighting"), category_paths=("의상 > 상의",)
    )
    assert req.to_argv()[:3] == ["--groups", "artist", "lighting"]
    assert req.selectors == ("artist", "lighting", "의상 > 상의")
    with pytest.raises(SystemExit):
        DropGroupRequest.parser().parse_args(["--groups", "nope"])
