"""The correction pass reads the revised caption first, the master as fallback.

That is the rule every caption stage follows (``stages/_walk_captions.py``);
the corrector is the one that used to re-mirror the master over what autotag
and the position rewrite had written into the revised caption.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from anime_tools.captions.correction import (
    CaptionCorrectionOptions,
    load_tag_knowledge_base,
)
from anime_tools.captions.position_clauses import parse_caption
from anime_tools.captions.taxonomy import normalize_tag
from anime_tools.captions.variants import variants_sidecar_path
from anime_tools.stages.autotag import AutotagOptions, run_autotag_captions
from anime_tools.stages.captions import write_corrected_preprocess_captions


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    csv = tmp_path_factory.mktemp("kb") / "danbooru_tags_classified.csv"
    csv.write_text(
        """name,category,post_count,description
1girl,0,10,"[인물 > 인원수] count"
solo,0,10,"[인물 > 인원수] count"
hatsune_miku,4,10,"[캐릭터 > vocaloid] character"
long_hair,0,10,"[머리카락 > 머리 길이] general"
smile,0,10,"[표정 > 입] general"
blue_eyes,0,10,"[눈 > 색] general"
""",
        encoding="utf-8",
    )
    return load_tag_knowledge_base(csv)


def _dataset(
    tmp_path: Path,
    master: dict[str, str | None],
    revised: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """``(resized_dir, source_dir)``; ``None`` → an image with no master caption."""
    resized = tmp_path / "resized"
    source = tmp_path / "master"
    resized.mkdir()
    source.mkdir()
    for stem, caption in master.items():
        Image.new("RGB", (8, 8), (10, 20, 30)).save(resized / f"{stem}.png")
        if caption is not None:
            (source / f"{stem}.txt").write_text(caption, encoding="utf-8")
    for stem, caption in (revised or {}).items():
        (resized / f"{stem}.txt").write_text(caption, encoding="utf-8")
    return resized, source


def _correct(resized, source, kb, *, options=None, **kwargs):
    # ``insert_no_artist=False`` keeps the sentinel out of the tag-set checks.
    options = options or CaptionCorrectionOptions(insert_no_artist=False)
    return write_corrected_preprocess_captions(
        source, resized, kb, options=options, **kwargs
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def test_autotag_merge_then_correct_keeps_the_merged_tags(tmp_path, kb):
    """The beta-gate chain: ``autotag --mode merge --apply`` then ``correct``.
    Autotag writes the revised caption; a correction that re-mirrored the
    master would drop every tag it added."""
    resized, source = _dataset(tmp_path, {"a": "1girl, long_hair"})
    run_autotag_captions(
        resized_dir=resized,
        source_dir=source,
        tag_fn=lambda _img: "1girl, long_hair, smile",
        options=AutotagOptions(mode="merge"),
        apply=True,
    )
    assert "smile" in parse_caption(_read(resized / "a.txt")).tag_keys

    stats = _correct(resized, source, kb)

    parsed = parse_caption(_read(resized / "a.txt"))
    assert "smile" in parsed.tag_keys
    assert "long hair" in parsed.tag_keys
    assert stats.from_master == 0
    assert _read(source / "a.txt") == "1girl, long_hair"


def test_a_caption_autotag_created_from_nothing_survives_correct(tmp_path, kb):
    """``autotag --mode missing`` captions an image that has no master; the
    correction must not treat the absent master as a vanished source."""
    resized, source = _dataset(tmp_path, {"a": None})
    run_autotag_captions(
        resized_dir=resized,
        source_dir=source,
        tag_fn=lambda _img: "1girl, smile",
        options=AutotagOptions(mode="missing"),
        apply=True,
    )
    assert (resized / "a.txt").exists()

    stats = _correct(resized, source, kb)

    assert (resized / "a.txt").exists()
    assert parse_caption(_read(resized / "a.txt")).tag_keys == {"1girl", "smile"}
    assert stats.no_caption == 0
    assert stats.written + stats.unchanged == 1


def test_a_position_clause_on_the_revised_caption_survives_correction(tmp_path, kb):
    """The rewrite moved ``long_hair`` out of the bag into a clause. Correcting
    the revised caption reorders the bag around the clause and never re-asserts
    the moved tag from the master."""
    resized, source = _dataset(
        tmp_path,
        {"a": "long_hair, 1girl, hatsune_miku"},
        revised={"a": "1girl, hatsune_miku. On the left, long_hair."},
    )

    stats = _correct(resized, source, kb)

    out = parse_caption(_read(resized / "a.txt"))
    assert [c.tags for c in out.clauses] == [("long_hair",)]
    assert "long hair" not in out.tag_keys
    assert {normalize_tag(t) for t in out.flat_tags} == {"1girl", "hatsune miku"}
    assert stats.clauses_preserved == 1


def test_the_master_is_mirrored_for_an_image_without_a_revised_caption(tmp_path, kb):
    resized, source = _dataset(tmp_path, {"a": "long_hair, hatsune_miku, 1girl"})

    stats = _correct(resized, source, kb)

    assert (resized / "a.txt").exists()
    assert stats.from_master == 1 and stats.written == 1
    assert parse_caption(_read(resized / "a.txt")).tag_keys == {
        "1girl",
        "hatsune miku",
        "long hair",
    }
    # The master is never edited.
    assert _read(source / "a.txt") == "long_hair, hatsune_miku, 1girl"


def test_a_master_edit_does_not_reach_an_image_with_a_revised_caption(tmp_path, kb):
    """The revised caption is the editable one once it exists. Delete it to
    re-mirror the master."""
    resized, source = _dataset(tmp_path, {"a": "1girl, long_hair"})
    _correct(resized, source, kb)
    before = _read(resized / "a.txt")
    (source / "a.txt").write_text("1girl, long_hair, blue_eyes", encoding="utf-8")

    stats = _correct(resized, source, kb)

    assert _read(resized / "a.txt") == before
    assert stats.unchanged == 1 and stats.from_master == 0

    (resized / "a.txt").unlink()
    stats = _correct(resized, source, kb)
    assert stats.from_master == 1
    assert "blue eyes" in parse_caption(_read(resized / "a.txt")).tag_keys


def test_an_image_with_no_caption_is_counted_and_its_orphan_sidecar_dropped(
    tmp_path, kb
):
    resized, source = _dataset(tmp_path, {"a": None})
    sidecar = variants_sidecar_path(resized / "a.txt")
    sidecar.write_text("v0\tstale\n", encoding="utf-8")

    stats = _correct(resized, source, kb)

    assert stats.no_caption == 1 and stats.written == 0
    assert not (resized / "a.txt").exists()
    assert not sidecar.exists()
    assert stats.variants_removed == 1


@pytest.mark.parametrize("insert_no_artist", [False, True])
def test_a_second_run_is_a_no_op(tmp_path, kb, insert_no_artist):
    """Neither the caption nor the variant sidecar is rewritten when nothing
    changed — a rewritten sidecar is a needless TE re-encode. With the
    ``@no-artist`` sentinel in the caption, ``v0`` is the caption *minus* the
    sentinel, and the currency check has to compare against that."""
    resized, source = _dataset(tmp_path, {"a": "long_hair, hatsune_miku, 1girl"})
    options = CaptionCorrectionOptions(insert_no_artist=insert_no_artist)
    first = _correct(resized, source, kb, options=options, num_variants=2)
    assert first.written == 1 and first.variants_written == 1
    sidecar = variants_sidecar_path(resized / "a.txt")
    stamp = sidecar.stat().st_mtime_ns

    second = _correct(resized, source, kb, options=options, num_variants=2)

    assert second.unchanged == 1 and second.written == 0
    assert second.variants_written == 0
    assert sidecar.stat().st_mtime_ns == stamp
