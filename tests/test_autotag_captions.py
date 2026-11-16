"""Batch auto-tagging — mode policy and the merge invariants.

1. ``missing`` never touches an existing caption.
2. ``merge`` never re-flattens a bound tag: a tag inside a position clause counts
   as present, and clauses round-trip verbatim.
3. Only one rating survives a merge (``predict_caption`` always emits one).
4. Dry run writes nothing; ``apply`` defaults off.
5. The write lands on the **revised** caption beside the resized image; the
   master is read as the fallback and never written, and what a write replaced
   stays in ``{stem}.history.txt``.
6. "Unchanged" is measured against what spoke for the image, master included: a
   proposal equal to the master writes no revised copy of it.
7. ``exclude_tags`` takes its tags out of every proposal, keyed like any tag
   compare, ``*`` the one wildcard; a tag the caption already holds stays.
8. ``mask_dir`` tags a masked image with its ignored region painted white, and
   drops the background tags only the painted copy has.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anime_tools.stages.autotag import (
    BLANK_FILL,
    AutotagOptions,
    drop_blank_backgrounds,
    drop_excluded,
    merge_tags,
    painted_out,
    run_autotag_captions,
)


def _dataset(
    tmp_path: Path,
    images: dict[str, str | None],
    revised: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Build ``(resized_dir, source_dir)``; ``None`` caption → no sidecar.

    ``images`` fills the **master**; ``revised`` fills the resized tree, which is
    both what the stage prefers to read and the only tree it writes.
    """
    resized = tmp_path / "resized"
    source = tmp_path / "master"
    resized.mkdir()
    source.mkdir()
    for stem, caption in images.items():
        Image.new("RGB", (8, 8), (10, 20, 30)).save(resized / f"{stem}.png")
        if caption is not None:
            (source / f"{stem}.txt").write_text(caption, encoding="utf-8")
    for stem, caption in (revised or {}).items():
        (resized / f"{stem}.txt").write_text(caption, encoding="utf-8")
    return resized, source


def _run(resized, source, tagged, **kwargs):
    return run_autotag_captions(
        resized_dir=resized,
        source_dir=source,
        tag_batch=lambda images: [tagged] * len(images),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# merge_tags
# ---------------------------------------------------------------------------


def test_merge_appends_only_novel_tags():
    merged, added = merge_tags(
        "safe, 1girl, blue hair", "safe, 1girl, blue hair, smile, outdoors"
    )
    assert merged == "safe, 1girl, blue hair, smile, outdoors"
    assert added == ("smile", "outdoors")


def test_merge_preserves_clauses_and_treats_their_tags_as_present():
    """A clause-bound tag must not be copied back into the flat bag."""
    existing = "safe, 2girls. On the left, akita neru. On the right, kasane teto."
    merged, added = merge_tags(existing, "safe, 2girls, akita neru, smile")

    assert added == ("smile",)
    assert merged == (
        "safe, 2girls, smile. On the left, akita neru. On the right, kasane teto."
    )
    # The clause text round-trips byte-for-byte.
    assert merged.endswith("On the left, akita neru. On the right, kasane teto.")


def test_merge_drops_a_second_rating():
    merged, added = merge_tags("safe, 1girl", "explicit, 1girl, nude")
    assert added == ("nude",)
    assert merged == "safe, 1girl, nude"
    assert "explicit" not in merged


def test_merge_keeps_the_rating_when_the_caption_has_none():
    merged, added = merge_tags("1girl, blue hair", "sensitive, 1girl, smile")
    assert added == ("sensitive", "smile")
    assert merged == "1girl, blue hair, sensitive, smile"


def test_merge_reads_the_two_underscore_spellings_as_one_tag():
    """The master's ``speech_bubble`` and the tagger's ``speech bubble`` are one
    tag under :func:`normalize_tag`; the caption's own spelling stays."""
    merged, added = merge_tags(
        "1girl, long_hair, speech_bubble",
        "1girl, long hair, speech bubble, blue eyes",
    )
    assert added == ("blue eyes",)
    assert merged == "1girl, long_hair, speech_bubble, blue eyes"
    # …and the other direction.
    assert merge_tags("1girl, long hair", "1girl, long_hair")[1] == ()


def test_merge_treats_an_underscored_clause_tag_as_present():
    """The clause half of the same key: a bound tag is present in either spelling."""
    existing = "safe, 2girls. On the left, long_hair."
    merged, added = merge_tags(existing, "safe, 2girls, long hair, smile")
    assert added == ("smile",)
    assert merged == "safe, 2girls, smile. On the left, long_hair."


def test_merge_is_case_insensitive_and_idempotent():
    first, added = merge_tags("safe, 1girl, Blue Hair", "safe, 1girl, blue hair")
    assert added == ()
    assert first == "safe, 1girl, Blue Hair"
    second, added_again = merge_tags(first, "safe, 1girl, blue hair, smile")
    assert added_again == ("smile",)
    assert merge_tags(second, "safe, smile")[1] == ()


# ---------------------------------------------------------------------------
# run_autotag_captions
# ---------------------------------------------------------------------------


def test_missing_mode_only_fills_gaps(tmp_path):
    resized, source = _dataset(
        tmp_path, {"a": "hand written, 1girl", "b": None, "c": None}
    )

    rows, stats = _run(
        resized,
        source,
        "safe, 1girl, smile",
        options=AutotagOptions(mode="missing"),
        apply=True,
    )

    assert stats.seen == 3
    assert stats.candidates == 2
    assert stats.written == 2
    assert stats.skipped["has-caption"] == 1
    # A master speaks for `a`, so `missing` skips it — and the master is never
    # written, here or anywhere.
    assert (source / "a.txt").read_text(encoding="utf-8") == "hand written, 1girl"
    assert not (resized / "a.txt").exists()
    assert (resized / "b.txt").read_text(encoding="utf-8") == "safe, 1girl, smile"
    assert not (source / "b.txt").exists()
    assert {r.caption_path for r in rows} == {"b.txt", "c.txt"}


def test_overwrite_mode_replaces_every_caption(tmp_path):
    resized, source = _dataset(tmp_path, {"a": "hand written, 1girl", "b": None})

    _, stats = _run(
        resized,
        source,
        "safe, 1girl, smile",
        options=AutotagOptions(mode="overwrite"),
        apply=True,
    )

    assert stats.written == 2
    assert (resized / "a.txt").read_text(encoding="utf-8") == "safe, 1girl, smile"
    # The hand-written master is what `overwrite` used to destroy.
    assert (source / "a.txt").read_text(encoding="utf-8") == "hand written, 1girl"


def test_merge_mode_leaves_a_saturated_revised_caption_untouched(tmp_path):
    """No novel tags and the revised caption already holds them → no write."""
    resized, source = _dataset(
        tmp_path, {"a": None}, revised={"a": "safe, 1girl, smile"}
    )

    rows, stats = _run(
        resized,
        source,
        "safe, 1girl, smile",
        options=AutotagOptions(mode="merge"),
        apply=True,
    )

    assert stats.written == 0
    assert stats.skipped["unchanged"] == 1
    assert rows[0].status == "skip:unchanged"


def test_a_proposal_equal_to_the_master_writes_nothing(tmp_path):
    """ "Unchanged" is measured against what spoke, master included.

    A master the tagger itself wrote reproduces itself exactly, and a revised
    copy of it says nothing new — it only gives a later hand-edit of the master
    a second spelling to go stale behind. Export reads the same ladder, so the
    image publishes that text either way.
    """
    resized, source = _dataset(tmp_path, {"a": "safe, 1girl, smile"})

    rows, stats = _run(
        resized,
        source,
        "safe, 1girl, smile",
        options=AutotagOptions(mode="overwrite"),
        apply=True,
    )

    assert stats.written == 0
    assert stats.skipped["unchanged"] == 1
    assert rows[0].status == "skip:unchanged"
    assert not (resized / "a.txt").exists()
    assert (source / "a.txt").read_text(encoding="utf-8") == "safe, 1girl, smile"


def test_a_saturated_merge_into_a_master_writes_nothing(tmp_path):
    """The merge adds nothing, so the merged text is the master verbatim — and
    a revised caption holding exactly the master is not worth creating."""
    resized, source = _dataset(tmp_path, {"a": "safe, 1girl, smile"})

    rows, stats = _run(
        resized,
        source,
        "safe, 1girl, smile",
        options=AutotagOptions(mode="merge"),
        apply=True,
    )

    assert stats.written == 0
    assert rows[0].status == "skip:unchanged"
    assert not (resized / "a.txt").exists()


def test_a_merge_that_adds_a_tag_to_a_master_creates_the_revised_caption(tmp_path):
    """The other half of the rule: novel tags mean the merged text differs from
    the master, and that text has to land somewhere — the revised tree."""
    resized, source = _dataset(tmp_path, {"a": "safe, 1girl"})

    rows, stats = _run(
        resized,
        source,
        "safe, 1girl, smile",
        options=AutotagOptions(mode="merge"),
        apply=True,
    )

    assert stats.written == 1
    assert rows[0].added == ("smile",)
    assert (resized / "a.txt").read_text(encoding="utf-8") == "safe, 1girl, smile"
    # The master is the read-only fallback, and nothing was replaced.
    assert (source / "a.txt").read_text(encoding="utf-8") == "safe, 1girl"
    assert not (resized / "a.history.txt").exists()


def test_dry_run_writes_nothing(tmp_path):
    resized, source = _dataset(tmp_path, {"a": None})

    rows, stats = _run(
        resized, source, "safe, 1girl", options=AutotagOptions(mode="missing")
    )

    assert stats.proposed == 1
    assert stats.written == 0
    assert not (resized / "a.txt").exists()
    assert rows[0].proposed == "safe, 1girl"


def test_missing_mode_creates_nested_caption_dirs(tmp_path):
    """The caption lands beside its image, however deep the subdir."""
    resized = tmp_path / "resized"
    source = tmp_path / "master"
    (resized / "artist_x").mkdir(parents=True)
    source.mkdir()
    Image.new("RGB", (8, 8)).save(resized / "artist_x" / "a.png")

    _, stats = _run(resized, source, "safe, 1girl", apply=True)

    assert stats.written == 1
    assert (resized / "artist_x" / "a.txt").read_text(encoding="utf-8") == "safe, 1girl"


def test_empty_tagger_output_is_a_skip_not_a_blank_caption(tmp_path):
    resized, source = _dataset(tmp_path, {"a": None})

    rows, stats = _run(resized, source, "  ", apply=True)

    assert stats.written == 0
    assert stats.skipped["no-tags"] == 1
    assert rows[0].status == "skip:no-tags"
    assert not (resized / "a.txt").exists()


def test_a_replaced_caption_is_kept_as_a_history_version(tmp_path):
    """No mode loses text: what a write replaces is one badge away."""
    from anime_tools.captions.history import history_sidecar_path, read_history

    resized, source = _dataset(
        tmp_path, {"a": "hand written, 1girl"}, revised={"a": "safe, 1girl"}
    )

    _, stats = _run(
        resized,
        source,
        "safe, 1girl, smile",
        options=AutotagOptions(mode="overwrite"),
        apply=True,
    )

    assert stats.written == 1
    entries = read_history(history_sidecar_path(resized / "a.txt"))
    assert [(e.seq, e.by, e.text) for e in entries] == [(1, "autotag", "safe, 1girl")]
    # The master keeps no history of its own: nothing wrote it.
    assert not history_sidecar_path(source / "a.txt").exists()


def test_merge_reads_the_revised_caption_over_the_master(tmp_path):
    """The clauses live in the revised caption, so that is what a merge merges
    into — reading the master would re-flatten a bound tag into the bag."""
    resized, source = _dataset(
        tmp_path,
        {"a": "safe, 2girls, akita neru"},
        revised={"a": "safe, 2girls. On the left, akita neru."},
    )

    rows, stats = _run(
        resized,
        source,
        "safe, 2girls, akita neru, smile",
        options=AutotagOptions(mode="merge"),
        apply=True,
    )

    assert stats.written == 1
    assert rows[0].added == ("smile",)
    assert (resized / "a.txt").read_text(encoding="utf-8") == (
        "safe, 2girls, smile. On the left, akita neru."
    )


def test_a_merge_into_a_master_records_the_absent_target_not_the_master(tmp_path):
    """``existing`` is what spoke for the image; ``target_before`` is what the
    file being written held. They differ exactly when the revised caption is
    being created, and a replay drifts unless it gates on the second."""
    resized, source = _dataset(tmp_path, {"a": "safe, 1girl"})

    rows, _stats = _run(
        resized, source, "safe, 1girl, smile", options=AutotagOptions(mode="merge")
    )

    assert rows[0].existing == "safe, 1girl"
    assert rows[0].target_before == ""


def test_bad_options_fail_at_construction():
    with pytest.raises(ValueError):
        AutotagOptions(mode="clobber")
    with pytest.raises(ValueError):
        AutotagOptions(min_confidence=1.5)


def test_the_tagger_sees_batches_and_never_a_skipped_image(tmp_path):
    """The forward is batched, and the first half of the pass is what decides
    which images it covers: an image `missing` mode skips is never decoded."""
    resized, source = _dataset(
        tmp_path,
        {"a": "hand written", "b": None, "c": None, "d": None, "e": None},
    )
    batches: list[int] = []

    def tag_batch(images):
        batches.append(len(images))
        return ["safe, 1girl"] * len(images)

    _rows, stats = run_autotag_captions(
        resized_dir=resized,
        source_dir=source,
        tag_batch=tag_batch,
        options=AutotagOptions(mode="missing"),
        batch_size=3,
        apply=True,
    )

    # Four candidates (`a` has a master), so 3 + 1 — not five calls of one.
    assert batches == [3, 1]
    assert stats.candidates == 4
    assert stats.written == 4
    assert not (resized / "a.txt").exists()


def test_progress_still_counts_every_image_in_tree_order(tmp_path):
    """Batching moved the forward, not the bar: the GUI's `[done/total]` line
    is per image walked, skipped ones included."""
    resized, source = _dataset(tmp_path, {"a": "hand written", "b": None, "c": None})
    seen: list[tuple[int, int, str]] = []

    _run(
        resized,
        source,
        "safe, 1girl",
        options=AutotagOptions(mode="missing"),
        batch_size=1,
        progress=lambda i, n, detail: seen.append((i, n, detail)),
    )

    assert seen == [(1, 3, "a.txt"), (2, 3, "b.txt"), (3, 3, "c.txt")]


# ---------------------------------------------------------------------------
# the master overlay
# ---------------------------------------------------------------------------


def test_a_master_edited_in_the_gui_is_what_the_image_already_says(tmp_path):
    """The GUI's caption editor writes a master edit to the overlay
    (``workspace/master/``), not to ``--src``. ``missing`` must count that as a
    caption, and ``merge`` must merge into it rather than into the stale
    hand-written text behind it."""
    resized, source = _dataset(tmp_path, {"a": "1girl", "b": None})
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "a.txt").write_text("1girl, smile", encoding="utf-8")
    (overlay / "b.txt").write_text("1boy", encoding="utf-8")

    rows, stats = _run(resized, source, "1girl, hat", master_dir=overlay)
    assert stats.candidates == 0 and stats.skipped["has-caption"] == 2

    rows, _ = _run(
        resized,
        source,
        "1girl, hat",
        master_dir=overlay,
        options=AutotagOptions(mode="merge"),
    )
    by_image = {r.image: r for r in rows}
    assert by_image["a.png"].existing == "1girl, smile"
    assert by_image["a.png"].proposed == "1girl, smile, hat"
    # The write would create the revised caption: nothing there yet.
    assert by_image["a.png"].target_before == ""


# ---------------------------------------------------------------------------
# exclude_tags
# ---------------------------------------------------------------------------


def test_exclude_keys_on_the_normalized_tag_and_globs_on_star():
    kept, dropped = drop_excluded(
        "safe, 1girl, speech bubble, english text, japanese text, smile, text focus",
        ["Speech_Bubble", "*text"],
    )
    assert kept == "safe, 1girl, smile, text focus"
    assert dropped == ("speech bubble", "english text", "japanese text")


def test_exclude_treats_brackets_and_question_marks_literally():
    kept, dropped = drop_excluded("safe, ?, [abc], a", ["?", "[abc]"])
    assert kept == "safe, a"
    assert dropped == ("?", "[abc]")


def test_overwrite_with_exclude_never_writes_the_excluded_tag(tmp_path):
    resized, source = _dataset(tmp_path, {"a": "safe, speech bubble"})
    rows, _ = _run(
        resized,
        source,
        "safe, 1girl, speech bubble",
        options=AutotagOptions(mode="overwrite"),
        exclude_tags=("speech bubble",),
        apply=True,
    )
    assert (resized / "a.txt").read_text(encoding="utf-8") == "safe, 1girl"
    assert rows[0].excluded == ("speech bubble",)


def test_merge_with_exclude_keeps_what_the_caption_already_holds(tmp_path):
    resized, source = _dataset(tmp_path, {"a": "safe, speech bubble"})
    rows, _ = _run(
        resized,
        source,
        "safe, 1girl, speech bubble, english text",
        options=AutotagOptions(mode="merge"),
        exclude_tags=("speech bubble", "*text"),
    )
    assert rows[0].proposed == "safe, speech bubble, 1girl"


# ---------------------------------------------------------------------------
# mask-aware tagging
# ---------------------------------------------------------------------------


def _mask(path: Path, ignore_left: bool, size=(8, 8)) -> None:
    alpha = Image.new("L", size, 255)
    if ignore_left:
        alpha.paste(0, (0, 0, size[0] // 2, size[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha.save(path)


def test_painted_out_blanks_only_the_ignored_region(tmp_path):
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    _mask(tmp_path / "m_mask.png", ignore_left=True, size=(4, 4))
    painted = painted_out(image, tmp_path / "m_mask.png")
    assert painted.getpixel((0, 0)) == BLANK_FILL
    assert painted.getpixel((7, 7)) == (10, 20, 30)
    _mask(tmp_path / "n_mask.png", ignore_left=False)
    assert painted_out(image, tmp_path / "n_mask.png") is None
    assert painted_out(image, None) is None


def test_blank_backgrounds_survive_only_when_the_plain_image_has_them():
    assert (
        drop_blank_backgrounds(
            "safe, 1girl, white background, simple background, blue background",
            "safe, 1girl, blue background, speech bubble",
        )
        == "safe, 1girl, blue background"
    )


def test_a_masked_image_is_tagged_painted_and_plain(tmp_path):
    resized, source = _dataset(tmp_path, {"a": None, "b": None})
    masks = tmp_path / "masks"
    _mask(masks / "a_mask.png", ignore_left=True)
    seen: list[tuple[int, int, int]] = []

    def tag_batch(images):
        seen.extend(im.getpixel((0, 0)) for im in images)
        return [
            "safe, 1girl, white background, simple background"
            if im.getpixel((0, 0)) == BLANK_FILL
            else "safe, 1girl, speech bubble, outdoors"
            for im in images
        ]

    rows, stats = run_autotag_captions(
        resized_dir=resized,
        source_dir=source,
        tag_batch=tag_batch,
        mask_dir=masks,
    )
    by_image = {r.image: r for r in rows}
    assert by_image["a.png"].proposed == "safe, 1girl"
    assert by_image["a.png"].masked
    assert by_image["b.png"].proposed == "safe, 1girl, speech bubble, outdoors"
    assert not by_image["b.png"].masked
    assert stats.masked == 1
    # One forward: both plain images, then a's painted copy.
    assert seen == [(10, 20, 30), (10, 20, 30), BLANK_FILL]
