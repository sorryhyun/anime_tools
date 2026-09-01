"""The one tab-delimited caption sidecar format, and the two vocabularies on it.

``{stem}.variants.txt`` and ``{stem}.history.txt`` are one format
(:mod:`anime_tools.captions._sidecar`) with two record shapes. These pin the
format's own rules once -- the multi-dot stem, the tolerance for a hand-edit,
the bounded split -- and then pin the two places the callers deliberately
*differ*, so a later tidy-up cannot quietly make one of them stricter or laxer
than the code it serves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anime_tools.captions._sidecar import (
    read_rows,
    sidecar_header,
    sidecar_path,
    write_rows,
)
from anime_tools.captions.history import (
    HISTORY_FIELDS,
    HISTORY_SIDECAR_SUFFIX,
    history_sidecar_path,
    push_history,
    read_history,
)
from anime_tools.captions.variants import (
    VARIANTS_FIELDS,
    VARIANTS_SIDECAR_SUFFIX,
    read_variants_sidecar,
    variants_sidecar_path,
    write_variants_sidecar,
)

# ---- the format -------------------------------------------------------------


def test_the_stem_keeps_its_dots():
    """``with_name``, not ``with_suffix``: ``a.b.png`` is not ``a``."""
    assert sidecar_path(Path("d/a.b.png"), ".variants.txt").name == "a.b.variants.txt"
    assert sidecar_path(Path("d/a.txt"), ".history.txt").name == "a.history.txt"
    # No extension at all: the whole name is the stem.
    assert sidecar_path(Path("d/a"), ".history.txt").name == "a.history.txt"
    # And the sidecar sits in the directory it was given, not the CWD.
    assert sidecar_path(Path("d/a.png"), ".variants.txt").parent == Path("d")


def test_both_public_path_builders_are_the_shared_rule():
    """Neither module gets to have its own opinion about the stem."""
    for p in (Path("d/a.png"), Path("d/a.b.c.txt"), Path("a")):
        assert variants_sidecar_path(p) == sidecar_path(p, VARIANTS_SIDECAR_SUFFIX)
        assert history_sidecar_path(p) == sidecar_path(p, HISTORY_SIDECAR_SUFFIX)


def test_the_header_is_a_comment_and_names_its_kind():
    assert sidecar_header("variants").startswith("#")
    assert "variants" in sidecar_header("variants")
    assert "history" in sidecar_header("history")
    # One line: a header that wrapped would parse as a second, malformed record.
    assert "\n" not in sidecar_header("history")


def test_a_damaged_sidecar_costs_the_sidecar_not_a_run(tmp_path: Path):
    """Blank, comment, short and long lines are skipped, never raised over."""
    p = tmp_path / "a.txt"
    lines = [
        "# a header",
        "",
        "   ",
        "no tab here",
        "  # an indented comment\ta\tb",
        "one\ttwo",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert read_rows(p, 2) == [["one", "two"]]
    # Read at a different arity, the very same line is the malformed one.
    assert read_rows(p, 3) == []


def test_a_crlf_sidecar_is_still_a_sidecar(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("# h\r\nv0\t1girl\r\n", encoding="utf-8")
    assert read_rows(p, 2) == [["v0", "1girl"]]


def test_the_split_is_bounded_so_the_last_field_keeps_its_tabs(tmp_path: Path):
    """Only the first ``fields - 1`` tabs are delimiters.

    A caption is comma-separated and does not contain tabs, but the format does
    not depend on that: the text rides last precisely so it cannot be cut.
    """
    p = tmp_path / "a.txt"
    write_rows(p, sidecar_header("history"), [("1", "now", "edit", "a\tb\tc")])
    assert read_rows(p, 4) == [["1", "now", "edit", "a\tb\tc"]]


def test_write_then_read_round_trips_and_makes_the_parent(tmp_path: Path):
    p = tmp_path / "deep" / "deeper" / "a.txt"
    rows = [("v0", "1girl"), ("v1", "solo, 1girl")]
    write_rows(p, sidecar_header("variants"), rows)
    assert read_rows(p, 2) == [list(r) for r in rows]
    text = p.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert text.endswith("\n")  # the trailing newline is part of the format


def test_an_empty_row_set_still_leaves_a_header(tmp_path: Path):
    """The format has no delete case -- that is history's own (see below)."""
    p = tmp_path / "a.txt"
    write_rows(p, sidecar_header("variants"), [])
    assert p.read_text(encoding="utf-8") == sidecar_header("variants") + "\n"
    assert read_rows(p, 2) == []


# ---- where the two vocabularies deliberately differ -------------------------


def test_a_missing_sidecar_is_no_history_but_a_bug_for_variants(tmp_path: Path):
    """The one contract difference between the two readers.

    A caption that has never been rewritten has no history, so ``read_history``
    answers with the empty list. Every ``read_variants_sidecar`` caller has
    already established the file is there before it asks, so a missing one is
    the caller's mistake and stays an error rather than an empty variant set.
    """
    missing = tmp_path / "nope.txt"
    assert read_history(missing) == []
    with pytest.raises(OSError):
        read_variants_sidecar(missing)


def test_only_history_deletes_itself_when_it_is_empty(tmp_path: Path):
    """A caption with no superseded versions has no sidecar; ``v0`` always is one."""
    cap = tmp_path / "a.txt"
    push_history(cap, "1girl", by="edit")
    assert history_sidecar_path(cap).is_file()

    from anime_tools.captions.history import write_history

    write_history(history_sidecar_path(cap), [])
    assert not history_sidecar_path(cap).exists()

    variants = variants_sidecar_path(tmp_path / "a.png")
    write_variants_sidecar(variants, [])
    assert variants.is_file()


def test_the_two_record_shapes_are_what_each_module_says_they_are(tmp_path: Path):
    """A history record is four fields, a variant two -- and the arities are not
    interchangeable in the direction that matters.

    Asked for *more* fields than a line carries, the reader drops it: a variants
    sidecar read as history is empty, which is why each module names its own
    arity rather than sniffing one. The other direction is not symmetric, and
    that is the bounded split working as designed rather than a leak -- asked
    for fewer, the surplus tabs simply ride along inside the last field.
    """
    assert (HISTORY_FIELDS, VARIANTS_FIELDS) == (4, 2)
    variants = variants_sidecar_path(tmp_path / "a.png")
    write_variants_sidecar(variants, [("v0", "1girl, solo")])
    assert read_rows(variants, HISTORY_FIELDS) == []

    cap = tmp_path / "a.txt"
    push_history(cap, "1girl, solo", by="edit")
    (row,) = read_rows(history_sidecar_path(cap), VARIANTS_FIELDS)
    assert row[0] == "1" and row[1].endswith("\tedit\t1girl, solo")


def test_a_variant_label_is_stripped_but_its_text_is_not(tmp_path: Path):
    """The label is a key the TE cache is addressed by; the text is a caption."""
    p = tmp_path / "a.variants.txt"
    p.write_text("# h\n  v0  \t 1girl, solo \n", encoding="utf-8")
    assert read_variants_sidecar(p) == [("v0", " 1girl, solo ")]
