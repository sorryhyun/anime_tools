"""What a caption claims about its own layout — counts, panels, candidacy.

The text-only prefilter for the position-clause pipeline
(:mod:`anime_tools.stages.position_captions`) and the multiview audit: how many
bindable subjects the caption claims, whether it describes one character drawn
several times (``multiple views`` / a comic page), and whether it is worth
running a detector over at all.

Kept apart from :mod:`anime_tools.captions.position_clauses` (pure grammar) and
:mod:`anime_tools.captions.clause_vocabulary` (tag-group taxonomy) because this is
a third thing: a small *vocabulary* of count and layout tags read off the flat
bag. Pure stdlib — no model, no pixels.
"""

from __future__ import annotations

import re

from anime_tools.captions.position_clauses import flat_tag_set, has_clauses
from anime_tools.captions.taxonomy import count_of

# ``2koma`` / ``4koma`` name the panel count. Deliberately anchored, so the
# open-ended ``multiple 4koma`` does not match and stays unbounded.
KOMA_COUNT_RE = re.compile(r"^(\d+)koma$")
MULTI_VIEW_TAGS = frozenset({"multiple views", "multiple_views"})

# Panel layouts: a comic page draws the same character once per panel, so like
# `multiple views` its girls-count counts *characters*, not bindable subjects —
# `1girl, 2koma` is routinely two. Without this, comic pages fail the candidate
# prefilter as `single-subject`.
#
# `page number` is deliberately EXCLUDED: it marks a scanned art-book page, not
# a layout (checked — every image it catches is a single illustration with a
# margin number), so it's a false signal, not a weak one.
PANEL_LAYOUT_TAGS = frozenset(
    {
        "comic",
        "silent comic",
        "silent_comic",
        "sequential",
        "2koma",
        "3koma",
        "4koma",
        "multiple 4koma",
        "multiple_4koma",
    }
)

# Every layout tag that decouples the girls-count from the bindable-subject
# count. Both branches of the prefilter and the count check read this.
LAYOUT_TAGS = MULTI_VIEW_TAGS | PANEL_LAYOUT_TAGS


def caption_subject_count(caption: str) -> int | None:
    """How many bindable subjects the caption itself claims, if it says.

    ``Ngirls`` gives a number; ``None`` means "more than one, count unknown"
    (the count-consistency check then trusts detection instead of skipping).

    A layout tag (:data:`LAYOUT_TAGS`) always forces ``None`` even alongside a
    girls-count, because that count tags *characters* while each view/panel is
    its own bindable subject (``1girl, multiple views`` is routinely four).
    ``multiple girls`` / open-ended ``N+girls`` are ``None`` too — an exact
    match against "six or more" can only fail; that part is
    :func:`~anime_tools.captions.taxonomy.count_of`'s rule, shared with the
    boys count and the panel ceiling below.
    """
    tags = flat_tag_set(caption)
    if tags & LAYOUT_TAGS:
        return None
    return count_of(tags, "girl")


def caption_panel_ceiling(caption: str) -> int | None:
    """Most bindable subjects an ``Nkoma`` page can hold, or ``None`` if unbounded.

    A layout tag makes :func:`caption_subject_count` return ``None``, waiving
    the count check entirely — this restores a backstop for a comic page so a
    subject detected twice (e.g. by a shredded overlapping-mask split) still
    gets caught. ``Nkoma`` names the panel count, so the ceiling is
    ``panels x (girls + boys)`` (generous by construction: every panel drawing
    every character at once). Plain ``comic`` / ``multiple views`` carry no
    panel count and stay unbounded.

    ``None`` whenever any term is unknown — an unbounded check can only produce
    false skips.
    """
    tags = flat_tag_set(caption)
    panels = [int(m.group(1)) for t in tags if (m := KOMA_COUNT_RE.match(t))]
    if not panels:
        return None
    girls = count_of(tags, "girl")
    boys = count_of(tags, "boy")
    if girls is None or boys is None:
        return None
    # A page with no counted character at all still draws somebody per panel.
    per_panel = max(girls + boys, 1)
    return max(panels) * per_panel


def caption_boy_count(caption: str) -> int | None:
    """How many *male* subjects the caption claims — the count check's slack.

    The SAM3 ``girl`` prompt does not reliably exclude males, so the count gate
    accepts the range ``girls .. girls + boys`` rather than equality. ``None`` =
    "some boys, count unknown", which drops the upper bound entirely.
    """
    return count_of(flat_tag_set(caption), "boy")


def is_repeated_subject_layout(caption: str) -> bool:
    """Is this one character drawn several times, rather than several characters?

    Any :data:`LAYOUT_TAGS` member says yes — an ``Nkoma``/``comic`` page is the
    same situation as ``multiple views`` panel-by-panel: the girl in panel 3 is
    the girl in panel 1, so whatever belongs to *her* discriminates nothing
    between panels.
    :meth:`anime_tools.captions.clause_vocabulary.ClauseVocabulary.select` drops
    the whole class (``view_invariant``). A comic can introduce a new character
    mid-page, which only makes a bound trait *sometimes* wrong instead of never
    — the bag keeps every suppressed tag regardless, so only the per-panel
    binding is lost.
    """
    return bool(flat_tag_set(caption) & LAYOUT_TAGS)


def is_candidate(caption: str) -> tuple[bool, str]:
    """Should this caption go through detection? Returns ``(ok, reason)``."""
    if has_clauses(caption):
        return False, "already-has-clauses"
    tags = flat_tag_set(caption)
    if tags & MULTI_VIEW_TAGS:
        return True, "multiple-views"
    if tags & PANEL_LAYOUT_TAGS:
        return True, "panel-layout"
    expected = caption_subject_count(caption)
    if expected is None or expected > 1:
        return True, "multi-girl"
    return False, "single-subject"
