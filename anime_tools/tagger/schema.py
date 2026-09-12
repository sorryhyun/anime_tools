"""What a tag *means* to a checkpoint — the vocabulary schema, torch-free.

Every value here is baked into a checkpoint's ``vocab.json`` at build time and
read back at inference, so changing one invalidates existing checkpoints. That
is also why it is a leaf: the vocab build (``tagger/cli/vocab.py``) and the
ComfyUI node need these four tuples and would otherwise import the model — and
with it torch, timm and the backbone — to read them.

:mod:`anime_tools.tagger.dbv4_meta` is the other torch-free leaf, holding where
a checkpoint's *files* live; this one holds what is in them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "GIRLS_COUNT_RE",
    "PEOPLE_COUNT_LABELS",
    "RATINGS",
    "SLOT_ORDER",
    "TAG_TYPE_NAMES",
    "TagEntry",
    "dedupe_count_tags",
    "fix_artist_category",
    "underscore_to_space",
]

# Canonical caption-format slot order (matches Anima training captions).
SLOT_ORDER: tuple[str, ...] = (
    "rating",
    "count",
    "character",
    "copyright",
    "artist",
    "general",
)

# Booru tag-type integer → category name. Written into vocab.json and read back
# at inference, so changes here invalidate existing checkpoints.
TAG_TYPE_NAMES: dict[int, str] = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "metadata",
    6: "deprecated",
}

# Canonical class-index order, least -> most restrictive. Do not reorder without
# rebuilding vocab.json/dataset.json.
RATINGS: tuple[str, ...] = ("safe", "sensitive", "nsfw", "explicit")

# 8-class people-count bucket from parsed count tags (``classify_people``).
# Order is the canonical class index — do not reorder without rebuilding vocab.
PEOPLE_COUNT_LABELS: tuple[str, ...] = (
    "no_people",  # 0 — no count tag at all
    "1girl",  # 1 — 1girl, no boy
    "1girl_1boy",  # 2 — exactly one of each
    "2girls",  # 3 — 2girls, no boy
    "2girls_1boy",  # 4 — 2girls + 1boy
    "2boys_1girl",  # 5 — 2boys + 1girl  (mirror of 2girls_1boy)
    "1boy",  # 6 — 1boy, no girl (solo male)
    "multi",  # 7 — 3+girls / 3+boys / 2g-2b+ / multiple_* / Nothers
)


@dataclass
class TagEntry:
    """One vocab row as the model holds it."""

    name: str
    index: int
    category: str
    median_pos: float


# Digit-prefixed girls counts ("1girl"…"6+girls").
GIRLS_COUNT_RE = re.compile(r"^(\d+)\+?girls?$")

# Exact people-count families ("3girls" / "2boys" / "1other", open "6+girls"
# included). "multiple_girls"/"multiple_boys" ride alongside a digit count, so
# neither this nor GIRLS_COUNT_RE matches them.
_EXACT_COUNT_RES = tuple(
    re.compile(rf"^\d+\+?{noun}s?$") for noun in ("girl", "boy", "other")
)


def dedupe_count_tags(kept: dict[str, float]) -> None:
    """Drop all but the highest-scoring exact count per family, in place.

    The sigmoid head has no mutual exclusion, so a near-threshold image can
    clear both ``3girls`` and ``4girls``, inflating the girls-count character
    cap and tripping the position-clause count-mismatch gate.
    """
    for cre in _EXACT_COUNT_RES:
        hits = sorted((n for n in kept if cre.match(n)), key=lambda n: -kept[n])
        for name in hits[1:]:
            kept.pop(name)


def underscore_to_space(s: str) -> str:
    """Apply at emit time, not vocab-build, so tag indexing stays stable."""
    return s.replace("_", " ")


def fix_artist_category(category: str, name: str) -> str:
    """Retype mis-categorized "artist" entries shipped in older vocab.json.

    Those builds typed any ``@``-prefixed tag as ``artist``, sweeping up booru
    emoticons like ``@_@``; the rule needs ``@`` followed by non-whitespace.
    """
    if category != "artist":
        return category
    if len(name) >= 2 and name[0] == "@" and not name[1].isspace():
        return "artist"
    return "general"
