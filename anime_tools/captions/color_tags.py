"""Colour tags of an Anima caption: what a colorize caption keeps.

A B&W page already carries composition, pose and layout; hue is what it cannot
carry, so a colorize caption keeps only colour tags (anima_lora's MOSS colorize
cache and ComfyUI-MOSS-Toolkit's caption helper build theirs from these).

* :func:`is_color_tag`: hair / eye / skin colour tags ("blue hair", "dark skin",
  "two-tone hair", "heterochromia"), any ``<color> <noun>`` tag ("yellow shirt",
  "white background"), palette descriptors ("monochrome", "pastel colors").
* :func:`filter_to_colors` reduces a caption to them, in order; none → ``""``.
* :func:`filter_to_colors_and_protected` puts the series tags (``copyright_tags``,
  from a caption index's ``groups.copyright``: :func:`load_copyright_tags`) and
  the comic / panel-format tags (:data:`COMIC_TAGS`) first.
* :func:`is_character_colour`: the colour tags one character's box carries
  (hair, eyes, clothing; not skin, background or sky).

A caption is read through :func:`~anime_tools.captions.position_clauses.parse_caption`:
the flat bag's tags, then each position clause's (the clause headers are not
tags, and a text clause's quoted lines are words in the picture, not colours).
The filters return a flat, comma-joined bag. Every tag comparison is keyed on
:func:`~anime_tools.captions.taxonomy.normalize_tag`, so ``blue_hair`` is
``blue hair``. Pure stdlib.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

from anime_tools._json import read_json
from anime_tools.captions.position_clauses import parse_caption
from anime_tools.captions.taxonomy import normalize_tag

# Single color words that head a "<color> <noun>" tag or end a "<x> hair/eyes".
# Deliberately permissive: a false keep ("blue rose") is still color info.
COLOR_WORDS: frozenset[str] = frozenset(
    {
        "aqua",
        "black",
        "blonde",
        "blue",
        "brown",
        "green",
        "grey",
        "gray",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "white",
        "yellow",
        "tan",
        "teal",
        "cyan",
        "magenta",
        "maroon",
        "navy",
        "olive",
        "violet",
        "beige",
        "gold",
        "golden",
        "cream",
        "crimson",
        "scarlet",
        "turquoise",
        "lavender",
        "peach",
        "azure",
        "ivory",
        "bronze",
    }
)

# Leading modifiers that precede a color word ("light blue hair", "dark green
# dress", "pale ...").
COLOR_MODIFIERS: frozenset[str] = frozenset(
    {"light", "dark", "pale", "deep", "bright", "pastel"}
)

# Multi-word / non-"<color> X" tags that are still pure color information and
# should be kept whole.
COLOR_PHRASES: frozenset[str] = frozenset(
    {
        "monochrome",
        "greyscale",
        "grayscale",
        "colorful",
        "muted color",
        "pastel colors",
        "limited palette",
        "spot color",
        "rainbow order",
        # hair/eye multi-color escapes (booru groups.yaml escapes)
        "two-tone hair",
        "multicolored hair",
        "gradient hair",
        "streaked hair",
        "split-color hair",
        "rainbow hair",
        "colored inner hair",
        "multicolored eyes",
        "heterochromia",
        "two-tone eyes",
        # skin tone
        "dark skin",
        "pale skin",
        "tan",
        "sun tan",
        "tanlines",
        "tan lines",
        "dark-skinned female",
        "dark-skinned male",
    }
)


# Comic / panel-format tags, kept as a leading token so a "comic" prompt steers
# toward panelled output. Fixed booru vocab (no caption-index lookup).
COMIC_TAGS: frozenset[str] = frozenset(
    {
        "comic",
        "manga",
        "doujinshi",
        "2koma",
        "3koma",
        "4koma",
        "5koma",
        "multiple 4koma",
    }
)


def is_comic_tag(tag: str) -> bool:
    """True if ``tag`` is a comic / panel-format tag worth keeping + binding."""
    return normalize_tag(tag) in COMIC_TAGS


def is_color_tag(tag: str) -> bool:
    """True if ``tag`` carries color/hue information worth keeping."""
    t = normalize_tag(tag)
    if not t:
        return False
    if t in COLOR_PHRASES:
        return True
    words = t.split()
    # "<...> hair/eyes/skin" with a color word anywhere → colored body feature.
    if words[-1] in ("hair", "eyes", "skin") and any(w in COLOR_WORDS for w in words):
        return True
    # Leading color word: "<color> <noun>" (clothing, object, background, …).
    if words[0] in COLOR_WORDS:
        return True
    # Modifier + color word: "light blue dress", "dark green skirt".
    return words[0] in COLOR_MODIFIERS and len(words) >= 2 and words[1] in COLOR_WORDS


def _caption_tags(caption: str) -> list[str]:
    """Every taxonomy tag of ``caption`` in reading order: the flat bag, then
    each position clause's tags. Clause headers and text clauses are not tags."""
    parsed = parse_caption(caption)
    return [
        *parsed.flat_tags,
        *(t for c in parsed.position_clauses for t in c.tags),
    ]


def filter_to_colors(caption: str) -> str:
    """Reduce an Anima caption to its color tags only.

    Returns a comma-joined string of the kept tags (original order and
    spelling, each once), or ``""`` when the caption has no color tags.
    """
    return filter_to_colors_and_protected(
        caption, keep_copyright=False, keep_comic=False
    )


# Copyright-group entries that name "no series" rather than a franchise.
# ``original`` is the most common copyright tag in the corpus (~a quarter of
# captions); kept, it would be a near-constant leading token with no series
# identity, so it is stripped from the copyright vocab.
_NON_SERIES_COPYRIGHT: frozenset[str] = frozenset({"original", "original character"})


@functools.lru_cache(maxsize=4)
def load_copyright_tags(index_path: str | Path) -> frozenset[str]:
    """Lowercased set of every copyright tag name from a caption index.

    Reads ``groups.copyright`` (a ``{copyright_name: [image_ids]}`` map) from the
    ``caption_index.json`` at ``index_path``; empty set if it is missing. Cached.
    Excludes ``_NON_SERIES_COPYRIGHT`` (e.g. ``original``).
    """
    p = Path(index_path)
    if not p.exists():
        return frozenset()
    copyright_group = read_json(p).get("groups", {}).get("copyright", {})
    return frozenset(
        key
        for name in copyright_group
        if (key := normalize_tag(name)) and key not in _NON_SERIES_COPYRIGHT
    )


def is_copyright_tag(tag: str, copyright_tags: frozenset[str]) -> bool:
    """True if ``tag`` is a known copyright/series name; compared by
    :func:`normalize_tag`, the form :func:`load_copyright_tags` returns."""
    return normalize_tag(tag) in copyright_tags


def filter_to_colors_and_protected(
    caption: str,
    copyright_tags: frozenset[str] | None = None,
    *,
    keep_copyright: bool = True,
    keep_comic: bool = False,
) -> str:
    """Reduce a caption to its protected leading tags followed by its color tags.

    The protected prefix is, in order: copyright/series tags (``keep_copyright``)
    then comic/panel-format tags (``keep_comic``), e.g. "genshin impact, comic,
    pink hair, blue eyes". Each tag is emitted once: a tag that is both
    copyright/comic and color-shaped stays in the prefix. ``copyright_tags`` is
    the series vocabulary (:func:`load_copyright_tags`); without it no tag counts
    as a series.
    """
    if copyright_tags is None:
        copyright_tags = frozenset()
    tags = _caption_tags(caption)
    kept: list[str] = []
    seen: set[str] = set()

    def _take(pred: Callable[[str], bool]) -> None:
        for t in tags:
            key = normalize_tag(t)
            if key not in seen and pred(t):
                kept.append(t)
                seen.add(key)

    if keep_copyright:
        _take(lambda t: is_copyright_tag(t, copyright_tags))
    if keep_comic:
        _take(is_comic_tag)
    _take(is_color_tag)
    return ", ".join(kept)


def filter_to_colors_and_copyright(
    caption: str, copyright_tags: frozenset[str] | None = None
) -> str:
    """Reduce a caption to its copyright tags (first) followed by its color tags.

    Wrapper over :func:`filter_to_colors_and_protected` with only the copyright
    prefix enabled.
    """
    return filter_to_colors_and_protected(
        caption, copyright_tags, keep_copyright=True, keep_comic=False
    )


CHARACTER_COLOUR_SKIP: frozenset[str] = frozenset(
    {"skin", "background", "sky", "border", "theme"}
)


def is_character_colour(tag: str) -> bool:
    """A colour tag a character box carries: hair, eyes and clothing, not skin
    (``dark-skinned …`` too) or a blanked crop's background."""
    words = normalize_tag(tag).split()
    return (
        bool(words)
        and is_color_tag(tag)
        and words[-1] not in CHARACTER_COLOUR_SKIP
        and not any(w.endswith("skinned") for w in words)
    )
