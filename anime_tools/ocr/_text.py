"""What a read line *says*, once the reader has answered: the content floors and
the page's reading order.

The reader answers one box at a time, but a watermark and a page number are boxes
too, so the difference between a sidecar and a wall of noise is made here rather
than in either model. Three tests, in :func:`keep_line`'s order:

1. :func:`is_latin_only` — the ``skip_en`` drop: page numbers, URLs, romaji sfx.
2. :func:`is_tally` — the ``正`` tally marks of body writing, which are a count
   and not a word.
3. :func:`char_count` — the ``min_chars`` floor: a stray glyph is a misread
   screentone far more often than it is a word.

:func:`reading_order` then settles the sidecar's sequence: a page set in columns
reads right to left, one set in rows top to bottom.

The CJK column join and the ``ー`` normalisation that used to sit in front of
these were retired with the CTC line recognizer (2026-09-07): the AnimeText
detector answers a balloon as a block, and the VL reader reads ``ー`` natively.

Stdlib only — no numpy, no cv2, no weights — so the content half of OCR is
testable on hand-built boxes.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from anime_tools.captions.ocr_sidecar import OcrLine

VERTICAL_RATIO = 1.5
"""Taller than this many times its width and a box is a *column*, not a row —
what :func:`reading_order` reads a page's direction off."""


def char_count(text: str) -> int:
    """Characters that carry something — whitespace does not count.

    The ``min_chars`` floor is about how much was *read*, and the reader keeps a
    space at a column boundary.
    """
    return sum(1 for ch in text if not ch.isspace())


def is_latin_only(text: str) -> bool:
    """Whether the line is pure ASCII: English, a URL, a page number, romaji.

    Deliberately the *negative* test rather than a script whitelist: an inclusion
    list would silently drop whatever it was not written for; anything outside
    ASCII — kana, kanji, hangul, fullwidth Latin — is kept.
    """
    return text.isascii()


TALLY_STROKES = frozenset("正一丁下卜TF ")
"""``正`` and the partial forms a reader spells its strokes as."""


def is_tally(text: str) -> bool:
    """Whether the line is tally marks — ``正T正正`` — rather than words.

    Body-writing counts are set in ``正`` strokes; a reader answers a complete
    one as the character and a partial one as whatever Latin or kanji the
    strokes resemble. At least one ``正`` and nothing that is not a stroke.
    """
    stripped = text.strip()
    return (
        bool(stripped)
        and "正" in stripped
        and all(ch in TALLY_STROKES for ch in stripped)
    )


def drop_symbols(text: str) -> str:
    """``text`` without its pictographs — ``♡`` ``★`` ``♪``, emoji, the rest of
    Unicode's *other symbol* class (``So``) — whitespace re-collapsed.

    Punctuation stays: ``…``, ``!?``, ``~`` and ``ー`` are how a line is said;
    a heart is decoration on it. A read of hearts alone comes back empty.
    """
    kept = "".join(ch for ch in text if unicodedata.category(ch) != "So")
    return " ".join(kept.split())


def keep_line(text: str, *, min_chars: int = 0, skip_en: bool = False) -> bool:
    """Whether a read line survives the content floors."""
    if skip_en and is_latin_only(text):
        return False
    if is_tally(text):
        return False
    return char_count(text) >= min_chars


def _vertical(line: OcrLine) -> bool:
    return line.height >= VERTICAL_RATIO * max(line.width, 1)


def reading_order(lines: Sequence[OcrLine]) -> list[OcrLine]:
    """Sort into the order the page is read.

    A page set mostly in columns reads **right to left**, then down within a
    column band; one set in rows reads top to bottom, then left to right within
    a row band. The band is half the median thickness (width of a column, height
    of a row), so two balloons side by side on one row read across rather than
    down, and two columns of one balloon are never swapped by a few pixels of
    skew. Mixed pages follow the majority; a horizontal sfx on a column page
    takes its place by its right edge like everything else.
    """
    if not lines:
        return []
    columns = sum(1 for ln in lines if _vertical(ln))
    if columns * 2 >= len(lines):
        widths = sorted(max(1, ln.width) for ln in lines)
        band = max(1, widths[len(widths) // 2] // 2)
        return sorted(lines, key=lambda ln: (-(ln.box[2] // band), ln.box[1]))
    heights = sorted(max(1, ln.height) for ln in lines)
    band = max(1, heights[len(heights) // 2] // 2)
    return sorted(lines, key=lambda ln: (ln.box[1] // band, ln.box[0]))
