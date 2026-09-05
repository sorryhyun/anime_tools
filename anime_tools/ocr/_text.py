"""What a recognized line *says*, once both sessions have stopped: the content
filters and the CJK join.

The models answer one box at a time, but a manga balloon is many boxes and a
watermark is one, so the difference between a sidecar and a wall of noise is made
here rather than in either graph. Five passes, in this order:

1. :func:`join_cjk` — the columns (or rows) of one balloon back into one record.
   It runs **first**, because a two-glyph column is only short until it is joined.
2. :func:`normalize_line` — the glyphs PP-OCRv6 has no Japanese for: a vertical
   ``ー`` comes back as ``1`` or ``|``, a horizontal one as ``-``; after kana they
   are put back.
3. :func:`is_latin_only` — the ``skip_en`` drop: page numbers, URLs, romaji sfx.
4. :func:`is_tally` — the ``正`` tally marks of body writing, which are a count
   and not a word.
5. :func:`char_count` — the ``min_chars`` floor: a stray glyph is a misread
   screentone far more often than it is a word.

:func:`reading_order` then settles the sidecar's sequence: a page set in columns
reads right to left, one set in rows top to bottom.

Stdlib only — no numpy, no cv2, no weights — so the content half of OCR is
testable on hand-built boxes.
"""

from __future__ import annotations

from collections.abc import Sequence

from anime_tools.captions.ocr_sidecar import OcrLine

VERTICAL_RATIO = 1.5
"""Taller than this many times its width and a box is a *column*, not a row —
the same threshold :func:`~anime_tools.ocr._onnx.crop_quad` uprights on, so a box
is joined along the axis it was recognized along."""

GAP_RATIO = 0.6
"""How far apart two columns (or two rows) may sit and still be one balloon,
measured in the thickness of the thicker one. Japanese sets its columns tighter
than it sets one balloon from the next, which is the whole signal here."""

OVERLAP_RATIO = 0.4
"""How much of the shorter box's length must face the other's for the two to be
side by side rather than merely near. Balloon text is ragged — the last column is
routinely half the first's — so this is a low bar."""

JOIN_SEP = " "
"""What :func:`_merge` puts between the columns (or rows) of one block: a
column boundary is kept as a space so a list stays a list (2026-09-05)."""

SIZE_RATIO = 2.5
"""How far the two thicknesses may differ. A sfx glyph twice the height of the
dialogue beside it is a different piece of text, however close it lands."""


def char_count(text: str) -> int:
    """Characters that carry something — whitespace does not count.

    The ``min_chars`` floor is about how much was *read*, and the recognizer
    emits a space wherever it is unsure between two glyphs.
    """
    return sum(1 for ch in text if not ch.isspace())


def is_latin_only(text: str) -> bool:
    """Whether the line is pure ASCII: English, a URL, a page number, romaji.

    Deliberately the *negative* test rather than a script whitelist. PP-OCRv6 is
    one model over fifty languages, so an inclusion list would silently drop
    whatever it was not written for; anything outside ASCII — kana, kanji,
    hangul, fullwidth Latin — is kept.
    """
    return text.isascii()


CHOON = "ー"
"""The prolonged sound mark. PP-OCRv6's vocabulary is Chinese-first: set
vertically the mark is a bare stroke it reads as ``1`` / ``|`` / ``l``, set
horizontally a dash — so ``おちんぼの時間だぞ1`` and ``でるッリ一チ`` are one
misread each, not two words."""

_CHOON_VERTICAL = frozenset("1|lI１｜丨")
"""What a vertical ``ー`` comes back as."""

_CHOON_HORIZONTAL = frozenset("-—–ｰ")
"""What a horizontal ``ー`` comes back as (``ｰ`` is its own halfwidth form)."""

_NI_AS_EQUALS = frozenset("=＝")
"""What ``ニ`` comes back as between katakana — ``メ=ュー`` — two strokes the
vocabulary has as an equals sign."""


def is_kana(ch: str) -> bool:
    """Hiragana or katakana, ``ー`` included — the scripts a ``ー`` follows."""
    return "\u3041" <= ch <= "\u3096" or "\u30a1" <= ch <= "\u30fc"


def _is_katakana(ch: str) -> bool:
    return "\u30a1" <= ch <= "\u30fc"


_KANA_COUNTERS = frozenset("かつヶケ")
"""Counters written in kana — ``1か月`` / ``あと1つ`` — after which a digit is a
digit even though kana precedes it."""


def _counts_something(nxt: str) -> bool:
    """Whether the glyph after a would-be ``ー`` makes it a numeral instead:
    a kanji (``もう1回``), another digit or Latin, or a kana counter."""
    if not nxt:
        return False
    if "\u4e00" <= nxt <= "\u9fff" or nxt.isascii() and nxt.isalnum():
        return True
    return "\uff10" <= nxt <= "\uff19" or nxt in _KANA_COUNTERS


def normalize_ja(text: str, *, vertical: bool) -> str:
    """Put back the ``ー`` the recognizer spelled as a digit or a dash.

    Only after kana, and only when nothing counted follows: ``1`` after a kanji
    is a number, ``-`` after Latin a hyphen, and ``もう1回`` / ``1か月`` keep
    their digit (:func:`_counts_something`). A horizontal ``一`` (the kanji *one*) is a ``ー`` only *between* katakana —
    ``リ一チ`` — since ``一杯`` after a katakana word is a real ``一``. Vertical
    text never confuses the two (the strokes cross), so ``一`` is left alone
    there. The mark chains: ``ーー`` is written as such, and a fixed ``ー`` is
    kana for the next glyph. Likewise ``=`` between katakana is ``ニ``
    (``メ=ュー``), in either orientation.
    """
    out: list[str] = []
    for i, ch in enumerate(text):
        prev = out[-1] if out else ""
        if prev and is_kana(prev):
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if (
                ch in _CHOON_HORIZONTAL or (vertical and ch in _CHOON_VERTICAL)
            ) and not _counts_something(nxt):
                out.append(CHOON)
                continue
            between_katakana = _is_katakana(prev) and bool(nxt) and _is_katakana(nxt)
            if not vertical and ch == "一" and between_katakana:
                out.append(CHOON)
                continue
            if ch in _NI_AS_EQUALS and between_katakana:
                out.append("ニ")
                continue
        out.append(ch)
    return "".join(out)


def normalize_line(line: OcrLine) -> OcrLine:
    """:func:`normalize_ja` applied along the axis the box was read along."""
    text = normalize_ja(line.text, vertical=_vertical(line))
    if text == line.text:
        return line
    return OcrLine(seq=line.seq, box=line.box, score=line.score, text=text)


TALLY_STROKES = frozenset("正一丁下卜TF ")
"""``正`` and the partial forms PP-OCRv6 spells its strokes as."""


def is_tally(text: str) -> bool:
    """Whether the line is tally marks — ``正T正正`` — rather than words.

    Body-writing counts are set in ``正`` strokes; the recognizer reads a
    complete one as the character and a partial one as whatever Latin or kanji
    the strokes resemble. At least one ``正`` and nothing that is not a stroke.
    """
    stripped = text.strip()
    return (
        bool(stripped)
        and "正" in stripped
        and all(ch in TALLY_STROKES for ch in stripped)
    )


def keep_line(text: str, *, min_chars: int = 0, skip_en: bool = False) -> bool:
    """Whether a joined line survives the content filters."""
    if skip_en and is_latin_only(text):
        return False
    if is_tally(text):
        return False
    return char_count(text) >= min_chars


def _vertical(line: OcrLine) -> bool:
    return line.height >= VERTICAL_RATIO * max(line.width, 1)


def _adjacent(a: OcrLine, b: OcrLine) -> bool:
    """Whether two boxes are neighbouring columns (or rows) of one block.

    Same orientation, facing each other along their length, close along their
    thickness, and of a comparable thickness. Mixed orientations never join: a
    column beside a row is a sfx over dialogue, not its continuation.
    """
    vertical = _vertical(a)
    if vertical != _vertical(b):
        return False
    if vertical:
        span = (a.box[1], a.box[3], b.box[1], b.box[3])
        near = (a.box[0], a.box[2], b.box[0], b.box[2])
        thick = (a.width, b.width)
        length = (a.height, b.height)
    else:
        span = (a.box[0], a.box[2], b.box[0], b.box[2])
        near = (a.box[1], a.box[3], b.box[1], b.box[3])
        thick = (a.height, b.height)
        length = (a.width, b.width)

    overlap = min(span[1], span[3]) - max(span[0], span[2])
    if overlap < OVERLAP_RATIO * max(1, min(length)):
        return False
    gap = max(near[0], near[2]) - min(near[1], near[3])
    if gap > GAP_RATIO * max(1, max(thick)):
        return False
    lo, hi = min(thick), max(thick)
    return hi <= SIZE_RATIO * max(1, lo)


def _clusters(lines: Sequence[OcrLine]) -> list[list[int]]:
    """Indices grouped by adjacency — a plain union-find over every pair.

    At most :attr:`~anime_tools.ocr._onnx.OcrEngine.max_boxes` boxes reach here,
    so the quadratic pass costs nothing worth avoiding.
    """
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if _adjacent(lines[i], lines[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _merge(lines: Sequence[OcrLine]) -> OcrLine:
    """One block's boxes as a single record, read in its own direction.

    Columns read right to left, rows top to bottom; the parts are joined with
    :data:`JOIN_SEP` (one space). Japanese sets no separator between the columns
    of a sentence, but a joined block is as often a list — a profile card's
    ``名前 / 身長 / 好きなもの`` rows — and glued (``椎名真昼ちゃん身長：156cm``)
    the boundary is lost for good, while a space inside a sentence costs a
    reader nothing. The box is the block's bound and the score the
    per-character mean of the parts, so a long confident column is not outvoted
    by the two glyphs beside it.
    """
    if len(lines) == 1:
        return lines[0]
    if _vertical(lines[0]):
        ordered = sorted(lines, key=lambda ln: -ln.box[0])
    else:
        ordered = sorted(lines, key=lambda ln: ln.box[1])
    weight = sum(char_count(ln.text) for ln in ordered)
    score = (
        sum(ln.score * char_count(ln.text) for ln in ordered) / weight
        if weight
        else min(ln.score for ln in ordered)
    )
    return OcrLine(
        seq=0,
        box=(
            min(ln.box[0] for ln in ordered),
            min(ln.box[1] for ln in ordered),
            max(ln.box[2] for ln in ordered),
            max(ln.box[3] for ln in ordered),
        ),
        score=score,
        text=JOIN_SEP.join(ln.text.strip() for ln in ordered),
    )


def join_cjk(lines: Sequence[OcrLine]) -> list[OcrLine]:
    """Merge each block of neighbouring CJK boxes into one line.

    Only lines carrying a non-ASCII character are candidates
    (:func:`is_latin_only`): English is set with spaces and wraps for width, so
    two English boxes stacked in a balloon are two lines and stay two. Everything
    else passes through untouched and in place — the caller re-sorts into reading
    order afterwards, since a merged block sits where neither part did.
    """
    joinable = [i for i, ln in enumerate(lines) if not is_latin_only(ln.text)]
    if len(joinable) < 2:
        return list(lines)

    out: list[tuple[int, OcrLine]] = [
        (i, ln) for i, ln in enumerate(lines) if is_latin_only(ln.text)
    ]
    candidates = [lines[i] for i in joinable]
    for group in _clusters(candidates):
        out.append((joinable[group[0]], _merge([candidates[k] for k in group])))
    return [ln for _, ln in sorted(out, key=lambda pair: pair[0])]


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
