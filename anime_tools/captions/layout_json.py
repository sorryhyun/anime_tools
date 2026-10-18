"""Anima caption → MOSS-Image layout-JSON prompt.

MOSS-Image reads a layout prompt as plain text: a compact JSON object whose
``summary.caption`` describes the scene and whose ``items`` carry one element
each (``label`` / ``caption`` / ``text``). This maps an Anima caption onto that
shape without inventing anything — every tag lands in exactly one field::

    sensitive, 1girl, hatsune miku, vocaloid, @sincos, long hair.
    On the left, necktie. Japanese text reads as "満足でしょうか"

    {"summary":{"meta":["sensitive"],"characters":["hatsune miku"],
     "series":["vocaloid"],"artists":["@sincos"],"caption":"1girl, long hair"},
     "items":[{"label":"character","position":"left","caption":"necktie"},
              {"label":"text","caption":"Japanese text","text":"満足でしょうか",
               "bbox":[-420,-172,-287,-149]}],
     "bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}

Position clauses become ``character`` items that keep their relative position
word and never take a box; a clause's character names move to the item's
``characters``. Each quoted line of a text clause is its own ``text`` item, in
reading order. A text item takes a ``bbox`` when ``text_boxes`` (the OCR
sidecar's boxes, :func:`~anime_tools.captions.ocr_sidecar.text_boxes`) holds
its line, in MOSS's ``abs_centralized`` / ``yxyx`` form: pixels of the image
the boxes were read on, origin at its centre, ``[y0, x0, y1, x1]``, clamped to
the canvas. The top-level ``bbox_coord_space`` / ``bbox_format`` keys appear
only when some item has a box. Artists carry the Anima ``@`` (added to a bare KB artist name);
``@no-artist`` is an Anima sentinel and is left out. Danbooru meta tags that
describe the pixels (``cropped``, ``scan``, ``jpeg artifacts``, … —
``_VISUAL_META``) go to ``summary.caption``; the rest of the meta category stays
in ``summary.meta``. Empty fields are omitted, except ``summary.caption``.

The serialization is compact (``","`` / ``":"``, no ASCII escaping) like the
MOSS gallery prompts, and deterministic for a given caption, KB and boxes —
training and inference must produce the same string. Pure stdlib.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from anime_tools.captions.correction import TagKnowledgeBase, tag_kind
from anime_tools.captions.position_clauses import (
    TEXT_PREFIXES,
    PositionClause,
    as_parsed,
)
from anime_tools.captions.taxonomy import (
    canonical_rating,
    normalize_tag,
    strip_artist_prefix,
)

# correct_tags buckets that read as image metadata, in Anima emit order.
_META_KINDS = ("quality", "meta", "year", "safety")

# Danbooru meta-category tags that describe what is in the image, not the post.
_VISUAL_META = frozenset(
    {
        "cropped",
        "scan",
        "magazine scan",
        "tall image",
        "wide image",
        "jpeg artifacts",
        "lowres",
        "upscaled",
        "rotated",
        "color halftone",
        "colorized",
        "decensored",
        "hard-translated",
        "nude filter",
        "photo (medium)",
        "game cg",
    }
)

_TEXT_CAPTIONS = {
    TEXT_PREFIXES[0]: "Japanese text",
    TEXT_PREFIXES[1]: "Japanese sound effect",
}


def _summary(tags, kb: TagKnowledgeBase) -> dict:
    meta: dict[str, list[str]] = {k: [] for k in _META_KINDS}
    characters: list[str] = []
    series: list[str] = []
    artists: list[str] = []
    counts: list[str] = []
    general: list[str] = []
    for tag in tags:
        kind = tag_kind(tag, kb)
        if kind == "meta" and normalize_tag(tag) in _VISUAL_META:
            general.append(tag)
        elif kind in meta:
            meta[kind].append(canonical_rating(tag) or tag)
        elif kind == "character":
            characters.append(tag)
        elif kind == "copyright":
            series.append(tag)
        elif kind == "artist":
            if tag != "@no-artist":
                artists.append("@" + strip_artist_prefix(tag))
        elif kind == "count":
            counts.append(tag)
        else:
            general.append(tag)
    out: dict = {}
    if metas := [t for k in _META_KINDS for t in meta[k]]:
        out["meta"] = list(dict.fromkeys(metas))
    if characters:
        out["characters"] = characters
    if series:
        out["series"] = series
    if artists:
        out["artists"] = artists
    # Counts lead the description, as they lead an Anima bag.
    out["caption"] = ", ".join((*counts, *general))
    return out


def _character_item(clause: PositionClause, kb: TagKnowledgeBase) -> dict:
    characters = [t for t in clause.tags if tag_kind(t, kb) == "character"]
    rest = [t for t in clause.tags if tag_kind(t, kb) != "character"]
    item: dict = {"label": "character", "position": clause.position}
    if characters:
        item["characters"] = characters
    if rest:
        item["caption"] = ", ".join(rest)
    return item


def _unquote(line: str) -> str:
    # ``text_clause`` wraps each line in ASCII quotes; a 「…」 inside is content.
    if len(line) >= 2 and line[0] == '"' == line[-1]:
        return line[1:-1]
    return line


def centered_box(box: Sequence[float], size: tuple[int, int]) -> list[int]:
    """``(x0, y0, x1, y1)`` pixels → MOSS ``abs_centralized`` ``yxyx``: origin
    at the centre of a ``size = (W, H)`` canvas, clamped to it."""
    w, h = size
    x0, y0, x1, y1 = (float(v) for v in box)

    def at(v: float, side: int) -> int:
        return round(min(max(v, 0.0), float(side)) - side / 2)

    return [at(y0, h), at(x0, w), at(y1, h), at(x1, w)]


def _text_items(
    clause: PositionClause,
    boxes: Mapping[str, Sequence[float]],
    size: tuple[int, int] | None,
) -> list[dict]:
    caption = _TEXT_CAPTIONS.get(clause.prefix, "text")
    items = []
    for line in clause.tags:
        text = _unquote(line)
        item: dict = {"label": "text", "caption": caption, "text": text}
        box = boxes.get(text.strip())
        if box is not None and size is not None:
            item["bbox"] = centered_box(box, size)
        items.append(item)
    return items


def caption_to_layout(
    caption: str,
    kb: TagKnowledgeBase,
    *,
    text_boxes: Mapping[str, Sequence[float]] | None = None,
    size: tuple[int, int] | None = None,
) -> dict:
    """The layout object for one Anima caption (see the module docstring).

    ``text_boxes`` maps a text line to its ``(x0, y0, x1, y1)`` box in the
    pixels of a ``size = (W, H)`` image; pass both or neither."""
    if (text_boxes is None) != (size is None):
        raise ValueError("text_boxes and size go together")
    parsed = as_parsed(caption.replace("\n", ","))
    items: list[dict] = []
    for clause in parsed.position_clauses:
        items.append(_character_item(clause, kb))
    for clause in parsed.text_clauses:
        items.extend(_text_items(clause, text_boxes or {}, size))
    layout: dict = {"summary": _summary(parsed.flat_tags, kb)}
    if items:
        layout["items"] = items
    if any("bbox" in item for item in items):
        layout["bbox_coord_space"] = "abs_centralized"
        layout["bbox_format"] = "yxyx"
    return layout


def layout_json(
    caption: str,
    kb: TagKnowledgeBase,
    *,
    text_boxes: Mapping[str, Sequence[float]] | None = None,
    size: tuple[int, int] | None = None,
) -> str:
    """:func:`caption_to_layout`, serialized the way MOSS prompts are."""
    return json.dumps(
        caption_to_layout(caption, kb, text_boxes=text_boxes, size=size),
        ensure_ascii=False,
        separators=(",", ":"),
    )
