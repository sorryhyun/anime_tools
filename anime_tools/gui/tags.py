"""The Danbooru tag KB behind the caption panel's click-a-tag panel.

One question, asked one tag at a time: *what is this tag?* The answer comes out
of ``danbooru_tags_classified.csv`` — the same table
:mod:`anime_tools.captions.correction` types tags against when it corrects a
caption, so the panel explains a tag with the very rows that decide its bucket.
It is the ``danbooru_tags`` row of :mod:`anime_tools.downloads`; without it this
module answers "not installed" and the UI points at Settings › Models rather
than at nothing.

Two files, one view. The base CSV carries the tag taxonomy (name, kind,
category path, post count) with **Korean** descriptions; the optional
``.en.csv`` sibling (``danbooru_tags_en``) carries English ones for the same
names and nothing else. So the base file is the KB and the English descriptions
are merged over it — never the other way round, or a tag missing from the wiki
mirror (half of them) would lose its category and count as well.

Torch-free like the rest of ``anime_tools.gui``: this is stdlib csv over a
16 MB table, loaded lazily on the first click and re-read only when the file's
mtime moves.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any

from anime_tools._env import curation_home
from anime_tools.captions.correction import (
    TAG_CSV_EN_NAME,
    TagKnowledgeBase,
    find_tag_csv,
    load_tag_knowledge_base,
    normalize_tag,
    tag_key,
)

_LOCK = threading.Lock()
_CACHE: tuple[tuple, TagKnowledgeBase, str | None] | None = None
"""``(stamp, kb, source)`` — the one parsed copy. The stamp is the identity of
*both* files (path + mtime, or ``None`` for the sibling that isn't there), so a
download of either — the base table, or the English descriptions built over it
— is picked up on the next click instead of at the next restart."""


def _stamp(path: Path | None) -> tuple[str, float] | None:
    if path is None:
        return None
    try:
        return (str(path), path.stat().st_mtime)
    except OSError:
        return None


def base_csv() -> Path | None:
    """The base table, or ``None`` when the KB has not been downloaded."""
    return find_tag_csv(curation_home())


def english_csv() -> Path | None:
    """The English-description sibling, if it was built."""
    base = base_csv()
    if base is None:
        return None
    p = base.with_name(TAG_CSV_EN_NAME)
    return p if p.exists() else None


def _merge_english(kb: TagKnowledgeBase, path: Path) -> None:
    """Overwrite descriptions with the English ones, in place.

    A second :func:`load_tag_knowledge_base` would parse — and hold — a whole
    second KB for one column, so this reads the file as plain rows and touches
    only the tags the base table already knows.
    """
    import dataclasses

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        cols = {name: i for i, name in enumerate(header)}
        i_name = cols.get("name", 0)
        i_desc = cols.get("description", 3)
        width = max(i_name, i_desc) + 1
        for row in reader:
            if len(row) < width:
                continue
            desc = row[i_desc].strip()
            if not desc:  # the wiki mirror covers ~half the table
                continue
            key = tag_key(row[i_name])
            info = kb.tags.get(key)
            if info is not None:
                kb.tags[key] = dataclasses.replace(info, description=desc)


def load() -> tuple[TagKnowledgeBase | None, str | None]:
    """The merged KB and the name of the file its descriptions came from."""
    global _CACHE
    path = base_csv()
    if path is None:
        return None, None
    en = english_csv()
    stamp = (_stamp(path), _stamp(en))
    if stamp[0] is None:
        return None, None
    with _LOCK:
        cached = _CACHE
        if cached is not None and cached[0] == stamp:
            return cached[1], cached[2]
        try:
            kb = load_tag_knowledge_base(path)
        except (OSError, ValueError):
            return None, None
        source = path.name
        if en is not None:
            try:
                _merge_english(kb, en)
                source = en.name
            except (OSError, ValueError):
                pass  # a truncated build is not a reason to lose the base KB
        _CACHE = (stamp, kb, source)
        return kb, source


def describe(tag: str) -> dict[str, Any]:
    """One tag's KB entry, as the caption panel's JSON.

    ``installed`` false means the CSV is not here at all (offer the download);
    ``known`` false means it is, and this tag simply is not a Danbooru tag —
    an Anima quality tag, an ``@artist`` the KB never heard of, a typo.
    """
    kb, source = load()
    out: dict[str, Any] = {
        "tag": tag,
        "installed": kb is not None,
        "known": False,
        "source": source,
        "download_id": "danbooru_tags",
    }
    if kb is None:
        return out
    info = kb.describe(tag)
    if info is None:
        return out
    out.update(
        known=True,
        name=info.name,
        kind=info.kind,
        category_path=info.category_path,
        description=info.description,
        post_count=info.post_count,
        # False when the row answered under another spelling -- an underscored
        # tag, or an @-prefixed artist looked up bare.
        exact=normalize_tag(tag) == info.name,
    )
    return out
