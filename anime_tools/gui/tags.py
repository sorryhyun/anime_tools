"""The Danbooru tag KB behind the caption panel's click-a-tag panel.

Answers come from ``danbooru_tags_classified.csv``; without it this module answers
"not installed". The base CSV carries the taxonomy with Korean descriptions and the
optional ``.en.csv`` sibling carries only English ones, so English is merged *over*
the base — the other direction would drop the category and count of the ~half of
tags the wiki mirror misses. Loaded lazily, re-read when either file's mtime moves.
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
"""``(stamp, kb, source)``. The stamp covers *both* files (path + mtime, or ``None``
for an absent sibling), so a download of either is picked up on the next call."""


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

    Reads plain rows rather than a second KB, and touches only tags the base
    table already knows.
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
                pass  # a truncated build still leaves the base KB usable
        _CACHE = (stamp, kb, source)
        return kb, source


def describe(tag: str) -> dict[str, Any]:
    """One tag's KB entry, as the caption panel's JSON.

    ``installed`` false means the CSV is absent; ``known`` false means it is
    present and this tag is not a Danbooru tag.
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
        # False when the row answered under another spelling (underscored tag,
        # @-prefixed artist looked up bare).
        exact=normalize_tag(tag) == info.name,
    )
    return out
