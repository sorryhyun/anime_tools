"""Caption-source helpers used by the tagger CLI build/eval modes.

Tag-taxonomy / caption-format constants (``SLOT_ORDER``, ``TAG_TYPE_NAMES``,
``RATINGS``, ``PEOPLE_COUNT_LABELS``) are the single source of truth for the
trainer's view of the corpus and live in
``anime_tools/tagger/tagger.py`` so the inference wrapper, training CLI,
and any downstream consumer all see the same definitions. The script-local
helpers below (caption-file discovery, count-tag detection, people-count
bucketing) consume those constants.
"""

from __future__ import annotations

from pathlib import Path

# Count-tag detection lives in the shared torch-free tag-shape module so the
# vocab build and caption-index builder can't drift. Re-exported here
# (``_COUNT_RE`` is also used by ``anime_tools.captions.group_router``); ``classify_people`` moved
# there too so the torch-free inference path (dbv4 backend without a sidecar)
# can bucket people-counts from count tags.
from anime_tools.captions.taxonomy import (
    _COUNT_RE,
    _LEADING_INT_RE,
    classify_people,
    is_count_tag,
)

__all__ = [
    "IMAGE_EXTS",
    "_COUNT_RE",
    "_LEADING_INT_RE",
    "classify_people",
    "find_image_for_caption",
    "is_count_tag",
]

# Image extensions next to each .txt caption; order is preference, first hit wins.
IMAGE_EXTS: tuple[str, ...] = (".webp", ".jpg", ".jpeg", ".png")


def find_image_for_caption(caption_path: Path) -> Path | None:
    """Return the sibling image file matching ``{stem}.<ext>``, or None."""
    for ext in IMAGE_EXTS:
        candidate = caption_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None
