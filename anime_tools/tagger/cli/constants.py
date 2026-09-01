"""Caption-source helpers used by the tagger CLI build/eval modes.

The tag-taxonomy / caption-format constants themselves (``SLOT_ORDER``,
``TAG_TYPE_NAMES``, ``RATINGS``, ``PEOPLE_COUNT_LABELS``) live in
``anime_tools/tagger/tagger.py``.
"""

from __future__ import annotations

from pathlib import Path

from anime_tools._walk import IMAGE_EXTENSIONS

# Count-tag detection and people-count bucketing; re-exported for the CLI modes.
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

# Image extensions next to each .txt caption; order is preference, first hit
# wins. The preferred four lead, then the rest of the curation walker's list, so
# a caption whose only sibling is a `.bmp` is still trainable.
_PREFERRED = (".webp", ".jpg", ".jpeg", ".png")
IMAGE_EXTS: tuple[str, ...] = _PREFERRED + tuple(
    e for e in dict.fromkeys(x.lower() for x in IMAGE_EXTENSIONS) if e not in _PREFERRED
)


def find_image_for_caption(caption_path: Path) -> Path | None:
    """Return the sibling image file matching ``{stem}.<ext>``, or None."""
    for ext in IMAGE_EXTS:
        candidate = caption_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None
