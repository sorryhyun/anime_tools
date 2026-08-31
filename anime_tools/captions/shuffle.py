"""Caption shuffle grammar — ``@artist`` prefix boundary, ``@no-artist``
sentinel, section-aware tag shuffle. Torch-free.
"""

from __future__ import annotations

import random

from anime_tools.captions.position_clauses import is_clause_header
from anime_tools.captions.taxonomy import is_artist_tag

# Sentinel users can drop into captions that lack a real artist tag, so the
# shuffle/drop boundary keeps working. Callers that feed a shuffle result to a
# tokenizer must strip it themselves — it is kept inside the shuffle so the
# boundary index stays consistent with the input.
NO_ARTIST_SENTINEL = "@no-artist"


def find_anima_prefix_end(tags: list[str]) -> int:
    """Index one past the trailing artist-handle in the leading run.

    Returns 0 if no artist tag is present anywhere (the case the ``@no-artist``
    sentinel exists to fix). Multi-artist captions protect the full handle run,
    not just the first handle.
    """
    split_idx = 0
    saw_artist = False
    for idx, tag in enumerate(tags):
        if is_artist_tag(tag):
            split_idx = idx + 1
            saw_artist = True
        elif saw_artist:
            break
    return split_idx


def strip_no_artist_sentinel(tags: list[str]) -> list[str]:
    """Drop every occurrence of :data:`NO_ARTIST_SENTINEL` from ``tags``."""
    return [t for t in tags if t != NO_ARTIST_SENTINEL]


def anima_smart_shuffle_caption(flex_tokens: list[str]) -> list[str]:
    """Shuffle caption tags with awareness of @artist prefix and 'on the ...' sections.

    - Tags up to and including the trailing artist tag of the leading run are
      kept in order (see :func:`find_anima_prefix_end`). Multi-artist captions
      and the ``@no-artist`` sentinel both preserve the full handle run.
    - Remaining tags are split into sections by 'on the ...' / 'in the ...'
      delimiters; tags within each section are shuffled independently.
    - The ``@no-artist`` sentinel is preserved in the output so the boundary
      index stays usable; callers that feed the result to a tokenizer must
      call :func:`strip_no_artist_sentinel` before joining.
    """
    split_idx = find_anima_prefix_end(flex_tokens)

    prefix = flex_tokens[:split_idx]
    suffix = flex_tokens[split_idx:]

    # Sections delimited by clause headers, via the grammar's own predicate so a
    # new header form is one edit, not three.
    sections: list[list[str]] = [[]]
    for tag in suffix:
        if is_clause_header(tag):
            sections.append([tag])
        else:
            sections[-1].append(tag)

    result = list(prefix)
    for section in sections:
        if not section:
            continue
        if is_clause_header(section[0]):
            header, body = [section[0]], section[1:]
        else:
            header, body = [], section
        shuffled = body.copy()
        random.shuffle(shuffled)
        result.extend(header + shuffled)
    return result


__all__ = [
    "NO_ARTIST_SENTINEL",
    "anima_smart_shuffle_caption",
    "find_anima_prefix_end",
    "strip_no_artist_sentinel",
]
