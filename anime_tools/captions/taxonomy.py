"""Low-level danbooru tag-*shape* primitives — the single source of truth.

These recognize the *form* of a tag (artist ``@``-prefix, count tag, raw rating
literal) without any vocab or model. They are shared by every consumer that
types tags so the two categorization paths can't silently drift:

* the Anima Tagger vocab build — ``anime_tools/tagger/cli/vocab.py::categorize``
  (image→tag model's view of the corpus), and
* the dataset caption index — ``anime_tools/captions/index.py``
  (method-agnostic typed-tag index for identity pairing / analytics).

Pure stdlib by design: importing this must NOT pull in torch. The richer,
*content*-aware heuristics (vocab-membership classification, danbooru
``name (series)`` paren recovery, positional bare-name recovery) stay with the
caption-index builder — they exist to compensate for the tagger's frozen vocab
and have no model-side counterpart.
"""

from __future__ import annotations

import re
from typing import Iterable

# People-count tags — shared definition that classify_people and the vocab categorizer both key off. The caption-index builder additionally counts "no girls"/"no boys", but those sit after @artist so they never reach the pre-artist span; keeping them out here avoids mistyping them in the model vocab.
_COUNT_RE = re.compile(
    r"^(?:\d+\+?(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
)

_LEADING_INT_RE = re.compile(r"^(\d+)")


def is_count_tag(tag: str) -> bool:
    """True for people-count tags (``1girl``, ``2girls``, ``multiple_boys``…)."""
    return bool(_COUNT_RE.match(tag))


def classify_people(tags: Iterable[str]) -> int:
    """Derive the 8-class :data:`PEOPLE_COUNT_LABELS` index for a parsed-tag list.

    Bucketing rules:

    * ``no_people`` (0) — no count tag at all
    * ``1girl`` (1), ``2girls`` (3), ``1boy`` (6) — exact-girls-no-boy /
      exact-boys-no-girl combos
    * ``1girl_1boy`` (2), ``2girls_1boy`` (4), ``2boys_1girl`` (5) —
      the three explicit mixed combos
    * ``multi`` (7) — anything else with a count tag: ``3+girls``,
      ``3+boys``, ``2girls+2+boys``, ``Nothers``, or a ``multiple_*`` tag
      with no explicit numeric companion. ``others`` count tags ride into
      ``multi`` since the head is girls/boys-shaped.

    Booru auto-fires ``multiple_girls`` / ``multiple_boys`` whenever the
    count is ≥2, not just ≥3 — so it cannot be treated as a ≥3 signal on
    its own. We defer to the explicit numeric count tag when one is
    present; ``multiple_*`` only contributes as a floor of 2 when no
    numeric tag for that gender was seen.

    Tag order in ``tags`` doesn't matter — counts are reduced first.
    """
    girls = boys = 0
    saw_multi_g = saw_multi_b = False
    saw_other = False
    for t in tags:
        if not is_count_tag(t):
            continue
        if t.startswith("multiple"):
            if "girl" in t:
                saw_multi_g = True
            elif "boy" in t:
                saw_multi_b = True
            elif "other" in t:
                saw_other = True
            continue
        m = _LEADING_INT_RE.match(t)
        if m is None:  # e.g. malformed; defensive
            continue
        n = int(m.group(1))
        if "girl" in t:
            girls = max(girls, n)
        elif "boy" in t:
            boys = max(boys, n)
        # "others" counts go to the "multi" indicator (no 7-bucket fit).
        elif "other" in t:
            saw_other = True
    # ``multiple_*`` only kicks in when the numeric tag is missing; treat as ≥2
    # not ≥3, since that's what the booru auto-tag means.
    if saw_multi_g and girls == 0:
        girls = 2
    if saw_multi_b and boys == 0:
        boys = 2
    if saw_other or girls >= 3 or boys >= 3 or (boys >= 2 and girls >= 2):
        return 7  # multi: 3+girls / 3+boys / 2g+2b+ / lonely multiple_* / Nothers
    if girls == 0 and boys == 0:
        return 0  # no_people (only when no count tag fired)
    if girls == 1 and boys == 0:
        return 1  # 1girl
    if girls == 1 and boys == 1:
        return 2  # 1girl_1boy
    if girls == 2 and boys == 0:
        return 3  # 2girls
    if girls == 2 and boys == 1:
        return 4  # 2girls_1boy
    if girls == 1 and boys == 2:
        return 5  # 2boys_1girl
    if girls == 0 and boys == 1:
        return 6  # 1boy
    return 7  # fallback (e.g. 0g/2b without "others")


def is_artist_tag(tag: str) -> bool:
    """True for Anima artist tags: a leading ``@`` immediately followed by a
    non-whitespace character (``@sincos``, ``@sumiyao (amam)``).

    The non-whitespace guard excludes booru emoticons like ``@ @`` (``@_@``
    after ``_``→`` `` normalization), which are general tags, not artists.
    """
    return len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace()


def strip_artist_prefix(tag: str) -> str:
    """Drop a leading ``@`` so the bare name can be looked up in a tag cache."""
    return tag[1:] if tag.startswith("@") else tag


# Anima's 4-class rating vocabulary — the leading safety band of a caption
# (``safe, 1girl, …``), and the same set the tagger's rating head predicts
# (``anime_tools.tagger.tagger.RATINGS``, which fixes the class order).
CAPTION_RATINGS = frozenset({"safe", "sensitive", "nsfw", "explicit"})

# Danbooru's own rating literals, mapped onto the Anima band. Anima renames two
# of the four (``general``→``safe``, ``questionable``→``nsfw``) and keeps the
# other two verbatim, so raw booru captions — and corpora/vocab.json built
# before the rename — still read as ratings instead of falling through to the
# ``general`` *category*.
LEGACY_RATING_ALIASES = {"general": "safe", "questionable": "nsfw"}

# Every literal that reads as a rating: canonical band + accepted legacy spellings.
RATING_LITERALS = CAPTION_RATINGS | frozenset(LEGACY_RATING_ALIASES)


def is_rating_tag(tag: str) -> bool:
    """True for any accepted rating literal (canonical Anima or legacy booru)."""
    return tag in RATING_LITERALS


def canonical_rating(tag: str) -> str | None:
    """The canonical Anima rating for ``tag``, or None when it isn't a rating."""
    if tag in CAPTION_RATINGS:
        return tag
    return LEGACY_RATING_ALIASES.get(tag)
