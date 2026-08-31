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
from collections.abc import Iterable

# People-count tags — shared definition that classify_people and the vocab categorizer both key off. The caption-index builder additionally counts "no girls"/"no boys", but those sit after @artist so they never reach the pre-artist span; keeping them out here avoids mistyping them in the model vocab.
_COUNT_RE = re.compile(
    r"^(?:\d+\+?(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
)

_LEADING_INT_RE = re.compile(r"^(\d+)")

_SPACE_RE = re.compile(r"\s+")

# The exact / open-ended halves of :data:`_COUNT_RE`, kept apart because
# :func:`count_of` needs the number and the noun out of the match. ``6+girls``
# is a crowd tag, not the number six, so it reads as "unknown", not as 6.
_EXACT_COUNT_RE = re.compile(r"^(\d+)(girl|boy|other)s?$")
_OPEN_COUNT_RE = re.compile(r"^\d+\+(girl|boy|other)s?$")


def normalize_tag(tag: str) -> str:
    """A tag's canonical space form: trimmed, ``_``→`` ``, lowercased, collapsed.

    The one normalizer. ``speech_bubble`` and ``Speech  Bubble`` both become
    ``speech bubble``, so either danbooru convention keys the same entry — the
    Danbooru KB is built through this, the caption corrector writes through it,
    and every "does the caption already say this?" comparison in the grammar
    and in grouping goes through it. Three near-copies used to disagree on the
    underscore fold, which is exactly the pair a comparison must not miss.

    It is a *key* function that happens also to be the corrector's output form;
    callers that only compare must not write the result back.
    """
    return _SPACE_RE.sub(" ", tag.strip().replace("_", " ").lower())


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


def exact_count(tag: str, noun: str) -> int | None:
    """The number in an exact ``N<noun>s`` tag (``3girls`` → 3), else ``None``.

    Exact means countable: the open-ended ``6+girls`` and the ``multiple_*``
    implication are not exact counts and answer ``None`` here — see
    :func:`count_of`, which is this over a whole bag.
    """
    m = _EXACT_COUNT_RE.match(normalize_tag(tag))
    return int(m.group(1)) if m is not None and m.group(2) == noun else None


def count_of(tags: Iterable[str], noun: str) -> int | None:
    """How many ``girl``/``boy``/``other`` subjects ``tags`` claims, if it says.

    * an integer — the largest exact ``N<noun>s`` count present (``0`` when the
      bag carries no count for this noun at all: the caption doesn't say)
    * ``None`` — "more than one, count unknown": ``multiple girls`` or an
      open-ended ``6+girls`` with no exact count to override it.

    An exact count always wins over an open-ended or ``multiple_*`` tag, which
    booru fires as an implication alongside it. Tags are read through
    :func:`normalize_tag`, so the ``multiple_girls`` / ``multiple girls``
    spelling disagreement resolves here instead of at each call site.
    """
    plural = f"multiple {noun}s"
    exact: list[int] = []
    unknown = False
    for raw in tags:
        tag = normalize_tag(raw)
        if (n := exact_count(tag, noun)) is not None:
            exact.append(n)
        elif tag == plural or ((m := _OPEN_COUNT_RE.match(tag)) and m.group(1) == noun):
            unknown = True
    if exact:
        return max(exact)
    return None if unknown else 0


# ``solo`` is a non-count membership tag — gelcrawl writes it alongside
# ``1girl``/``1boy`` when there is exactly one figure — so the single-subject
# predicate is "one of these fired, and no *other* count tag did". The
# single-count names also match :data:`_COUNT_RE` (``\d+girls?``), which is why
# they have to be excluded from the multi test rather than merely not counted.
SINGLE_COUNT_NAMES = frozenset({"solo", "1girl", "1boy", "1other"})


def is_solo_names(names: Iterable[str]) -> bool:
    """True when a tag-name set describes a single subject.

    The name-side twin of :func:`solo_multi_indices` (and of the trainer's
    ``GroupRouter.solo_mask``): what ``softmax_when_solo`` groups gate on.
    """
    names = set(names)
    has_multi = any(n not in SINGLE_COUNT_NAMES and is_count_tag(n) for n in names)
    return bool(names & SINGLE_COUNT_NAMES) and not has_multi


def solo_multi_indices(vocab_tags: Iterable[dict]) -> tuple[set[int], set[int]]:
    """``(single_count_indices, multi_count_indices)`` over ``vocab.json`` rows.

    The index-side twin of :func:`is_solo_names`, for consumers that hold a
    multi-hot tensor rather than names. Same precedence: a single-count name is
    never also counted as multi.
    """
    single: set[int] = set()
    multi: set[int] = set()
    for t in vocab_tags:
        name = t["name"]
        idx = int(t["index"])
        if name in SINGLE_COUNT_NAMES:
            single.add(idx)
        elif is_count_tag(name):
            multi.add(idx)
    return single, multi


def is_artist_tag(tag: str) -> bool:
    """True for Anima artist tags: a leading ``@`` immediately followed by a
    non-whitespace character (``@sincos``, ``@sumiyao (amam)``).

    The non-whitespace guard excludes booru emoticons like ``@ @`` (``@_@``
    after ``_``→`` `` normalization), which are general tags, not artists.
    """
    return len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace()


def strip_artist_prefix(tag: str) -> str:
    """Drop a leading ``@`` so the bare name can be looked up in a tag cache."""
    return tag.removeprefix("@")


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
