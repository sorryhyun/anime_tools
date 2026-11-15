"""Tags a more specific tag in the same caption already says.

``very long hair`` says ``long hair``, ``diagonal-striped neckerchief`` says
``neckerchief``, ``red bowtie`` says ``bow``. :func:`drop_implied_tags` cuts
the generic one. A bag tag goes when any tag in the caption implies it; a
clause tag only when a tag in its own clause does, since a specific tag on
another subject says nothing about this one. Text clauses pass through.

The rule is lexical: ``<modifier> X`` implies ``X``, unless the modifier changes
what ``X`` names (:data:`NON_IMPLYING_PREFIXES`: ``no pants`` is not ``pants``,
``spoken heart`` is not ``heart``, ``looking back`` is not ``back``).
:data:`EXTRA_IMPLIES` holds the pairs no suffix reaches, and
:data:`SUPERSEDED_BY` the tags another one names better: ``diagonal-striped bow``
beside ``diagonal-striped bowtie`` is the same knot, which the tagger reads as
a bowtie. Pure stdlib.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from anime_tools.captions.position_clauses import (
    PositionClause,
    parse_caption,
    partition_tags,
)
from anime_tools.captions.taxonomy import normalize_tag

NON_IMPLYING_PREFIXES = frozenset({"no", "spoken", "looking", "covered"})

# generic -> the specific tags (and their ``<modifier> X`` forms) that imply it
# without ending in it.
EXTRA_IMPLIES: dict[str, frozenset[str]] = {
    "bow": frozenset({"bowtie"}),
    "very long hair": frozenset({"absurdly long hair"}),
}


# head noun -> the head noun that replaces it: ``<M> bow`` goes when ``<M> bowtie``
# is in scope, with the same modifier ``<M>`` (empty included), so a ``hair bow``
# beside a bowtie stays.
SUPERSEDED_BY: dict[str, str] = {
    "bow": "bowtie",
    "neckerchief": "bowtie",
}


@dataclass(frozen=True)
class ImpliedDropResult:
    text: str
    dropped_tags: tuple[str, ...] = ()


def _superseded(specific: str, generic: str) -> bool:
    for head, better in SUPERSEDED_BY.items():
        if generic == head or generic.endswith(" " + head):
            modifier = generic[: len(generic) - len(head)]
            if specific == modifier + better:
                return True
    return False


def implies(specific: str, generic: str) -> bool:
    """Whether ``specific`` says everything ``generic`` does (normalized keys)."""
    if specific == generic:
        return False
    if _superseded(specific, generic):
        return True
    for target in (generic, *EXTRA_IMPLIES.get(generic, ())):
        if specific == target:
            return True
        if specific.endswith(" " + target):
            modifier = specific[: -len(target) - 1]
            if modifier.split()[0] not in NON_IMPLYING_PREFIXES:
                return True
    return False


def drop_implied_tags(text: str, *, keep: Iterable[str] = ()) -> ImpliedDropResult:
    """Cut every tag another tag in scope implies; ``keep`` names tags never cut
    (both keyed on :func:`normalize_tag`). A caption with nothing to cut comes
    back byte for byte. The cut is
    :func:`~anime_tools.captions.position_clauses.partition_tags`."""
    shield = frozenset(normalize_tag(t) for t in keep)
    parsed = parse_caption(text)
    everything = frozenset(
        normalize_tag(t)
        for t in (
            *parsed.flat_tags,
            *(t for c in parsed.position_clauses for t in c.tags),
        )
    )
    own: dict[PositionClause, frozenset[str]] = {}

    def implied(tag: str, clause: PositionClause | None) -> bool:
        # A bag tag goes for a tag anywhere; a clause tag only for one in its
        # own clause — a specific tag on another subject says nothing here.
        key = normalize_tag(tag)
        if key in shield:
            return False
        if clause is None:
            pool = everything
        elif (pool := own.get(clause)) is None:
            pool = own[clause] = frozenset(normalize_tag(t) for t in clause.tags)
        return any(implies(other, key) for other in pool)

    kept, dropped = partition_tags(parsed, implied)
    if not dropped:
        return ImpliedDropResult(text)
    return ImpliedDropResult(kept.render(), dropped)
