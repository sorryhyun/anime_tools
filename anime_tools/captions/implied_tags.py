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
    compose_caption,
    parse_caption,
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


def _kept(
    tags: Iterable[str], pool: frozenset[str], shield: frozenset[str]
) -> tuple[list[str], list[str]]:
    keep: list[str] = []
    drop: list[str] = []
    for tag in tags:
        key = normalize_tag(tag)
        implied = key not in shield and any(implies(other, key) for other in pool)
        (drop if implied else keep).append(tag)
    return keep, drop


def drop_implied_tags(text: str, *, keep: Iterable[str] = ()) -> ImpliedDropResult:
    """Cut every tag another tag in scope implies; ``keep`` names tags never cut.
    A caption with nothing to cut comes back byte for byte."""
    shield = frozenset(normalize_tag(t) for t in keep)
    parsed = parse_caption(text.replace("\n", ","))
    scoped = [c for c in parsed.clauses if not c.is_text]
    everything = frozenset(
        normalize_tag(t)
        for t in (*parsed.flat_tags, *(t for c in scoped for t in c.tags))
    )
    flat, dropped = _kept(parsed.flat_tags, everything, shield)
    clauses: list[PositionClause] = []
    for clause in parsed.clauses:
        if clause.is_text:
            clauses.append(clause)
            continue
        own = frozenset(normalize_tag(t) for t in clause.tags)
        kept, drop = _kept(clause.tags, own, shield)
        dropped.extend(drop)
        clauses.append(
            PositionClause(
                position=clause.position, tags=tuple(kept), prefix=clause.prefix
            )
        )
    if not dropped:
        return ImpliedDropResult(text)
    return ImpliedDropResult(compose_caption(flat, clauses), tuple(dropped))
