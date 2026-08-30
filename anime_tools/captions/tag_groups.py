"""Anima caption tag-groups — structural typing on top of the flat tag vocab.

Companion to :mod:`tag_rules`. Where ``tag_rules.yaml`` enforces *which*
tags survive caption normalization, ``tag_groups.yaml`` enforces *how* the
surviving tags relate to each other — which sets are mutually exclusive
on a single subject (eye color, hair color), which are always exclusive
(rating), and which are flat multi-label but worth grouping for
introspection (top-garment families).

The trainer consumes this to route per-group losses (softmax-within-group
when the image is single-subject and no escape-tag fires; BCE otherwise).
Inference can use the same groups to produce a structured prediction
("eye_color: blue eyes (0.91)"). At training-time the YAML is loaded once,
intersected with the kept vocab, and the resolved index sets are written
into ``vocab.json`` so the trainer doesn't re-touch the YAML each step.

YAML schema
-----------

::

    version: 1

    eye_color:
      mode: softmax_when_solo            # | softmax | multilabel
      description: "Primary eye color"   # optional
      escape: [heterochromia, ...]       # optional — disables softmax routing
      tags:                              # canonical space form (matches vocab)
        - blue eyes
        - red eyes
        - ...

Mode semantics
~~~~~~~~~~~~~~

* ``softmax_when_solo`` — at training time, K-way CE on the group's logits
  when (a) the image is single-subject (``solo``/``1girl``/``1boy`` and no
  ``multiple_*``/``2+girls``/``2+boys``) **and** (b) no tag in ``escape``
  fires; falls back to per-tag BCE otherwise. The "solo + no escape" check
  is the trainer's job — the loader just exposes the group structure.
* ``softmax`` — always K-way CE. Used for genuinely exclusive groups like
  rating (which already has its own dedicated head).
* ``multilabel`` — sigmoid/BCE per tag. Listed only for documentation /
  introspection (so a UI can show "what kind of top is in this image").

Validation
~~~~~~~~~~

* Each tag name appears in at most one group.
* Tag names use the canonical *space* form (matches parsed-caption tags
  and ``vocab.json[tags][i][name]``).
* Tags listed in YAML that aren't in the kept vocab (after ``min_freq``
  cut) are silently dropped at :func:`resolve_groups` time — the YAML is
  intended to be stable across min_freq changes, not re-curated each time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

import yaml

# Allowed values for a group's ``mode`` field — loader rejects typos at parse time.
GROUP_MODES: FrozenSet[str] = frozenset(
    {
        "softmax_when_solo",
        "softmax",
        "multilabel",
    }
)

# ``sentinel: true`` on a softmax group adds a synthetic "none of this group"
# class: the vocab build appends a tag slot named ``sentinel_tag_name(group)``,
# CE supervises it on applicable samples with no member label (so the gate
# flips from "exactly one member present" to "at most one"), and decode emits
# nothing when it wins the argmax. The angle-bracket name can never collide
# with a caption tag (canonical tags are bare space-form words).
_SENTINEL_FMT = "<none:{group}>"

# ``tags: ["$category:artist"]`` expands at vocab-build time to every kept
# vocab tag of that category — used for groups whose membership is the whole
# category (artist) and would otherwise drift as the vocab is rebuilt.
_CATEGORY_MEMBER_PREFIX = "$category:"


def sentinel_tag_name(group_name: str) -> str:
    """Synthetic vocab-tag name for a group's "none of these" class."""
    return _SENTINEL_FMT.format(group=group_name)


def is_sentinel_name(tag_name: str) -> bool:
    """True for names produced by :func:`sentinel_tag_name` — never emitted."""
    return tag_name.startswith("<none:") and tag_name.endswith(">")


@dataclass(frozen=True)
class TagGroup:
    """One typed group of tag names (resolved indices come later)."""

    name: str
    mode: str
    description: str
    escape: Tuple[str, ...]
    tags: Tuple[str, ...]
    sentinel: bool = False


@dataclass(frozen=True)
class TagGroups:
    """All groups + a tag → group reverse map.

    The reverse map is built once and used at training time to mask each
    group's tags out of the residual BCE head (so a tag is supervised by
    *exactly one* loss term, never both).
    """

    version: int
    groups: Tuple[TagGroup, ...]
    tag_to_group: Mapping[str, str]

    def by_name(self, name: str) -> Optional[TagGroup]:
        for g in self.groups:
            if g.name == name:
                return g
        return None

    def to_dict(self) -> dict:
        """Round-trippable dict for snapshotting into the checkpoint dir."""
        out: dict = {"version": self.version}
        for g in self.groups:
            body: dict = {"mode": g.mode, "tags": list(g.tags)}
            if g.description:
                body["description"] = g.description
            if g.escape:
                body["escape"] = list(g.escape)
            if g.sentinel:
                body["sentinel"] = True
            out[g.name] = body
        return out


def load_groups(path: str | Path) -> TagGroups:
    """Load a ``tag_groups.yaml`` into a :class:`TagGroups`.

    Validates: known modes, no overlapping tag names across groups. Empty
    ``tags:`` lists are allowed (group exists but has no current members)
    so the YAML can be checked in before the corpus catches up.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    version = int(raw.pop("version", 1))

    groups: List[TagGroup] = []
    tag_to_group: Dict[str, str] = {}

    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(
                f"group {name!r}: expected a mapping with 'mode' / 'tags', "
                f"got {type(body).__name__}"
            )
        mode = str(body.get("mode", "")).strip()
        if mode not in GROUP_MODES:
            raise ValueError(
                f"group {name!r}: mode={mode!r} not in {sorted(GROUP_MODES)}"
            )
        description = str(body.get("description", "") or "")
        escape = tuple(str(t) for t in (body.get("escape") or []))
        tags = tuple(str(t) for t in (body.get("tags") or []))
        sentinel = bool(body.get("sentinel", False))
        if sentinel and mode == "multilabel":
            raise ValueError(
                f"group {name!r}: sentinel=true only makes sense on softmax modes"
            )

        for t in tags:
            existing = tag_to_group.get(t)
            if existing is not None:
                raise ValueError(
                    f"tag {t!r} listed under both {existing!r} and {name!r}"
                )
            tag_to_group[t] = name

        groups.append(
            TagGroup(
                name=name,
                mode=mode,
                description=description,
                escape=escape,
                tags=tags,
                sentinel=sentinel,
            )
        )

    return TagGroups(version=version, groups=tuple(groups), tag_to_group=tag_to_group)


def from_dict(d: dict) -> TagGroups:
    """Inverse of :meth:`TagGroups.to_dict` — load a snapshot from JSON/YAML dict."""
    version = int(d.get("version", 1))
    groups: List[TagGroup] = []
    tag_to_group: Dict[str, str] = {}
    for name, body in d.items():
        if name == "version":
            continue
        if not isinstance(body, dict):
            continue
        mode = str(body.get("mode", "")).strip()
        if mode not in GROUP_MODES:
            raise ValueError(
                f"group {name!r}: mode={mode!r} not in {sorted(GROUP_MODES)}"
            )
        description = str(body.get("description", "") or "")
        escape = tuple(str(t) for t in (body.get("escape") or []))
        tags = tuple(str(t) for t in (body.get("tags") or []))
        sentinel = bool(body.get("sentinel", False))
        for t in tags:
            existing = tag_to_group.get(t)
            if existing is not None:
                raise ValueError(
                    f"tag {t!r} listed under both {existing!r} and {name!r}"
                )
            tag_to_group[t] = name
        groups.append(
            TagGroup(
                name=name,
                mode=mode,
                description=description,
                escape=escape,
                tags=tags,
                sentinel=sentinel,
            )
        )
    return TagGroups(version=version, groups=tuple(groups), tag_to_group=tag_to_group)


@dataclass(frozen=True)
class ResolvedGroup:
    """A :class:`TagGroup` projected onto a built vocab's tag indices.

    ``tag_indices`` and ``escape_indices`` are sorted and disjoint with
    every other resolved group's ``tag_indices``. Tags / escape tags
    listed in the YAML but absent from the vocab (e.g. fell below
    ``min_freq``) are silently dropped.
    """

    name: str
    mode: str
    description: str
    tag_indices: Tuple[int, ...]
    escape_indices: Tuple[int, ...]
    # Names kept for snapshot/debug; dropped names are omitted.
    tag_names: Tuple[str, ...]
    escape_names: Tuple[str, ...]
    # Vocab index of the group's synthetic "none of these" class, when the
    # group declares ``sentinel: true`` AND the vocab carries the slot. It is
    # folded into ``tag_indices`` (always last — the synthetic slot is
    # appended after every real tag, so it sorts highest) and duplicated here
    # so consumers don't index-guess.
    sentinel_index: Optional[int] = None


def resolve_groups(
    groups: TagGroups,
    vocab_tag_to_idx: Mapping[str, int],
) -> Tuple[Tuple[ResolvedGroup, ...], Dict[str, str]]:
    """Project ``groups`` onto a built vocab's ``tag_to_idx`` map.

    Returns ``(resolved_groups, dropped)`` where ``dropped`` maps each
    YAML-listed tag/escape name that didn't survive the vocab cut to a
    short reason (``"not_in_vocab"``). The dropped map is informational —
    the trainer doesn't care about absent tags, but the build step logs it
    so the YAML curator can spot drift between corpus and vocab.
    """
    resolved: List[ResolvedGroup] = []
    dropped: Dict[str, str] = {}
    for g in groups.groups:
        kept_tags: List[Tuple[int, str]] = []
        for t in g.tags:
            idx = vocab_tag_to_idx.get(t)
            if idx is None:
                dropped[t] = "not_in_vocab"
                continue
            kept_tags.append((idx, t))
        kept_escape: List[Tuple[int, str]] = []
        for t in g.escape:
            idx = vocab_tag_to_idx.get(t)
            if idx is None:
                dropped[t] = "not_in_vocab"
                continue
            kept_escape.append((idx, t))
        kept_tags.sort()
        kept_escape.sort()
        sentinel_index: Optional[int] = None
        if g.sentinel and g.mode in ("softmax", "softmax_when_solo"):
            s_name = sentinel_tag_name(g.name)
            sentinel_index = vocab_tag_to_idx.get(s_name)
            if sentinel_index is None:
                # Vocab was built without the slot (pre-sentinel build) —
                # degrade to legacy exactly-one behavior rather than error.
                dropped[s_name] = "sentinel_slot_not_in_vocab"
            else:
                kept_tags.append((sentinel_index, s_name))
                kept_tags.sort()
        resolved.append(
            ResolvedGroup(
                name=g.name,
                mode=g.mode,
                description=g.description,
                tag_indices=tuple(i for i, _ in kept_tags),
                tag_names=tuple(n for _, n in kept_tags),
                escape_indices=tuple(i for i, _ in kept_escape),
                escape_names=tuple(n for _, n in kept_escape),
                sentinel_index=sentinel_index,
            )
        )
    return tuple(resolved), dropped


def resolved_to_dict(resolved: Tuple[ResolvedGroup, ...]) -> List[dict]:
    """Round-trippable list-of-dicts for embedding into ``vocab.json``."""
    return [
        {
            "name": g.name,
            "mode": g.mode,
            "description": g.description,
            "tag_indices": list(g.tag_indices),
            "tag_names": list(g.tag_names),
            "escape_indices": list(g.escape_indices),
            "escape_names": list(g.escape_names),
            "sentinel_index": g.sentinel_index,
        }
        for g in resolved
    ]


def expand_category_members(
    groups: TagGroups,
    name_to_category: Mapping[str, str],
) -> TagGroups:
    """Expand ``$category:<cat>`` member entries against a built vocab.

    Returns a new :class:`TagGroups` whose ``tags`` tuples carry the concrete
    vocab tag names — run this before :func:`resolve_groups` AND before
    snapshotting to the checkpoint dir, so the shipped ``groups.yaml`` is
    self-contained (the inference wrapper re-resolves names, never markers).
    Expanded names claimed by another group raise, same as the loader.
    """
    out_groups: List[TagGroup] = []
    tag_to_group: Dict[str, str] = {}
    for g in groups.groups:
        expanded: List[str] = []
        for t in g.tags:
            if t.startswith(_CATEGORY_MEMBER_PREFIX):
                cat = t[len(_CATEGORY_MEMBER_PREFIX) :]
                expanded.extend(
                    sorted(n for n, c in name_to_category.items() if c == cat)
                )
            else:
                expanded.append(t)
        for t in expanded:
            existing = tag_to_group.get(t)
            if existing is not None:
                raise ValueError(
                    f"tag {t!r} listed under both {existing!r} and {g.name!r} "
                    f"after $category expansion"
                )
            tag_to_group[t] = g.name
        out_groups.append(
            TagGroup(
                name=g.name,
                mode=g.mode,
                description=g.description,
                escape=g.escape,
                tags=tuple(expanded),
                sentinel=g.sentinel,
            )
        )
    return TagGroups(
        version=groups.version, groups=tuple(out_groups), tag_to_group=tag_to_group
    )
