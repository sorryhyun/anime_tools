"""The caption ladder: one image's captions, read, expanded and written.

:data:`CAPTION_LADDER` is the order the sidebar's dots and the panel's badges
are drawn in; :func:`caption_versions` expands a sidecar rung into a badge per
caption it holds; :func:`write_caption` is the one writer, editable rungs only.
Re-exported by :mod:`anime_tools.gui.dataset`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anime_tools.captions.history import (
    history_sidecar_path,
    push_history,
    read_history,
)
from anime_tools.captions.position_clauses import parse_caption, tag_spans
from anime_tools.captions.variants import read_variants_sidecar, variants_sidecar_path
from anime_tools.gui._dataset_roots import DatasetError, Roots, _rel_key, rel_to_home
from anime_tools.stages._walk_captions import master_caption


@dataclass(frozen=True)
class Rung:
    """One rung of the caption ladder: a caption file and what may be done to it.

    ``root`` names the :class:`Roots` field the file lives in. ``expand`` marks a
    rung that is a *sidecar* of many captions rather than one: ``"variants"`` holds
    the generated ``v0``/``v1``/``r1`` samples, ``"history"`` the versions a write
    superseded. Each is one file, expanded into a badge apiece by
    :func:`caption_versions`.
    """

    kind: str
    root: str
    editable: bool
    expand: str | None = None
    """``"variants"`` / ``"history"`` — the sidecar beside the caption, not the
    caption."""
    of: str = ""
    """Which rung a history sidecar records the past of; its badges wear that
    rung's name (``revised@2``)."""
    overlay: str = ""
    """A second root that *shadows* ``root``: read from here when it holds the
    file, and write here always. How the master rung stays editable without
    writing ``src`` — see :data:`CAPTION_LADDER`."""
    dot: bool = True
    """Whether this rung is one of the sidebar row's caption dots. The strip
    answers "what does this image say?" at a glance, one dot per text; the
    history rung is not a text but the record of a change, so it is a badge in
    the panel and no dot out here. Costs the listing one stat per row less."""


CAPTION_LADDER: tuple[Rung, ...] = (
    Rung("master", "src", editable=True, overlay="master"),
    Rung("revised", "dst", editable=True),
    Rung("variants", "dst", editable=False, expand="variants"),
    Rung("history", "dst", editable=False, expand="history", of="revised", dot=False),
)
"""The captions of one image: the texts first, in the order a caption becomes
them, and the record of what changed after them.

``master`` → ``revised`` is the pipeline, and ``variants`` is the sidecar
generated beside the second. ``history`` comes **last** even though it holds the
oldest text of all: it is not a caption you read or write but the record of a
change, and the panel draws it as a diff rather than as a text. Between the two
writable rungs it read as a third one and confused them. The run bar has no
Apply gate, so the text a run replaces survives as a badge here. The sidebar
strip, the panel's badges and :func:`write_caption`'s guard all read this tuple.

The master rung carries an ``overlay``: ``src`` is the INPUT tree and
**read-only for the tools** (``anime_tools.workspace``), so editing a master
caption writes ``workspace/master/<rel>.txt`` and reads it back in preference to
the hand-written original, which stays untouched as its own undo. Export's
``master`` row publishes the overlay back to ``src`` — that row is the one
consumer, and its ``is_file()`` guard is exactly "this master was edited".
"""

CAPTION_KINDS = tuple(r.kind for r in CAPTION_LADDER if r.editable)
"""The writable rungs — what :func:`write_caption` accepts."""

_SIDECAR_PATH = {
    "variants": variants_sidecar_path,
    "history": history_sidecar_path,
}
"""``Rung.expand`` → the sidecar path beside a caption."""

HISTORY_OF = {r.of: r.kind for r in CAPTION_LADDER if r.expand == "history"}
"""Editable rung → the history rung recording it, for writers that push a version
before overwriting one. Empty for a rung the ladder gives no history."""


def ladder_schema() -> list[dict[str, Any]]:
    """:data:`CAPTION_LADDER` as the listing hands it to the browser.

    The sidebar's dot strip is drawn from this, so it cannot come apart from
    :func:`_row`'s ``captions`` map — both are ``Rung.dot``, which is why the
    history rung is in neither. The ``root`` stays server-side.
    """
    return [{"kind": r.kind, "editable": r.editable} for r in CAPTION_LADDER if r.dot]


def caption_paths(roots: Roots, rel: Path) -> dict[str, Path]:
    """Every rung's file to READ, keyed by rung kind; :data:`CAPTION_LADDER`
    names the root each lives in.

    A rung with an ``overlay`` resolves to it when it holds the file and to
    ``root`` otherwise — ``_walk_captions.master_caption``, the master half of
    the ladder every caption stage and Export read, so an edited master shadows
    the hand-written one here exactly as it does there. Costs the master rung a
    second stat per row on a whole-dataset listing. :func:`caption_write_path`
    is the write side.
    """
    txt = rel.with_suffix(".txt")
    out: dict[str, Path] = {}
    for r in CAPTION_LADDER:
        root = getattr(roots, r.root)
        p = (
            master_caption(getattr(roots, r.overlay), root, txt)
            if r.overlay
            else root / txt
        )
        out[r.kind] = _SIDECAR_PATH[r.expand](p) if r.expand else p
    return out


def caption_write_path(roots: Roots, rel: Path, kind: str) -> Path:
    """Where writing rung ``kind`` lands — its ``overlay`` when it has one.

    Never ``roots.src``: that is the input tree, and Export is the only thing
    that writes it (``anime_tools.workspace``).
    """
    rung = next(r for r in CAPTION_LADDER if r.kind == kind)
    return getattr(roots, rung.overlay or rung.root) / rel.with_suffix(".txt")


def parsed_caption(text: str) -> dict[str, Any]:
    """The caption grammar as JSON: flat bag + position clauses + where each tag
    sits in the string, never a ``split(",")`` on the client.

    ``spans`` lets the caption editor draw a box around every tag inside the
    textarea by slicing the text it holds at offsets the parse handed it.
    """
    parsed = parse_caption(text)
    return {
        "flat_tags": list(parsed.flat_tags),
        "spans": [
            {"start": s.start, "end": s.end, "kind": s.kind, "clause": s.clause}
            for s in tag_spans(text)
        ],
        "clauses": [
            {
                "header": c.header,
                "prefix": c.prefix,
                "position": c.position,
                "tags": list(c.tags),
            }
            for c in parsed.clauses
        ],
    }


def _version_entry(
    kind: str,
    *,
    path: str,
    exists: bool,
    editable: bool,
    mtime: float | None,
    text: str,
    rung: str = "",
    note: str = "",
) -> dict[str, Any]:
    """One caption version as the panel gets it — the badge's wire shape.

    ``rung`` is the :data:`CAPTION_LADDER` row this entry came out of, which is not
    always its ``kind``: an expanded sidecar entry is called ``v1`` or ``revised@2``.
    It defaults to ``kind``. ``note`` is the one extra line a badge can carry (a
    history entry's who and when).

    ``exists`` is told, not read off ``text``: an empty caption file is a caption
    that says nothing, a different answer from a rung nobody ever wrote, and only
    the second draws hollow.
    """
    return {
        "kind": kind,
        "rung": rung or kind,
        "note": note,
        "path": path,
        "exists": exists,
        "editable": editable,
        "mtime": mtime,
        "text": text,
        "parsed": parsed_caption(text) if exists else None,
    }


def _caption_entry(
    kind: str, p: Path, *, editable: bool, rung: str = "", note: str = ""
) -> dict[str, Any]:
    """One caption *file* as a version: read it, or say it is not there.

    An absent file is still a (hollow) entry, so the rung keeps its place on the
    badge row. ``editable`` is the rung's property, not the file's.
    """
    exists = p.is_file()
    return _version_entry(
        kind,
        path=rel_to_home(p),
        exists=exists,
        editable=editable,
        mtime=p.stat().st_mtime if exists else None,
        text=p.read_text(encoding="utf-8").strip() if exists else "",
        rung=rung,
        note=note,
    )


def _expanded(rung: Rung, sidecar: Path) -> list[tuple[str, str, str]]:
    """A sidecar rung's captions as ``(label, note, text)``, oldest first.

    A variant's label *is* its name (``v0``); a history entry wears the rung's
    name plus its sequence, with a note saying who replaced it and when.
    """
    if rung.expand == "history":
        return [(e.label(rung.of), e.note(), e.text) for e in read_history(sidecar)]
    return [(label, "", text) for label, text in read_variants_sidecar(sidecar)]


def caption_versions(roots: Roots, rel: Path) -> list[dict[str, Any]]:
    """Every caption this image has, oldest first — the panel's badge row.

    :data:`CAPTION_LADDER` in order, with a sidecar rung *expanded* into one entry
    per caption it holds (``v0``, ``v1``… for variants, ``revised@1``, ``revised@2``…
    for superseded versions). An absent sidecar still contributes one hollow rung.
    Every entry arrives parsed, since the browser may not split a caption.
    """
    caps = caption_paths(roots, rel)
    out: list[dict[str, Any]] = []
    for r in CAPTION_LADDER:
        p = caps[r.kind]
        rows = _expanded(r, p) if r.expand and p.is_file() else []
        if not rows:
            out.append(_caption_entry(r.kind, p, editable=r.editable))
            continue
        # The sidecar is one file however many captions it holds: stat'd and
        # named once, with each row's text already in hand.
        mtime = p.stat().st_mtime
        home = rel_to_home(p)
        out.extend(
            _version_entry(
                label,
                path=home,
                exists=True,
                editable=False,
                mtime=mtime,
                text=text,
                rung=r.kind,
                note=note,
            )
            for label, note, text in rows
        )
    return out


def write_caption(roots: Roots, rel_str: str, kind: str, text: str) -> dict[str, Any]:
    """Write one caption file. The ladder's editable rungs only.

    A caption is a single line by contract, so newlines fold to spaces. An empty
    body is refused rather than treated as a delete. A hand edit pushes history
    like a stage run does, for the rungs :data:`HISTORY_OF` gives one — the
    master rung has none because it does not need one: the edit goes to the
    ``workspace/master/`` overlay and the hand-written original stays put.

    The answer carries the whole ladder back under ``versions``, not just the
    entry written: the push above *creates a rung* (``revised@1``), and a caller
    that folded one entry into the row it already had could never show it.
    """
    if kind not in CAPTION_KINDS:
        raise DatasetError(f"not an editable caption: {kind!r}")
    rel = _rel_key(rel_str)
    if not (roots.src / rel).is_file():
        raise DatasetError(f"not in the dataset: {rel.as_posix()}")
    body = " ".join(str(text).split())
    if not body:
        raise DatasetError("refusing to write an empty caption")

    caps = caption_paths(roots, rel)
    p = caption_write_path(roots, rel, kind)
    if kind in HISTORY_OF and p.is_file():
        push_history(p, p.read_text(encoding="utf-8").strip(), by="edit")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")

    entry = _caption_entry(kind, p, editable=True)
    # The sidecar was generated from the previous revised text, so its v0 no
    # longer matches what the TE step would encode.
    entry["variants_stale"] = kind == "revised" and caps["variants"].is_file()
    entry["versions"] = caption_versions(roots, rel)
    return entry
