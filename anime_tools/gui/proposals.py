"""A stage report read as per-image caption proposals — and put back again.

The dry pass the GUI's **Run** button starts already writes down everything a
diff needs: per image, the caption path it would write, the text that was on
disk when it ran, and the text it proposes. This module turns that report into
rows keyed by *dataset rel* (what the sidebar selects), so the caption panel can
show the proposal next to the caption it would replace, and **Apply** is the
plain write of what you just looked at.

:func:`undo` is the same reading in reverse: an ``--apply`` run's report records
the before-text of every caption it wrote, so restoring them needs no snapshot
directory — only the guard that the file still holds what the run put there.

Torch-free, like the rest of :mod:`anime_tools.gui`.

The report shapes below are a copy of each stage CLI's ``ReplaySpec`` rather
than an import: those live next to ``build_tag_fn`` / the SAM3 loaders, and
importing one would pull torch into the server process (``test_boundary`` pins
that it stays out). ``tests/test_gui_proposals.py`` compares the two copies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from anime_tools._json import read_json
from anime_tools.gui import dataset as D

# Statuses a replayed row carries (``stages.replay.ReplayRow``), as opposed to
# the source stage's own vocabulary. Both mean "this row is a real proposal".
_REPLAY_OK = ("written", "would-write")


@dataclass(frozen=True)
class Shape:
    """How to read one stage's report, and which caption it writes.

    Mirrors ``anime_tools.stages.replay.ReplaySpec`` for the same stage.
    ``root`` is the tree ``caption_path`` is relative to — ``src`` for the
    stages that write the caption master, ``dst`` for the clause rewrite, which
    only ever touches the derived caption.
    """

    rows_key: str
    before: str
    after: str
    root: str
    ok_status: str | None = None
    newline: bool = False
    drop_variants: bool = False


SHAPES: dict[str, Shape] = {
    "autotag": Shape("rows", "existing", "proposed", "src", ok_status="ok"),
    "position": Shape(
        "images",
        "original",
        "proposed",
        "dst",
        ok_status="proposed",
        drop_variants=True,
    ),
    # The audit gates on verdict/confidence rather than a row status, so there
    # is no ``ok_status`` to match: a row is a proposal when it proposes text.
    "audit": Shape("images", "caption", "proposed", "src", newline=True),
}

CAPTION_KIND: dict[str, str] = {"src": "master", "dst": "derived"}
"""Which of the two editable captions a stage's ``root`` names."""


class ProposalError(ValueError):
    """The report cannot be read as proposals — the server turns this into a 400."""


@dataclass(frozen=True)
class Proposal:
    """One image's pending change, as the caption panel shows it."""

    rel: str
    image: str
    kind: str
    path: str
    before: str
    after: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **vars(self),
            # Parsed here and never in the browser: the caption grammar has one
            # implementation, and a diff that split on commas would disagree
            # with it about every position clause.
            "before_parsed": D.parsed_caption(self.before) if self.before else None,
            "after_parsed": D.parsed_caption(self.after) if self.after else None,
        }


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProposalError(f"report not found: {path}")
    try:
        report = read_json(path)
    except (OSError, ValueError) as exc:
        raise ProposalError(f"report is not readable JSON: {path} ({exc})") from exc
    if not isinstance(report, dict):
        raise ProposalError(f"report is not a stage report object: {path}")
    return report


def _rows(report: Mapping[str, Any], shape: Shape) -> list[Mapping[str, Any]]:
    rows = report.get(shape.rows_key)
    if not isinstance(rows, list):
        raise ProposalError(f"report has no {shape.rows_key!r} list")
    return [r for r in rows if isinstance(r, Mapping)]


def _texts(row: Mapping[str, Any], shape: Shape) -> tuple[str, str]:
    """``(before, after)``, from either report dialect.

    A stage's own report names them per :class:`Shape`; the report a *replay*
    writes (``stages.replay.ReplayRow``) always calls them ``before``/``after``,
    and Apply is a replay, so both shapes reach here.
    """
    before, after = row.get(shape.before), row.get(shape.after)
    if before is None and after is None:
        before, after = row.get("before"), row.get("after")
    return str(before or "").strip(), str(after or "").strip()


def _proposes(row: Mapping[str, Any], shape: Shape) -> bool:
    status = row.get("status")
    if status is None:
        return True
    if status in _REPLAY_OK:
        return True
    return shape.ok_status is None or status == shape.ok_status


def _walk(
    report: Mapping[str, Any], shape: Shape
) -> Iterator[tuple[Mapping[str, Any], str, str]]:
    """Every row that carries a real change, with its texts."""
    for row in _rows(report, shape):
        before, after = _texts(row, shape)
        if not after or after == before or not _proposes(row, shape):
            continue
        yield row, before, after


def read(report_path: Path, roots: D.Roots, stage: str) -> dict[str, Proposal]:
    """The proposals in ``report_path``, keyed by dataset rel.

    Cached on (path, mtime) — the caption panel asks for one image at a time as
    the selection moves, and a batch report is not something to re-parse per
    click.
    """
    shape = SHAPES.get(stage)
    if shape is None:
        raise ProposalError(f"{stage} does not propose captions")
    if not report_path.is_file():
        raise ProposalError(f"report not found: {report_path}")
    return _cached(str(report_path), report_path.stat().st_mtime, roots, stage)


@lru_cache(maxsize=8)
def _cached(path: str, mtime: float, roots: D.Roots, stage: str) -> dict[str, Proposal]:
    # `mtime` is unused on purpose: it is part of the cache key, so a re-run's
    # report misses the cache instead of serving the previous run's proposals.
    shape = SHAPES[stage]
    root = roots.src if shape.root == "src" else roots.dst
    out: dict[str, Proposal] = {}
    for row, before, after in _walk(load(Path(path)), shape):
        image = str(row.get("image") or "")
        rel = D.rel_for_image(roots, image)
        if rel is None:
            continue
        caption_path = str(row.get("caption_path") or "")
        out[rel] = Proposal(
            rel=rel,
            image=image,
            kind=CAPTION_KIND[shape.root],
            path=D.rel_to_home(root / caption_path) if caption_path else "",
            before=before,
            after=after,
            status=str(row.get("status") or "proposed"),
        )
    return out


def undo(report_path: Path, roots: D.Roots, stage: str) -> dict[str, Any]:
    """Put back the captions the ``--apply`` run in ``report_path`` wrote.

    Only rows the file still agrees with are touched: a caption edited since the
    apply reads as ``drifted`` and is left alone, one already back at its
    before-text is ``already-undone``. A row whose before-text was *empty* was a
    file the run created (autotag ``missing`` mode), so its inverse is a delete,
    not a write of nothing.

    Returns the images restored — which is what the caller reloads — plus a
    count per skip reason, so the run bar can say what it could not put back.
    """
    shape = SHAPES.get(stage)
    if shape is None:
        raise ProposalError(f"{stage} writes no captions to undo")
    root = roots.src if shape.root == "src" else roots.dst
    report = load(report_path)
    restored: list[str] = []
    removed: list[str] = []
    skipped: Counter[str] = Counter()

    for row, before, after in _walk(report, shape):
        if str(row.get("status") or "").startswith("skip:"):
            skipped["not-written"] += 1
            continue
        caption_path = str(row.get("caption_path") or "")
        if not caption_path:
            skipped["no-caption-path"] += 1
            continue
        try:
            target = D.under_home(root / caption_path)
        except D.DatasetError:
            skipped["outside-home"] += 1
            continue
        if not target.is_file():
            # Absent is the undone state for a row that created its file.
            skipped["already-undone" if not before else "missing"] += 1
            continue
        current = target.read_text(encoding="utf-8").strip()
        if current == before:
            skipped["already-undone"] += 1
            continue
        if current != after:
            skipped["drifted"] += 1
            continue
        image = str(row.get("image") or "")
        if before:
            target.write_text(
                before + ("\n" if shape.newline else ""), encoding="utf-8"
            )
            restored.append(image)
        else:
            target.unlink()
            removed.append(image)
        if shape.drop_variants:
            # The apply dropped ``{stem}.variants.txt`` because the sidecar wins
            # over the caption at encode time; a regenerated one is just as
            # stale against the caption we just put back.
            from anime_tools.captions.variants import variants_sidecar_path

            sidecar = variants_sidecar_path(target)
            if sidecar.is_file():
                sidecar.unlink()

    images = [*restored, *removed]
    return {
        "stage": stage,
        "report": D.rel_to_home(report_path),
        "restored": len(restored),
        "removed": len(removed),
        "skipped": dict(skipped.most_common()),
        # The dataset rels the sidebar should re-stat, same contract as a job's
        # ``written`` list.
        "written": [r for r in (D.rel_for_image(roots, i) for i in images) if r],
    }
