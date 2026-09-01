"""A stage report read as per-image caption proposals — and put back again.

Keys a report's rows by *dataset rel* (what the sidebar selects) so the caption
panel can show a proposal beside the caption it would replace. :func:`undo` is
the same reading in reverse, guarded on the file still holding what the run put
there. Torch-free, like the rest of :mod:`anime_tools.gui`.

The report *shapes* below are a copy of each stage CLI's ``REPLAY_SPEC`` rather
than an import: those live next to ``build_tag_fn`` / the SAM3 loaders, and
importing one would pull torch into the server process (``test_boundary`` pins
that it stays out). Only the instances are copied — the ``ReplaySpec`` class,
the report reader and the drift ladder come from
:mod:`anime_tools.stages.replay`, which is torch-free by construction.
``tests/test_gui_proposals.py`` compares the copies against the originals field
for field.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from anime_tools.gui import dataset as D
from anime_tools.stages._caption_io import read_caption
from anime_tools.stages.replay import (
    ReplaySpec,
    StaleReportError,
    apply_one,
    load_report,
    report_rows,
)

# Statuses a replayed row carries (``stages.replay.ReplayRow``); both mean
# "this row is a real proposal".
_REPLAY_OK = ("written", "would-write")


SHAPES: dict[str, ReplaySpec] = {
    "autotag": ReplaySpec(
        stage="autotag_captions",
        rows_key="rows",
        stats_key="stats",
        ok_status="ok",
        before_field="existing",
        after_field="proposed",
        target_root="src",
    ),
    "position": ReplaySpec(
        stage="position_captions",
        rows_key="images",
        stats_key="summary",
        ok_status="proposed",
        before_field="original",
        after_field="proposed",
        target_root="dst",
        drop_variants=True,
        history_by="position",
    ),
    # OCR's caption half only: a row's ``lines`` are the sidecar, which an undo
    # leaves alone. They describe the image rather than the caption, so putting
    # one back would mean re-reading the pixels, which is the run itself.
    "ocr": ReplaySpec(
        stage="ocr_captions",
        rows_key="rows",
        stats_key="stats",
        ok_status="ok",
        before_field="existing",
        after_field="proposed",
        target_root="dst",
        drop_variants=True,
        history_by="ocr",
    ),
    # The audit gates on verdict/confidence rather than a row status, so there
    # is no ``ok_status`` to match: a row is a proposal when it proposes text.
    "audit": ReplaySpec(
        stage="audit_multiview",
        rows_key="images",
        stats_key="summary",
        before_field="caption",
        after_field="proposed",
        target_root="src",
        newline=True,
    ),
}
"""GUI stage id → the report shape its CLI declares. Copied, not imported."""

CAPTION_KIND: dict[str, str] = {"src": "master", "dst": "revised"}
"""Which of the two editable captions a stage's ``target_root`` names."""

# ``apply_one``'s ladder, said from the undo side: putting a caption back is an
# apply with the two texts swapped, so its statuses need only be renamed.
_UNDO_STATUS = {
    "already-applied": "already-undone",
    "missing-caption": "missing",
    "drifted": "drifted",
    "no-proposal": "nothing-to-restore",
}


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
            # Parsed here and never in the browser: a diff that split on commas
            # would disagree with the grammar about every position clause.
            "before_parsed": D.parsed_caption(self.before) if self.before else None,
            "after_parsed": D.parsed_caption(self.after) if self.after else None,
        }


def load(path: Path) -> dict[str, Any]:
    """The stage's own reader, with its refusal re-typed for the HTTP layer."""
    try:
        return load_report(path)
    except StaleReportError as exc:
        raise ProposalError(str(exc)) from exc


def _rows(report: Mapping[str, Any], shape: ReplaySpec) -> list[Mapping[str, Any]]:
    try:
        return report_rows(report, shape)
    except StaleReportError as exc:
        raise ProposalError(str(exc)) from exc


def _texts(row: Mapping[str, Any], shape: ReplaySpec) -> tuple[str, str]:
    """``(before, after)``, from either report dialect: a stage's own report
    names them per its ``ReplaySpec``, while a replay's always says
    ``before``/``after`` — and Apply is a replay, so both reach here."""
    before, after = row.get(shape.before_field), row.get(shape.after_field)
    if before is None and after is None:
        before, after = row.get("before"), row.get("after")
    return str(before or "").strip(), str(after or "").strip()


def _proposes(row: Mapping[str, Any], shape: ReplaySpec) -> bool:
    status = row.get("status")
    if status is None:
        return True
    if status in _REPLAY_OK:
        return True
    return shape.ok_status is None or status == shape.ok_status


def _walk(
    report: Mapping[str, Any], shape: ReplaySpec
) -> Iterator[tuple[Mapping[str, Any], str, str]]:
    """Every row that carries a real change, with its texts."""
    for row in _rows(report, shape):
        before, after = _texts(row, shape)
        if not after or after == before or not _proposes(row, shape):
            continue
        yield row, before, after


def read(report_path: Path, roots: D.Roots, stage: str) -> dict[str, Proposal]:
    """The proposals in ``report_path``, keyed by dataset rel. Cached on
    (path, mtime), since the panel asks one image at a time as the selection
    moves."""
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
    root = roots.src if shape.target_root == "src" else roots.dst
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
            kind=CAPTION_KIND[shape.target_root],
            path=D.rel_to_home(root / caption_path) if caption_path else "",
            before=before,
            after=after,
            status=str(row.get("status") or "proposed"),
        )
    return out


EXPORT_STAGE = "export"
"""The one stage whose report is not a caption diff.

Its rows are file copies, read back by
:mod:`anime_tools.stages.export_workspace` rather than by the drift ladder, but
it reaches the server through the same route and answers in the same shape — so
the branch lives at the top of :func:`undo`, not in a second entry point.
"""


def _undo_export(report_path: Path, roots: D.Roots) -> dict[str, Any]:
    """Unpublish an export: delete what it created, put back what it overwrote."""
    from anime_tools.stages.export_workspace import revert_export, rows_from_report

    rows, stats = revert_export(rows_from_report(load(report_path)), apply=True)
    # An export row's rel is relative to the *resized* tree, so it goes back
    # through the same sibling lookup a report's ``image`` does.
    touched = [
        D.rel_for_image(roots, r.rel)
        for r in rows
        if r.status in ("removed", "restored") and r.kind != "index"
    ]
    return {
        "stage": EXPORT_STAGE,
        "report": D.rel_to_home(report_path),
        "restored": stats.restored,
        "removed": stats.removed,
        "skipped": dict(stats.skipped.most_common()),
        "written": [r for r in touched if r],
    }


def undo(report_path: Path, roots: D.Roots, stage: str) -> dict[str, Any]:
    """Put back what the ``--apply`` run in ``report_path`` wrote.

    Only rows the file still agrees with are touched: a caption edited since the
    apply reads as ``drifted`` and is left alone. Returns the images restored —
    what the caller reloads — plus a count per skip reason.
    """
    if stage == EXPORT_STAGE:
        return _undo_export(report_path, roots)
    shape = SHAPES.get(stage)
    if shape is None:
        raise ProposalError(f"{stage} writes no captions to undo")
    root = roots.src if shape.target_root == "src" else roots.dst
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
            target = D.reachable(root / caption_path)
        except D.DatasetError:
            skipped["outside-dataset"] += 1
            continue
        image = str(row.get("image") or "")
        if before:
            # An undo is an apply with the two texts swapped — same drift
            # ladder, same write, and the same drop of the variants sidecar,
            # which wins over the caption at encode time and is just as stale
            # against the text being put back. It supersedes a version like any
            # other write, so the text being undone is filed under ``undo``
            # rather than vanishing: an undo you did not mean is then one badge
            # away, which is the whole promise of the history rung.
            status = apply_one(
                target,
                after,
                before,
                apply=True,
                newline=shape.newline,
                drop_variants=shape.drop_variants,
                history_by="undo" if shape.history_by else None,
            )
            if status != "written":
                skipped[_UNDO_STATUS.get(status, status)] += 1
                continue
            restored.append(image)
        else:
            # An empty before-text was a file the run *created* (autotag's
            # ``missing`` mode), so the inverse is a delete, not a write of
            # nothing — but it is gated on the same two questions.
            if not target.is_file():
                skipped["already-undone"] += 1
                continue
            if read_caption(target) != after:
                skipped["drifted"] += 1
                continue
            target.unlink()
            removed.append(image)
            if shape.drop_variants:
                from anime_tools.captions.variants import variants_sidecar_path

                sidecar = variants_sidecar_path(target)
                if sidecar.is_file():
                    sidecar.unlink()
            if shape.history_by:
                # The versions of a caption that no longer exists are versions
                # of nothing — and this run is what created it, so there is no
                # earlier text the history could still be describing.
                from anime_tools.captions.history import drop_history

                drop_history(target)

    images = [*restored, *removed]
    return {
        "stage": stage,
        "report": D.rel_to_home(report_path),
        "restored": len(restored),
        "removed": len(removed),
        "skipped": dict(skipped.most_common()),
        # The rels the sidebar should re-stat, like a job's ``written`` list.
        "written": [r for r in (D.rel_for_image(roots, i) for i in images) if r],
    }
