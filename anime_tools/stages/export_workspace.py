"""Publish the workspace to the paths the trainer reads.

Six artifact kinds, and where each lands:

``image``     ``workspace/resized/<rel>``           → ``--out/resized/<rel>``
``caption``   ``workspace/resized/<rel>.txt``       → ``--out/resized/<rel>.txt``
``variants``  ``workspace/resized/<rel>.variants.txt`` → beside the caption
``mask``      ``workspace/masks/<sub>/<stem>_mask.png`` → ``--out/masks/…``
``master``    ``workspace/master/<rel>.txt``        → ``--src/<rel>.txt``
``index``     ``workspace/captions/caption_index.json`` → ``--out/captions/…``

and once more for what curation took *out*::

``image`` …   ``workspace/_excluded/resized/<rel>``  → ``--out/_excluded/resized/<rel>``
``mask``      ``workspace/_excluded/masks/…``        → ``--out/_excluded/masks/…``

``master`` publishes back over the *input* tree, where the contract says the
master lives: the only row that writes outside ``--out`` and the only one that
can overwrite hand-written text, so it is copied only when the overlay holds a
revision, with its previous text recorded for the revert.

**Always copies**, never links, so the export tree survives the workspace being
cleared. An identical destination is skipped and :func:`shutil.copy2` preserves
mtime, so a second export is a walk and a stat apiece.

With ``combine_ocr`` (:attr:`ExportPaths.ocr` set) the ``caption`` and
``variants`` rows of an image that has a ``{stem}.ocr.txt`` are *rendered*
rather than copied: the OCR'd lines ride along as a trailing text clause
(:func:`~anime_tools.captions.ocr_sidecar.with_ocr_clause`) on the caption and
on every variant line, held to the det and glyph floors the paths carry. The workspace files stay as curated, so the combine is a
property of the published tree, taken back by exporting without it. Such a row
carries the sidecar it read (``ocr``) and the text it publishes (``text``); the
text is re-derived from disk whenever the row is decided, and a revert compares
against the text recorded at the time.

The excluded tree (:mod:`anime_tools.exclude`) publishes as a straight mirror
under ``<out>/_excluded/``: same kinds, same compare, no OCR combine (its
sidecars moved in with it) and no master row (a hand-written caption never left
``--src``). It lands *beside* the trainer's tree rather than in it, so an
excluded image is still there to look at and is never trained on. The rows carry
:attr:`ExportRow.excluded`, which is the only thing that tells them apart in the
report.

Rows are per *artifact*, not per image, each decided on its own.

Torch-free.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from anime_tools import workspace as WS
from anime_tools._walk import walk_images
from anime_tools.captions._sidecar import render_rows, sidecar_header
from anime_tools.captions.ocr_sidecar import (
    DEFAULT_MIN_DET,
    DEFAULT_MIN_GLYPH,
    ocr_sidecar_path,
    read_ocr,
    with_ocr_clause,
)
from anime_tools.captions.variants import (
    read_variants_sidecar,
    variants_sidecar_path,
)
from anime_tools.masking._masks import mask_name, mask_path_for

KINDS = ("image", "caption", "variants", "mask", "master", "index")
"""Every artifact kind, in the order :func:`plan_export` emits them per image."""

TEXT_KINDS = frozenset({"caption", "variants", "master", "index"})
"""Kinds compared (and reverted) by content rather than by stat. Captions are
small enough for a byte compare; pixels go by ``(size, mtime_ns)``."""

COMBINE_KINDS = frozenset({"caption", "variants"})
"""The kinds ``combine_ocr`` renders with the text clause attached — what the
trainer encodes. The master is hand-written and stays so; the index carries no
caption text."""


@dataclass(frozen=True)
class ExportPaths:
    """The directories one export reads from and writes to.

    ``src`` (the caption master tree) and ``out`` (the export root) are
    destinations here, not sources, but keep their usual names.
    """

    resized: Path
    masks: Path
    master: Path
    index: Path
    src: Path
    out: Path
    excluded: Path | None = None
    """The excluded tree to republish under ``<out>/_excluded/``, or ``None``.
    Set by ``--excluded_dir``; an absent tree contributes no rows."""
    ocr: Path | None = None
    """The OCR tree to combine from, or ``None`` to publish captions as they
    are. Set by ``--combine_ocr``; mirrors ``resized``."""
    ocr_min_det: float = DEFAULT_MIN_DET
    """The detector confidence a sidecar line needs to reach the published
    caption (``--ocr_min_det``); the sidecar itself keeps every line."""
    ocr_min_glyph: float = DEFAULT_MIN_GLYPH
    """The glyph size, in the read image's pixels, a sidecar line needs to
    reach the published caption (``--ocr_min_glyph``)."""


@dataclass
class ExportRow:
    """One artifact and what publishing it would do."""

    rel: str
    kind: str
    src: str
    dst: str
    status: str = "would-create"
    before: str = ""
    """The destination's previous text, for the kinds a revert can put back.
    Empty for a pixel kind, and for a destination that did not exist."""
    ocr: str = ""
    """The OCR sidecar a combined row read, or empty for a verbatim copy."""
    ocr_min_det: float = DEFAULT_MIN_DET
    """The det floor the combine held the sidecar's lines to — on the row, so a
    replay of the report publishes what the plan showed."""
    ocr_min_glyph: float = DEFAULT_MIN_GLYPH
    """The glyph floor the combine held the sidecar's lines to; on the row for
    the same reason as ``ocr_min_det``."""
    text: str = ""
    """What a combined row publishes — the source with the text clause attached.
    Empty for a verbatim copy, whose bytes are the source's."""
    excluded: bool = False
    """Came out of the excluded tree, so it publishes under ``<out>/_excluded/``
    rather than into the tree the trainer reads. The kind is the live one, so
    nothing about the compare or the revert changes."""

    @property
    def combined(self) -> bool:
        return bool(self.ocr)

    def to_dict(self) -> dict[str, object]:
        return dict(vars(self))


@dataclass
class ExportStats:
    rows: int = 0
    created: int = 0
    overwrote: int = 0
    combined: int = 0
    """Rows published (or, dry, to be published) with the OCR clause attached."""
    excluded: int = 0
    """Rows published under ``<out>/_excluded/`` rather than into the trainer's
    tree."""
    by_kind: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "created": self.created,
            "overwrote": self.overwrote,
            "combined": self.combined,
            "excluded": self.excluded,
            "by_kind": dict(sorted(self.by_kind.items())),
            "skipped": dict(sorted(self.skipped.items())),
        }


def _combine(kind: str, src: Path, lines, *, min_det: float, min_glyph: float) -> str:
    """The text a combined row publishes: the caption, or every variant line,
    with the OCR clauses attached (the lines that clear both floors)."""
    if kind == "caption":
        return with_ocr_clause(
            src.read_text(encoding="utf-8"),
            lines,
            min_det=min_det,
            min_glyph=min_glyph,
        )
    rows = [
        (label, with_ocr_clause(text, lines, min_det=min_det, min_glyph=min_glyph))
        for label, text in read_variants_sidecar(src)
    ]
    return render_rows(sidecar_header("variants"), rows)


def _derive(row: ExportRow) -> None:
    """Refresh a combined row's ``text`` from the source and sidecar as they
    are on disk now. A sidecar that has gone away publishes the caption with
    no text clause — the OCR pass deletes it when it finds nothing."""
    if row.combined:
        row.text = _combine(
            row.kind,
            Path(row.src),
            read_ocr(Path(row.ocr)),
            min_det=row.ocr_min_det,
            min_glyph=row.ocr_min_glyph,
        )


def _published(row: ExportRow) -> bytes:
    """The bytes this row puts at its destination."""
    if row.combined:
        return row.text.encode("utf-8")
    return Path(row.src).read_bytes()


def _same(row: ExportRow, dst: Path) -> bool:
    """Is the destination already what this row publishes?

    Pixels are compared by ``(size, mtime_ns)``, which :func:`shutil.copy2`
    preserves, so an unchanged image compares equal without being read.
    """
    try:
        if row.kind in TEXT_KINDS:
            return _published(row) == dst.read_bytes()
        a, b = Path(row.src).stat(), dst.stat()
        return (a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns)
    except OSError:
        return False


def _decide(row: ExportRow) -> ExportRow:
    """Fill in ``status`` (and ``before``) from what is on disk right now."""
    src, dst = Path(row.src), Path(row.dst)
    text = row.kind in TEXT_KINDS
    if not src.is_file():
        row.status = "missing-source"
        return row
    _derive(row)
    if not dst.exists():
        row.status = "would-create"
    elif _same(row, dst):
        row.status = "identical"
    else:
        row.status = "would-overwrite"
        if text:
            try:
                row.before = dst.read_text(encoding="utf-8")
            except OSError:
                row.before = ""
    return row


def _row(
    rel: Path,
    kind: str,
    src: Path,
    dst: Path,
    *,
    ocr: Path | None = None,
    min_det: float = DEFAULT_MIN_DET,
    min_glyph: float = DEFAULT_MIN_GLYPH,
    excluded: bool = False,
) -> ExportRow:
    """One row, decided. ``ocr`` (the image's sidecar, when combining) marks it
    combined only if the sidecar exists — an image with no text publishes its
    caption verbatim, as a plain copy."""
    combined = ocr is not None and kind in COMBINE_KINDS and ocr.is_file()
    return _decide(
        ExportRow(
            rel=rel.as_posix(),
            kind=kind,
            src=str(src),
            dst=str(dst),
            ocr=str(ocr) if combined else "",
            ocr_min_det=min_det,
            ocr_min_glyph=min_glyph,
            excluded=excluded,
        )
    )


def _mask_source(masks: Path, images: Path, image: Path, rel: Path) -> Path:
    """The mask for ``rel``: the mirrored layout, or the legacy flat one — the
    same two-step lookup ``gui.dataset.mask_path`` does.

    Takes the two roots rather than the :class:`ExportPaths` so the excluded
    tree's own mirror (``_excluded/masks`` over ``_excluded/resized``) resolves
    through the same rule.
    """
    nested = mask_path_for(image, images, masks)
    if nested.is_file():
        return nested
    return masks / mask_name(rel.stem)


def plan_export(paths: ExportPaths) -> list[ExportRow]:
    """Every artifact this export would publish, decided against disk.

    Enumerates the *resized* tree, which is what curation produced -- the whole
    of it, since a publish narrowed to part of the workspace is a dataset the
    trainer would read as all of it. An artifact absent from the workspace
    contributes no row at all; ``missing-source`` is left for a replay of the
    report, where the file vanished after the plan.
    """
    rows: list[ExportRow] = []
    for image in walk_images(paths.resized, recursive=True):
        rel = image.relative_to(paths.resized)
        out_image = paths.out / "resized" / rel
        rows.append(_row(rel, "image", image, out_image))

        ocr = ocr_sidecar_path(paths.ocr / rel) if paths.ocr is not None else None
        caption = image.with_suffix(".txt")
        if caption.is_file():
            rows.append(
                _row(
                    rel,
                    "caption",
                    caption,
                    out_image.with_suffix(".txt"),
                    ocr=ocr,
                    min_det=paths.ocr_min_det,
                    min_glyph=paths.ocr_min_glyph,
                )
            )

        variants = variants_sidecar_path(caption)
        if variants.is_file():
            rows.append(
                _row(
                    rel,
                    "variants",
                    variants,
                    variants_sidecar_path(out_image),
                    ocr=ocr,
                    min_det=paths.ocr_min_det,
                    min_glyph=paths.ocr_min_glyph,
                )
            )

        mask = _mask_source(paths.masks, paths.resized, image, rel)
        if mask.is_file():
            rows.append(
                _row(
                    rel,
                    "mask",
                    mask,
                    paths.out / "masks" / mask.relative_to(paths.masks),
                )
            )

        revised = paths.master / rel.with_suffix(".txt")
        if revised.is_file():
            rows.append(
                _row(rel, "master", revised, paths.src / rel.with_suffix(".txt"))
            )

    if paths.index.is_file():
        rel = Path(paths.index.name)
        rows.append(
            _row(rel, "index", paths.index, paths.out / "captions" / paths.index.name)
        )
    return rows + _excluded_rows(paths)


def _excluded_rows(paths: ExportPaths) -> list[ExportRow]:
    """The excluded tree, mirrored under ``<out>/_excluded/``.

    A second, smaller plan over the same shapes: ``_excluded/resized`` is walked
    the way the live resized tree is, and ``_excluded/masks`` is looked up
    against it by the same rule, so the two halves of the export tree have the
    same layout. No ``master`` row (the hand-written caption never left ``src``,
    so an exclusion has nothing to publish back) and no OCR combine — the
    sidecars moved into ``_excluded/ocr`` with the image, and what is archived
    here is what was curated, not a render of it.

    Nothing at all when the tree is absent, which is every workspace where
    nobody has excluded anything.
    """
    root = paths.excluded
    if root is None:
        return []
    resized = root / "resized"
    if not resized.is_dir():
        return []
    masks = root / "masks"
    out = paths.out / WS.EXCLUDED_SUBDIR

    rows: list[ExportRow] = []
    for image in walk_images(resized, recursive=True):
        rel = image.relative_to(resized)
        out_image = out / "resized" / rel
        rows.append(_row(rel, "image", image, out_image, excluded=True))

        caption = image.with_suffix(".txt")
        if caption.is_file():
            rows.append(
                _row(
                    rel,
                    "caption",
                    caption,
                    out_image.with_suffix(".txt"),
                    excluded=True,
                )
            )

        variants = variants_sidecar_path(caption)
        if variants.is_file():
            rows.append(
                _row(
                    rel,
                    "variants",
                    variants,
                    variants_sidecar_path(out_image),
                    excluded=True,
                )
            )

        mask = _mask_source(masks, resized, image, rel)
        if mask.is_file():
            rows.append(
                _row(
                    rel,
                    "mask",
                    mask,
                    out / "masks" / mask.relative_to(masks),
                    excluded=True,
                )
            )
    return rows


def export_one(row: ExportRow, *, apply: bool) -> str:
    """Copy one artifact, or say what copying it would do.

    Re-decides against disk first, so an ``--apply`` of an older report reports
    a destination edited since rather than clobbering it.
    """
    _decide(row)
    if row.status in ("identical", "missing-source"):
        return row.status
    if not apply:
        return row.status
    dst = Path(row.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if row.combined:
        dst.write_text(row.text, encoding="utf-8")
    else:
        shutil.copy2(row.src, dst)
    row.status = "created" if row.status == "would-create" else "overwrote"
    return row.status


def run_export(
    paths: ExportPaths,
    *,
    apply: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[ExportRow], ExportStats]:
    """Plan the export and, with ``apply``, perform it."""
    rows = plan_export(paths)
    return _run(rows, apply=apply, progress=progress)


def _run(
    rows: list[ExportRow],
    *,
    apply: bool,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[ExportRow], ExportStats]:
    stats = ExportStats(rows=len(rows))
    for i, row in enumerate(rows, 1):
        status = export_one(row, apply=apply)
        if status in ("created", "overwrote"):
            stats.by_kind[row.kind] += 1
            stats.combined += row.combined
            stats.excluded += row.excluded
            setattr(stats, status, getattr(stats, status) + 1)
        elif status.startswith("would-"):
            stats.by_kind[row.kind] += 1
            stats.combined += row.combined
            stats.excluded += row.excluded
        else:
            stats.skip(status)
        if progress:
            progress(i, len(rows), f"{row.kind} {row.rel} — {status}")
    return rows, stats


# ---- putting an export back --------------------------------------------


@dataclass
class RevertStats:
    rows: int = 0
    removed: int = 0
    restored: int = 0
    skipped: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "removed": self.removed,
            "restored": self.restored,
            "skipped": dict(sorted(self.skipped.items())),
        }


_REVERT = {"created": "remove", "overwrote": "restore"}
"""Apply status → what putting that row back means. Anything else published
nothing and so has nothing to undo."""

ROW_FIELDS = ("rel", "kind", "src", "dst", "status", "before", "ocr", "text")
"""The text fields a report round-trips. ``excluded`` is read separately, being
the one field that is not a string."""


def rows_from_report(report: Mapping[str, object]) -> list[ExportRow]:
    """The rows of an export report, as :class:`ExportRow` again."""
    raw = report.get("rows")
    return [
        ExportRow(
            **{k: str(r.get(k) or "") for k in ROW_FIELDS},
            excluded=bool(r.get("excluded")),
        )
        for r in (raw if isinstance(raw, list) else [])
        if isinstance(r, Mapping)
    ]


def revert_export(
    rows: list[ExportRow], *, apply: bool = False
) -> tuple[list[ExportRow], RevertStats]:
    """Unpublish what an ``--apply`` export wrote.

    A row it *created* is deleted; a text row it *overwrote* is put back to the
    text recorded at the time. Both are guarded: the destination must still hold
    what the export put there, or it is left alone as ``drifted``.

    A **pixel** row it overwrote reports ``not-undoable`` — the previous bytes
    were never kept.

    A combined row is checked against the ``text`` the report recorded, not
    against a fresh combine: what was published is what must still be there.
    """
    stats = RevertStats(rows=len(rows))
    for row in rows:
        verb = _REVERT.get(row.status)
        dst = Path(row.dst)
        if verb is None:
            row.status = "nothing-to-undo"
        elif not dst.exists():
            row.status = "already-undone"
        elif not _same(row, dst):
            row.status = "drifted"
        elif verb == "restore" and row.kind not in TEXT_KINDS:
            row.status = "not-undoable"
        elif not apply:
            row.status = f"would-{verb}"
        elif verb == "remove":
            dst.unlink()
            row.status = "removed"
            stats.removed += 1
        else:
            dst.write_text(row.before, encoding="utf-8")
            row.status = "restored"
            stats.restored += 1

        if not row.status.startswith(("would-", "removed", "restored")):
            stats.skip(row.status)
    return rows, stats
