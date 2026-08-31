"""Publish the workspace to the paths the trainer reads.

The workspace is where every stage writes; this is the one thing that leaves it,
so until it runs a curation pass is a proposal the trainer cannot see.

Six artifact kinds, and where each lands (``docs/contract.md`` §2 is the
destination side of this table):

``image``     ``workspace/resized/<rel>``           → ``post_image_dataset/resized/<rel>``
``caption``   ``workspace/resized/<rel>.txt``       → ``post_image_dataset/resized/<rel>.txt``
``variants``  ``workspace/resized/<rel>.variants.txt`` → beside the caption
``mask``      ``workspace/masks/<sub>/<stem>_mask.png`` → ``post_image_dataset/masks/…``
``master``    ``workspace/master/<rel>.txt``        → ``image_dataset/<rel>.txt``
``index``     ``workspace/captions/caption_index.json`` → ``post_image_dataset/captions/…``

``master`` is the odd one: it publishes back over the *input* tree, because
that is where the contract says the master lives. The only row that writes
outside ``--out`` and the only one that can overwrite something a human
hand-wrote — hence copied only when the overlay holds a revision, with its
previous text recorded for the revert.

**Always copies**, never links, so the export tree is independent bytes that
survive the workspace being cleared. Re-exporting is cheap anyway: an identical
destination is skipped and :func:`shutil.copy2` preserves mtime, so a second
export is a walk and a stat apiece.

Rows are per *artifact*, not per image, each decided on its own — so a caption
that changed publishes without recopying the pixels beside it.

Torch-free.
"""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from anime_tools._walk import walk_images
from anime_tools.captions.variants import variants_sidecar_path
from anime_tools.masking._masks import mask_name, mask_path_for

KINDS = ("image", "caption", "variants", "mask", "master", "index")
"""Every artifact kind, in the order :func:`plan_export` emits them per image."""

TEXT_KINDS = frozenset({"caption", "variants", "master", "index"})
"""Kinds compared (and reverted) by content rather than by stat.

Captions are small and a byte compare is exact; pixels go by ``(size,
mtime_ns)``, because hashing a resized tree on every dry run would cost more
than ``copy2`` already gives in certainty.
"""


@dataclass(frozen=True)
class ExportPaths:
    """The six directories one export reads from and writes to.

    ``src`` and ``out`` are destinations here, not sources: this stage runs the
    pipeline backwards, but the names keep the meaning they have everywhere else
    (``src`` the caption master tree, ``out`` the export root).
    """

    resized: Path
    masks: Path
    master: Path
    index: Path
    src: Path
    out: Path


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

    def to_dict(self) -> dict[str, object]:
        return dict(vars(self))


@dataclass
class ExportStats:
    rows: int = 0
    created: int = 0
    overwrote: int = 0
    by_kind: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "created": self.created,
            "overwrote": self.overwrote,
            "by_kind": dict(sorted(self.by_kind.items())),
            "skipped": dict(sorted(self.skipped.items())),
        }


def _same(src: Path, dst: Path, *, text: bool) -> bool:
    """Is the destination already this file?

    Pixels are compared by ``(size, mtime_ns)``, which :func:`shutil.copy2`
    makes true again on every copy — so an unchanged image compares equal on the
    next export without either side being read.
    """
    try:
        if text:
            return src.read_bytes() == dst.read_bytes()
        a, b = src.stat(), dst.stat()
        return (a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns)
    except OSError:
        return False


def _decide(row: ExportRow) -> ExportRow:
    """Fill in ``status`` (and ``before``) from what is on disk right now."""
    src, dst = Path(row.src), Path(row.dst)
    text = row.kind in TEXT_KINDS
    if not src.is_file():
        row.status = "missing-source"
    elif not dst.exists():
        row.status = "would-create"
    elif _same(src, dst, text=text):
        row.status = "identical"
    else:
        row.status = "would-overwrite"
        if text:
            try:
                row.before = dst.read_text(encoding="utf-8")
            except OSError:
                row.before = ""
    return row


def _row(rel: Path, kind: str, src: Path, dst: Path) -> ExportRow:
    return _decide(ExportRow(rel=rel.as_posix(), kind=kind, src=str(src), dst=str(dst)))


def _mask_source(paths: ExportPaths, image: Path, rel: Path) -> Path:
    """The mask for ``rel``: the mirrored layout, or the legacy flat one.

    Same two-step lookup ``gui.dataset.mask_path`` does — a flat mask tree is
    still a valid one to publish.
    """
    nested = mask_path_for(image, paths.resized, paths.masks)
    if nested.is_file():
        return nested
    return paths.masks / mask_name(rel.stem)


def plan_export(
    paths: ExportPaths, *, path_pattern: str | None = "*"
) -> list[ExportRow]:
    """Every artifact this export would publish, decided against disk.

    Enumerates the *resized* tree, because that is what curation produced: an
    image only in the caption master was never resized, and publishing it would
    just re-publish the input.

    An artifact absent from the workspace contributes no row rather than a
    ``missing-source`` one, which is left for the report's own replay — where a
    file that vanished between the plan and the apply is worth saying aloud.
    """
    rows: list[ExportRow] = []
    for image in walk_images(paths.resized, recursive=True, pattern=path_pattern):
        rel = image.relative_to(paths.resized)
        out_image = paths.out / "resized" / rel
        rows.append(_row(rel, "image", image, out_image))

        caption = image.with_suffix(".txt")
        if caption.is_file():
            rows.append(_row(rel, "caption", caption, out_image.with_suffix(".txt")))

        variants = variants_sidecar_path(caption)
        if variants.is_file():
            rows.append(
                _row(rel, "variants", variants, variants_sidecar_path(out_image))
            )

        mask = _mask_source(paths, image, rel)
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
    return rows


def export_one(row: ExportRow, *, apply: bool) -> str:
    """Copy one artifact, or say what copying it would do.

    Re-decides against disk first, which on an ``--apply`` of an older report is
    the guard that a destination edited since is reported, not clobbered blind.
    """
    _decide(row)
    if row.status in ("identical", "missing-source"):
        return row.status
    if not apply:
        return row.status
    dst = Path(row.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(row.src, dst)
    row.status = "created" if row.status == "would-create" else "overwrote"
    return row.status


def run_export(
    paths: ExportPaths,
    *,
    path_pattern: str | None = "*",
    apply: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[ExportRow], ExportStats]:
    """Plan the export and, with ``apply``, perform it."""
    rows = plan_export(paths, path_pattern=path_pattern)
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
            setattr(stats, status, getattr(stats, status) + 1)
        elif status.startswith("would-"):
            stats.by_kind[row.kind] += 1
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

ROW_FIELDS = ("rel", "kind", "src", "dst", "status", "before")


def rows_from_report(report: Mapping[str, object]) -> list[ExportRow]:
    """The rows of an export report, as :class:`ExportRow` again."""
    raw = report.get("rows")
    return [
        ExportRow(**{k: str(r.get(k) or "") for k in ROW_FIELDS})
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

    A **pixel** row it overwrote cannot be put back — the previous bytes were
    not kept, and keeping them would mean snapshotting the resized tree for an
    operation whose source still sits in the workspace. Those report
    ``not-undoable``; re-exporting is the idempotent way back.
    """
    stats = RevertStats(rows=len(rows))
    for row in rows:
        verb = _REVERT.get(row.status)
        src, dst = Path(row.src), Path(row.dst)
        if verb is None:
            row.status = "nothing-to-undo"
        elif not dst.exists():
            row.status = "already-undone"
        elif not _same(src, dst, text=row.kind in TEXT_KINDS):
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
