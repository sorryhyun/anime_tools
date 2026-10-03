"""Publish the workspace to the paths the trainer reads.

Six artifact kinds, and where each lands:

``image``     ``--src/<dir>/<stem>.*``              → ``--out/resized/<dir>/<stem>.*``
``caption``   ``workspace/resized/<rel>.txt``       → ``--out/resized/<rel>.txt``
``variants``  ``workspace/resized/<rel>.variants.txt`` → beside the caption
``mask``      ``workspace/masks/<sub>/<stem>_mask.png`` → ``--out/masks/…``
``master``    ``workspace/master/<rel>.txt``        → ``--src/<rel>.txt``
``index``     ``workspace/captions/caption_index.json`` → ``--out/captions/…``

The ``image`` row publishes the *original* each resized image stands for —
the file under ``--src`` with the same directory and stem, whatever its
extension, byte for byte (the resized PNG only when the original has gone).
The trainer buckets it itself. Two knobs make it a *render* instead
(:func:`_render`): ``resize_cap`` (:attr:`ExportPaths.cap`, the request's
``resize_cap_tokens``, 4200 by default) downscales an original over that token
count to it at its native aspect, and ``webp`` (:attr:`ExportPaths.webp`) re-encodes every original not
already in WebP as ``{stem}.webp``. A render carries the original's mtime, so a
re-export compares it by mtime and header size without a decode (:func:`_same`).
The ``mask`` row follows its image: the workspace mask is at the resized
geometry, so it is mapped back through the resize's own crop (the anchor and
margins stamped on the resized PNG, the cropped-away strip edge-padded) and
scaled to the published image's size (:func:`_render_mask`).

The ``caption`` row reads the caption ladder rather than one file
(:func:`_caption_source`): the revised caption when there is one, the master
otherwise — the overlay, then the hand-written original under ``--src``. Nothing
copies a master into the resized tree, so an image no caption stage has touched
would otherwise publish captionless, which the trainer reads as unlabelled. The
``variants`` sidecar stays a revised-tree artifact and is looked up there
whichever rung the caption came from.

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

With ``sidecars_only`` (:attr:`ExportPaths.images` off) no pixels publish at
all: no ``image`` rows and no excluded mirror, only the decisions -- captions,
variants, masks, the master and the index. The trainer then resizes from
``--src`` itself, skipping what the ``_excluded`` ledger names, and the captions
land beside the PNGs it writes. Each mask is fitted to its original at the
original's own size, the geometry the trainer's resize starts from.

The excluded tree (:mod:`anime_tools.exclude`) publishes as a straight mirror
under ``<out>/_excluded/``: same kinds, same compare, no OCR combine (its
sidecars moved in with it) and no master row (a hand-written caption never left
``--src``). It lands *beside* the trainer's tree rather than in it, so an
excluded image is still there to look at and is never trained on. The rows carry
:attr:`ExportRow.excluded`, which is the only thing that tells them apart in the
report.

Rows are per *artifact*, not per image, each decided on its own.

:func:`publish` plans and performs one export; :func:`run_export` is the stage
runner over :class:`~anime_tools.stages.requests.ExportRequest`, which is where
the workspace roots become an :class:`ExportPaths`.

Torch-free.
"""

from __future__ import annotations

import math
import os
import shutil
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from anime_tools import workspace as WS
from anime_tools._env import resolve_path
from anime_tools._walk import IMAGE_EXTENSIONS, sibling_image, suffix_key, walk_images
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
from anime_tools.stages._progress import make_progress
from anime_tools.stages._report import (
    print_dry_run_footer,
    stage_report_header,
    write_stage_report,
)

if TYPE_CHECKING:
    from anime_tools.stages.requests import ExportRequest

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
    cap: int = 0
    """The token ceiling a published image is held to, or 0 to publish every
    original at its own size. ``--resize_cap_tokens`` under ``--resize_cap``."""
    webp: bool = False
    """Publish every image as ``{stem}.webp`` (``--webp``)."""
    images: bool = True
    """Publish the images (and the excluded mirror), or only the sidecars
    beside where they would land. Off is ``--sidecars_only``; ``cap`` and
    ``webp`` then have nothing to shape."""


WEBP_QUALITY = 95


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
    cap: int = 0
    """The token ceiling an ``image`` row downscales its source to, or 0. Set
    only on a source that is over the export's cap."""
    render: bool = False
    """Written by :func:`_render` / :func:`_render_mask` rather than copied: an
    image downscaled, re-encoded or both (the format is the destination's
    suffix), or a mask fitted to the published image."""
    ref: str = ""
    """A ``mask`` row's original image: the published image's size is read off
    it (capped by ``cap``), and the mask is fitted to that."""
    fit: str = ""
    """A ``mask`` row's resized image — the geometry the mask was drawn at, and
    whose ``anima_resize_*`` keys say how the resize cropped."""

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
    capped: int = 0
    """Images published downscaled to the cap."""
    rendered: int = 0
    """Images published re-encoded (capped, converted or both) rather than
    copied."""
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
            "capped": self.capped,
            "rendered": self.rendered,
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


def capped_size(size: tuple[int, int], cap: int) -> tuple[int, int] | None:
    """``size`` downscaled so its 16 px patch grid holds at most ``cap`` tokens,
    aspect kept — or ``None`` when it already does (or ``cap`` is 0).

    Floored, so the result never lands a token over the ceiling.
    """
    w, h = size
    if cap <= 0 or w * h <= cap * 256:
        return None
    scale = math.sqrt(cap * 256 / (w * h))
    return max(1, math.floor(w * scale)), max(1, math.floor(h * scale))


def _oriented(path: Path) -> tuple[int, int] | None:
    """An image's upright ``(W, H)`` from its header, or ``None`` if unreadable."""
    from PIL import Image

    from anime_tools.stages.resize import _oriented_size

    try:
        with Image.open(path) as im:
            return _oriented_size(im)
    except (OSError, ValueError):
        return None


def _render_size(row: ExportRow) -> tuple[int, int] | None:
    """The upright size a rendered row writes — its image's, capped — read now."""
    size = _oriented(Path(row.ref or row.src))
    if size is None:
        return None
    return capped_size(size, row.cap) or size


def _render(row: ExportRow, dst: Path) -> None:
    """Write the source upright, downscaled to the cap if it has one, in the
    format ``dst``'s suffix names, then give it the source's mtime — which is
    what :func:`_same` recognises it by."""
    from PIL import Image, ImageOps

    from anime_tools.stages.resize import _collect_metadata

    src = Path(row.src)
    fmt = Image.registered_extensions().get(suffix_key(dst), "PNG")
    with Image.open(src) as im:
        save_kwargs = _collect_metadata(im)
        img = ImageOps.exif_transpose(im)
        # The transpose rewrote the orientation tag; the source's would turn
        # the upright pixels a second time.
        save_kwargs.pop("exif", None)
        if exif := img.info.get("exif"):
            save_kwargs["exif"] = exif
        if img.mode == "P":
            img = img.convert("RGBA")
        if target := capped_size(img.size, row.cap):
            img = img.resize(target, Image.Resampling.LANCZOS)
        else:
            img.load()
    if fmt == "JPEG":
        if img.mode not in ("RGB", "L", "CMYK"):
            img = img.convert("RGB")
        save_kwargs["quality"] = 95
    elif fmt == "WEBP":
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.mode else "RGB")
        save_kwargs["quality"] = WEBP_QUALITY
    if fmt != "PNG":
        save_kwargs.pop("pnginfo", None)
    img.save(dst, format=fmt, **save_kwargs)
    st = src.stat()
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))


def _uncrop(mask, orig: tuple[int, int], anchor: str, margins):
    """A mask drawn on a resized image, back on the ``orig``-sized original.

    The inverse of ``resize_to_bucket`` over ``margin_box``: the cover-scale's
    cropped-away strip and the margins are edge-padded (no one drew there), the
    scale is undone with NEAREST so a hard mask stays hard.
    """
    import numpy as np
    from PIL import Image

    from anime_tools.stages.resize import CROP_ANCHORS, margin_box

    W, H = orig
    bw, bh = mask.size
    x0, y0, x1, y1 = margin_box(W, H, margins)
    ww, wh = x1 - x0, y1 - y0
    if ww / wh > bw / bh:
        nw, nh = round(bh * ww / wh), bh
    else:
        nw, nh = bw, round(bw * wh / ww)
    ax, ay = CROP_ANCHORS[anchor]
    left, top = round((nw - bw) * ax), round((nh - bh) * ay)
    scaled = np.pad(
        np.asarray(mask), ((top, nh - bh - top), (left, nw - bw - left)), mode="edge"
    )
    work = Image.fromarray(scaled).resize((ww, wh), Image.Resampling.NEAREST)
    full = np.pad(np.asarray(work), ((y0, H - y1), (x0, W - x1)), mode="edge")
    return Image.fromarray(full)


def _render_mask(row: ExportRow, dst: Path) -> None:
    """Fit the workspace mask to the published image and stamp the mask's mtime.

    Through the resize's crop when the mask's image was resized from ``ref``;
    a plain scale when there is no original (``fit`` is ``ref``).
    """
    from PIL import Image

    from anime_tools.stages.resize import (
        _ANCHOR_KEY,
        _MARGINS_KEY,
        normalize_crop_anchor,
        normalize_crop_margins,
    )

    target = _render_size(row)
    orig = _oriented(Path(row.ref))
    if target is None or orig is None:
        raise OSError(f"cannot read the image this mask belongs to: {row.ref}")
    src = Path(row.src)
    with Image.open(src) as m:
        mask = m.convert("L")
    if row.fit and Path(row.fit) != Path(row.ref):
        with Image.open(row.fit) as fit:
            bucket = fit.size
            text = getattr(fit, "text", {}) or {}
        if mask.size != bucket:
            mask = mask.resize(bucket, Image.Resampling.NEAREST)
        mask = _uncrop(
            mask,
            orig,
            normalize_crop_anchor(text.get(_ANCHOR_KEY)),
            normalize_crop_margins(text.get(_MARGINS_KEY)),
        )
    if mask.size != target:
        mask = mask.resize(target, Image.Resampling.NEAREST)
    mask.save(dst, format="PNG")
    st = src.stat()
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))


def _published(row: ExportRow) -> bytes:
    """The bytes this row puts at its destination."""
    if row.combined:
        return row.text.encode("utf-8")
    return Path(row.src).read_bytes()


def _same(row: ExportRow, dst: Path) -> bool:
    """Is the destination already what this row publishes?

    Pixels are compared by ``(size, mtime_ns)``, which :func:`shutil.copy2`
    preserves, so an unchanged image compares equal without being read. A
    render is never the source's byte size; it carries the source's mtime, and
    its header says whether it was rendered at this row's size.
    """
    try:
        if row.kind in TEXT_KINDS:
            return _published(row) == dst.read_bytes()
        a, b = Path(row.src).stat(), dst.stat()
        if row.render:
            return a.st_mtime_ns == b.st_mtime_ns and _oriented(dst) == _render_size(
                row
            )
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
                row.before = dst.read_text(encoding="utf-8", newline="")
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


def _caption_source(paths: ExportPaths, image: Path, rel: Path) -> Path | None:
    """The caption that speaks for ``rel``, or ``None`` if none does.

    The caption ladder, read from the bottom: the **revised** caption
    (``workspace/resized/<rel>.txt``), then the **master** — the workspace
    overlay first, the hand-written original under ``--src`` behind it, the
    same overlay-first rule ``gui.dataset.caption_paths`` and
    ``_walk_captions.resolve_caption`` read.

    Falling back matters because nothing copies the master into the resized
    tree: an image no caption stage has touched has only its hand-written
    master, and publishing it captionless is publishing a dataset the trainer
    reads as unlabelled.
    """
    txt = rel.with_suffix(".txt")
    for path in (image.with_suffix(".txt"), paths.master / txt, paths.src / txt):
        if path.is_file():
            return path
    return None


def plan_export(paths: ExportPaths) -> list[ExportRow]:
    """Every artifact this export would publish, decided against disk.

    Enumerates the *resized* tree, which is what curation produced -- the whole
    of it, since a publish narrowed to part of the workspace is a dataset the
    trainer would read as all of it. An artifact absent from the workspace
    contributes no row at all; ``missing-source`` is left for a replay of the
    report, where the file vanished after the plan.
    """
    rows: list[ExportRow] = []
    # No image publishes, so a mask is fitted to the original as it is.
    cap = paths.cap if paths.images else 0
    for image in walk_images(paths.resized, recursive=True):
        rel = image.relative_to(paths.resized)
        source = sibling_image(paths.src / rel.parent, rel.stem) or image
        suffix = ".webp" if paths.webp else source.suffix
        out_image = paths.out / "resized" / rel.with_suffix(suffix)
        if paths.images:
            rows.append(_image_row(rel, source, out_image, cap=paths.cap))

        ocr = ocr_sidecar_path(paths.ocr / rel) if paths.ocr is not None else None
        caption = _caption_source(paths, image, rel)
        if caption is not None:
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

        variants = variants_sidecar_path(image.with_suffix(".txt"))
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
            dst = paths.out / "masks" / mask.relative_to(paths.masks)
            rows.append(_mask_row(rel, mask, dst, image=image, source=source, cap=cap))

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
    # Sidecars only: the trainer reads exclusions from the ledger, and an
    # archive of captions without their images is nothing to look at.
    return rows + (_excluded_rows(paths) if paths.images else [])


def _image_row(rel: Path, source: Path, dst: Path, *, cap: int) -> ExportRow:
    """The ``image`` row: a verbatim copy, or a render when ``source`` is over
    ``cap`` or ``dst`` names another format. The cap is decided from the
    header, so an image under it is never decoded."""
    size = _oriented(source) if cap else None
    over = size is not None and capped_size(size, cap) is not None
    converts = suffix_key(dst) != suffix_key(source)
    return _decide(
        ExportRow(
            rel=rel.as_posix(),
            kind="image",
            src=str(source),
            dst=str(dst),
            cap=cap if over else 0,
            render=over or converts,
        )
    )


def _mask_row(
    rel: Path, mask: Path, dst: Path, *, image: Path, source: Path, cap: int
) -> ExportRow:
    """The ``mask`` row, fitted to what the ``image`` row publishes: a render
    unless that is the resized image itself at its own size."""
    size = _oriented(source) if cap else None
    over = size is not None and capped_size(size, cap) is not None
    render = source != image or over
    return _decide(
        ExportRow(
            rel=rel.as_posix(),
            kind="mask",
            src=str(mask),
            dst=str(dst),
            cap=cap if render else 0,
            render=render,
            ref=str(source) if render else "",
            fit=str(image) if render else "",
        )
    )


def stale_siblings(rows: list[ExportRow]) -> list[Path]:
    """Images beside a published one that share its stem but not its suffix.

    The ``.png`` an earlier export published sits beside the ``.jpg`` original
    (or the ``.webp``) this one does, and the trainer refuses a folder with two
    images of one stem. Export never deletes, so these are named instead.
    """
    exts = dict.fromkeys(suffix_key(Path(f"x{e}")) for e in IMAGE_EXTENSIONS)
    out: list[Path] = []
    for row in rows:
        if row.kind != "image":
            continue
        dst = Path(row.dst)
        for ext in exts:
            other = dst.with_suffix(ext)
            if ext != suffix_key(dst) and other.is_file():
                out.append(other)
    return out


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


def export_one(row: ExportRow, *, apply: bool, decided: bool = False) -> str:
    """Copy one artifact, or say what copying it would do.

    Re-decides against disk first, so an ``--apply`` of an older report reports
    a destination edited since rather than clobbering it. ``decided`` is the
    in-process run, where :func:`plan_export` just decided this row against the
    same disk: deciding it again would re-read both sides of every text row and
    re-render every combined one for an answer that cannot have changed.
    """
    if not decided:
        _decide(row)
    if row.status in ("identical", "missing-source"):
        return row.status
    if not apply:
        return row.status
    dst = Path(row.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if row.combined:
        # `_published` is the comparison, so it is also the write: text mode
        # translates LF to CRLF on Windows, and then no combined row with a
        # newline in it (every variants sidecar) ever compares identical --
        # it republishes every run and reverts as drifted.
        dst.write_bytes(_published(row))
    elif row.render:
        (_render_mask if row.kind == "mask" else _render)(row, dst)
    else:
        shutil.copy2(row.src, dst)
    row.status = "created" if row.status == "would-create" else "overwrote"
    return row.status


def publish(
    paths: ExportPaths,
    *,
    apply: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[ExportRow], ExportStats]:
    """Plan the export and, with ``apply``, perform it."""
    rows = plan_export(paths)
    return _run(rows, apply=apply, progress=progress, decided=True)


def _run(
    rows: list[ExportRow],
    *,
    apply: bool,
    progress: Callable[[int, int, str], None] | None = None,
    decided: bool = False,
) -> tuple[list[ExportRow], ExportStats]:
    stats = ExportStats(rows=len(rows))
    for i, row in enumerate(rows, 1):
        status = export_one(row, apply=apply, decided=decided)
        if status in ("created", "overwrote"):
            stats.by_kind[row.kind] += 1
            stats.combined += row.combined
            stats.excluded += row.excluded
            stats.capped += row.kind == "image" and bool(row.cap)
            stats.rendered += row.kind == "image" and row.render
            setattr(stats, status, getattr(stats, status) + 1)
        elif status.startswith("would-"):
            stats.by_kind[row.kind] += 1
            stats.combined += row.combined
            stats.excluded += row.excluded
            stats.capped += row.kind == "image" and bool(row.cap)
            stats.rendered += row.kind == "image" and row.render
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

ROW_FIELDS = (
    "rel",
    "kind",
    "src",
    "dst",
    "status",
    "before",
    "ocr",
    "text",
    "ref",
    "fit",
)
"""The text fields a report round-trips. ``excluded``, ``cap`` and ``render``
are read separately, being the fields that are not strings."""


def rows_from_report(report: Mapping[str, object]) -> list[ExportRow]:
    """The rows of an export report, as :class:`ExportRow` again."""
    raw = report.get("rows")
    return [
        ExportRow(
            **{k: str(r.get(k) or "") for k in ROW_FIELDS},
            excluded=bool(r.get("excluded")),
            cap=int(r.get("cap") or 0),
            render=bool(r.get("render")),
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
            dst.write_text(row.before, encoding="utf-8", newline="")
            row.status = "restored"
            stats.restored += 1

        if not row.status.startswith(("would-", "removed", "restored")):
            stats.skip(row.status)
    return rows, stats


# ---- the stage runner ----------------------------------------------------


def run_export(req: ExportRequest):
    """Publish the workspace under ``out``. Returns ``(rows, stats)``."""
    paths = ExportPaths(
        resized=resolve_path(req.dst),
        masks=resolve_path(req.masks),
        master=resolve_path(req.master),
        index=resolve_path(req.index),
        src=resolve_path(req.src),
        out=resolve_path(req.out),
        excluded=resolve_path(req.excluded_dir),
        ocr=resolve_path(req.ocr_dir) if req.combine_ocr else None,
        ocr_min_det=req.ocr_min_det,
        ocr_min_glyph=req.ocr_min_glyph,
        cap=req.resize_cap_tokens if req.resize_cap else 0,
        webp=req.webp,
        images=not req.sidecars_only,
    )
    if not paths.resized.is_dir():
        raise FileNotFoundError(
            f"nothing to export: {paths.resized} does not exist. "
            "Run the Resize stage first."
        )
    rows, stats = publish(paths, apply=req.apply, progress=make_progress(50))
    stale = stale_siblings(rows)
    path = write_stage_report(
        resolve_path(req.report_dir),
        {
            **stage_report_header(
                src=paths.src,
                dst=paths.resized,
                path_pattern=None,
                apply=req.apply,
            ),
            "out": str(paths.out),
            "excluded_dir": str(paths.excluded),
            "combine_ocr": req.combine_ocr,
            "ocr_dir": str(paths.ocr) if paths.ocr is not None else None,
            "ocr_min_det": req.ocr_min_det,
            "ocr_min_glyph": req.ocr_min_glyph,
            "resize_cap": paths.cap,
            "webp": paths.webp,
            "sidecars_only": req.sidecars_only,
            "stale": [str(p) for p in stale],
            "stats": stats.to_dict(),
            "rows": [r.to_dict() for r in rows],
        },
    )
    print(f"\nreport → {path}")
    combined = f", {stats.combined} with OCR attached" if req.combine_ocr else ""
    excluded = f", {stats.excluded} under _excluded/" if stats.excluded else ""
    capped = f", {stats.capped} downscaled to {paths.cap} tokens" if paths.cap else ""
    webp = f", {stats.rendered} re-encoded" if paths.webp else ""
    # Named, not deleted: Export removes nothing it did not write this run.
    for p in stale:
        print(f"  stale: {p} shares its stem with a published image — remove it")
    print_dry_run_footer(
        req.apply,
        f"published: {stats.created} created, "
        f"{stats.overwrote} overwritten{combined}{excluded}{capped}{webp}",
    )
    return rows, stats
