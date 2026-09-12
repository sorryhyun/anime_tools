"""Batch OCR: image → text detector → reader → a text sidecar in the OCR tree.

Walks the resized tree and writes what each picture says into ``{stem}.ocr.txt``
under :data:`anime_tools.workspace.OCR`, mirroring its layout: every recognized
line with box and confidence (:mod:`anime_tools.captions.ocr_sidecar`).

It reads no caption and writes no caption, so it sits outside the caption ladder
and invalidates no TE cache. The reader is an argument; what :func:`run_ocr`, the
stage runner, hands :func:`read_tree` is the one path the package has: the AnimeText
text-block detector (:mod:`anime_tools.ocr.animetext`, detect-only) with every
box read by the manga VL reader through :class:`anime_tools.ocr.reread.RereadEngine`,
plus, under ``--mask_dir``, the text mask's uncovered components. The PP-OCRv6
det/rec pair that preceded it was retired 2026-09-07 with no fallback. **Dry-run
is the default** (the caller passes ``apply``), and a dry run reports every line
it would have written.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from anime_tools._env import resolve_path
from anime_tools._progress import phase
from anime_tools._walk import walk_images
from anime_tools.captions.ocr_sidecar import OcrLine, write_ocr_for

from ._progress import make_progress
from ._report import write_stage_report
from .resize import resized_tree

if TYPE_CHECKING:
    from anime_tools.ocr import OcrEngine
    from anime_tools.stages.requests import OcrRequest


@dataclass
class OcrProposal:
    """One image's OCR and where its sidecar goes."""

    image: str = ""
    sidecar: str = ""
    lines: tuple[OcrLine, ...] = ()
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_row(self) -> dict[str, object]:
        """The report row, with ``lines`` expanded so a dry run's report *is*
        the sidecar it would write."""
        return {
            "image": self.image,
            "sidecar": self.sidecar,
            "lines": [line.to_dict() for line in self.lines],
            "status": self.status,
        }


@dataclass
class OcrStats:
    seen: int = 0
    with_text: int = 0
    lines: int = 0
    sidecars: int = 0
    skipped: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1


def number_lines(lines: Sequence[OcrLine]) -> tuple[OcrLine, ...]:
    """The lines, numbered from 1 in the order they arrived.

    The stage filters nothing: the ``min_chars`` / ``skip_en`` floors belong to
    the reader (:mod:`anime_tools.ocr._text`), which knows the pixels each line
    came from.
    """
    return tuple(
        OcrLine(seq=i, box=ln.box, score=ln.score, text=ln.text, det=ln.det)
        for i, ln in enumerate(lines, 1)
    )


def read_tree(
    *,
    resized_dir: Path,
    ocr_dir: Path,
    read_fn: Callable[[Path], list[OcrLine]],
    read_iter_fn: Callable[[list[Path]], Iterable[list[OcrLine]]] | None = None,
    path_pattern: str | None = None,
    apply: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[OcrProposal], OcrStats]:
    """Read every image in the resized tree and (with ``apply``) write its sidecar.

    Every image is a candidate; no caption is consulted. The write is
    unconditional, because
    :func:`~anime_tools.captions.ocr_sidecar.write_ocr` deletes the sidecar when
    a pass finds nothing — so a re-run over re-cropped pixels drops the record
    of text no longer in the image.

    ``read_iter_fn`` reads the *whole* run, yielding one result per image as it
    lands; ``read_fn`` is the one-image fallback, and the two must answer the same
    thing for the same image. The batching is entirely the reader's
    (:meth:`~anime_tools.ocr.OcrEngine.read_iter`) — handing it the run in slices
    would only starve its prefetch, and there is nothing to gain by it, since the
    loop below already writes and reports per image.
    """
    stats = OcrStats()
    rows: list[OcrProposal] = []

    images = walk_images(resized_dir, recursive=True, pattern=path_pattern)
    stats.seen = len(images)
    stream = (
        read_iter_fn(images)
        if read_iter_fn is not None
        else (read_fn(p) for p in images)
    )

    for index, (image_path, raw) in enumerate(zip(images, stream, strict=True), 1):
        rel = image_path.relative_to(resized_dir)
        if progress is not None:
            progress(index, len(images), str(rel))

        lines = number_lines(raw)
        stats.lines += len(lines)
        if lines:
            stats.with_text += 1
        else:
            stats.skip("no-text")

        proposal = OcrProposal(
            image=str(rel),
            sidecar=str(rel.with_suffix(".ocr.txt")),
            lines=lines,
            status="ok" if lines else "no-text",
        )
        rows.append(proposal)

        if apply:
            write_ocr_for(ocr_dir, rel.with_suffix(".txt"), lines)
            stats.sidecars += 1

    return rows, stats


def _vl_engine(
    req: OcrRequest, engine: OcrEngine, resized_dir: Path, device: str
) -> OcrEngine:
    """The engine's boxes read by the manga VL reader, the text mask's uncovered
    components read too when ``--mask_dir`` names one
    (:class:`anime_tools.ocr.reread.RereadEngine`).

    Takes the device the runner resolved rather than probing again: the two
    models cannot land on different ones.
    """
    from anime_tools.ocr.reread import RereadEngine
    from anime_tools.ocr.sfx import SfxReader

    print(f"Loading the manga VL reader ({device})...", flush=True)
    with phase("load vl reader"):
        reader = SfxReader.load(device=device, batch_size=req.vl_batch_size)
    mask_dir = resolve_path(req.mask_dir) if req.mask_dir else None
    if mask_dir is not None and not mask_dir.is_dir():
        raise FileNotFoundError(f"--mask_dir {mask_dir} is not a directory")
    return RereadEngine(
        engine=engine,
        read_boxes=reader.read_boxes_scored,
        resized_dir=resized_dir,
        masks=mask_dir,
        comp_min_side=req.comp_min_side,
        comp_max=req.comp_max,
        min_chars=req.min_chars,
        skip_en=req.skip_en,
        min_det=req.min_det,
        min_score=req.min_score,
        strip_symbols=req.strip_symbols,
    )


def run_ocr(req: OcrRequest):
    """Read the text in every resized image and (with ``apply``) write the
    ``{stem}.ocr.txt`` sidecars. Returns ``(rows, stats)``.

    One path: the AnimeText detector boxes every page (detect-only), the manga
    VL reader reads every box, and what came back is the sidecar. Both run on
    torch, so both take the one device this run resolved.
    """
    resized_dir = resized_tree(req.dst)
    ocr_dir = resolve_path(req.ocr_dir)
    report_dir = resolve_path(req.report_dir)

    # Deferred: torch is the heaviest thing either model touches.
    from anime_tools._device import resolve_device
    from anime_tools.ocr import load_ocr

    device = resolve_device(req.device)
    print(f"Loading the AnimeText detector ({device})...", flush=True)
    with phase("load ocr"):
        engine = load_ocr(
            device=device,
            min_box_px=req.min_box_px,
            max_boxes=req.max_boxes,
            det_conf=req.det_conf,
        )
    engine = _vl_engine(req, engine, resized_dir, device)

    rows, stats = read_tree(
        resized_dir=resized_dir,
        ocr_dir=ocr_dir,
        read_fn=engine.read,
        read_iter_fn=engine.read_iter,
        path_pattern=req.path_pattern,
        apply=req.apply,
        progress=make_progress(25, first=True),
    )

    report_path = write_stage_report(
        report_dir,
        {
            "min_chars": req.min_chars,
            "skip_en": bool(req.skip_en),
            "strip_symbols": bool(req.strip_symbols),
            "min_det": req.min_det,
            "min_score": req.min_score,
            "min_box_px": req.min_box_px,
            "max_boxes": req.max_boxes,
            "det_conf": req.det_conf,
            "vl_batch_size": req.vl_batch_size,
            "mask_dir": str(resolve_path(req.mask_dir)) if req.mask_dir else None,
            "comp_min_side": req.comp_min_side,
            "comp_max": req.comp_max,
            "applied": bool(req.apply),
            "apply": bool(req.apply),
            "dst": str(resized_dir),
            "ocr_dir": str(ocr_dir),
            "path_pattern": req.path_pattern,
            "stats": {
                "seen": stats.seen,
                "with_text": stats.with_text,
                "lines": stats.lines,
                "sidecars": stats.sidecars,
                "skipped": dict(stats.skipped),
            },
            "rows": [r.to_row() for r in rows],
        },
    )

    print(
        f"\nseen={stats.seen} with_text={stats.with_text} "
        f"lines={stats.lines} sidecars={stats.sidecars}"
    )
    for reason, count in stats.skipped.most_common():
        print(f"  skip:{reason} {count}")
    print(f"report: {report_path}")
    if req.apply:
        print(f"sidecars: {ocr_dir}")
    else:
        print("\nDry run — no sidecars written. Re-run with --apply to write.")
    return rows, stats
