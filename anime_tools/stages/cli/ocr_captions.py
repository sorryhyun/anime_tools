"""Read the text in each image with PP-OCRv6 and record what it says.

Thin CLI over ``anime_tools.stages.ocr``: walks the resized tree, detects and
recognizes every text line, and writes ``{stem}.ocr.txt`` into the OCR tree,
mirroring the resized tree's layout. **Dry-run is the default** — a dry run
emits ``report.json`` carrying every line it would have written, so the sidecars
can be eyeballed before they exist.

**No caption is read or written.** This stage used to also append a Danbooru
script tag (``english text`` / ``chinese text``) inferred from the language of
what it recognized, and then, for a while, to keep a ``--lang`` allowlist that
dropped a line whose guessed script was not asked for. Both are gone, for one
reason: the language was guessed back off the characters, and most of what it
decided rested on two-character fragments — see ``anime_tools/stages/ocr.py``.
So there is no ``--apply`` gate to be careful about here, no ``--from_report``
replay, and no TE re-encode afterwards: the only thing an ``--apply`` writes is
the sidecar tree.

The weights have no flag. Both halves are read from the download catalog's
``ppocr_det`` / ``ppocr_rec`` rows, for the reason the MIT stage's ``--ctd-gate``
net has none: a path you could point elsewhere is a Download button aimed at a
directory the loader does not read.

    python -m anime_tools.stages.cli.ocr_captions
    python -m anime_tools.stages.cli.ocr_captions --apply --min_score 0.7
"""

from __future__ import annotations

import argparse

from anime_tools import workspace as WS
from anime_tools._device import add_device_arg, resolve_device
from anime_tools._env import resolve_path
from anime_tools.stages.cli._args import (
    add_path_pattern_arg,
    add_report_dir_arg,
    make_progress,
)
from anime_tools.stages.cli._report import write_stage_report
from anime_tools.stages.ocr import run_ocr

DEFAULT_REPORT_DIR = f"{WS.REPORTS}/ocr"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dst", default=WS.RESIZED, help="Resized images")
    p.add_argument(
        "--ocr_dir",
        "--ocr-dir",
        dest="ocr_dir",
        default=WS.OCR,
        help=f"Where the {{stem}}.ocr.txt sidecars land, mirroring --dst "
        f"(default: {WS.OCR})",
    )
    add_path_pattern_arg(
        p, help="fnmatch glob (| to OR-combine) on the path relative to --dst"
    )
    p.add_argument(
        "--min_score",
        "--min-score",
        dest="min_score",
        type=float,
        default=0.6,
        help="Drop a recognized line below this mean per-character confidence (0-1)",
    )
    p.add_argument(
        "--min_box_px",
        "--min-box-px",
        dest="min_box_px",
        type=int,
        default=12,
        help="Ignore a detected box whose longest side is under this many "
        "pixels — screentone and hatching, not text",
    )
    p.add_argument(
        "--max_boxes",
        "--max-boxes",
        dest="max_boxes",
        type=int,
        default=64,
        help="Recognize at most this many boxes per image, largest first, so "
        "one misread texture cannot cost a thousand crops",
    )
    p.add_argument(
        "--det_limit_side",
        "--det-limit-side",
        dest="det_limit_side",
        type=int,
        default=960,
        help="Longest side the detector sees; larger finds smaller text and "
        "costs quadratically",
    )
    p.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=8,
        help="Line crops recognized per forward pass",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write the sidecars (default: dry run). Touches no caption",
    )
    add_report_dir_arg(p, DEFAULT_REPORT_DIR)
    add_device_arg(p)
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> None:
    args = parse_args()
    resized_dir = resolve_path(args.dst)
    ocr_dir = resolve_path(args.ocr_dir)
    if not resized_dir.exists():
        raise SystemExit(
            f"resized dir not found: {resized_dir} — run `make preprocess-resize` first"
        )

    report_dir = resolve_path(args.report_dir)

    # Not at import time: onnxruntime is the heaviest thing this CLI touches and
    # `--help` should not pay for it.
    from anime_tools.ocr import OcrWeightsMissing, load_ocr

    device = resolve_device(args.device)
    print(f"Loading PP-OCRv6 ({device})...", flush=True)
    try:
        engine = load_ocr(
            device=device,
            min_score=args.min_score,
            min_box_px=args.min_box_px,
            max_boxes=args.max_boxes,
            limit_side=args.det_limit_side,
            batch_size=args.batch_size,
        )
    except OcrWeightsMissing as exc:
        raise SystemExit(str(exc)) from exc

    rows, stats = run_ocr(
        resized_dir=resized_dir,
        ocr_dir=ocr_dir,
        read_fn=engine.read,
        read_many_fn=engine.read_many,
        path_pattern=args.path_pattern,
        apply=args.apply,
        progress=make_progress(25, first=True),
    )

    report_path = write_stage_report(
        report_dir,
        {
            "min_score": args.min_score,
            "min_box_px": args.min_box_px,
            "max_boxes": args.max_boxes,
            "det_limit_side": args.det_limit_side,
            "applied": bool(args.apply),
            "apply": bool(args.apply),
            "dst": str(resized_dir),
            "ocr_dir": str(ocr_dir),
            "path_pattern": args.path_pattern,
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
    if args.apply:
        print(f"sidecars: {ocr_dir}")
    else:
        print("\nDry run — no sidecars written. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
