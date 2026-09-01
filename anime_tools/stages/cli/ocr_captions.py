"""Read the text in each image with PP-OCRv6 and record what it says.

Walks the resized tree, detects and recognizes every text line, and writes
``{stem}.ocr.txt`` into the OCR tree, mirroring the resized layout. No caption is
read or written, and no TE re-encode is needed afterwards.

Dry-run by default: a dry run emits ``report.json`` carrying every line it would
have written, and ``--apply`` writes the sidecars and nothing else.
"""

from __future__ import annotations

import argparse

from anime_tools import workspace as WS
from anime_tools._device import add_device_arg
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
        "--min_chars",
        "--min-chars",
        dest="min_chars",
        type=int,
        default=3,
        help="Drop a line shorter than this many non-space characters, after "
        "the CJK join — one or two glyphs is a misread screentone far more "
        "often than it is a word",
    )
    p.add_argument(
        "--keep_en",
        "--keep-en",
        dest="skip_en",
        action="store_false",
        help="Keep ASCII-only lines. Dropped by default: on a scanned comic they "
        "are the page number, the URL and the romaji sfx, never the dialogue",
    )
    p.add_argument(
        "--no_join_cjk",
        "--no-join-cjk",
        dest="join_cjk",
        action="store_false",
        help="Record each CJK box on its own line. Joined by default: a balloon "
        "of vertical Japanese is detected as one box per column, and the columns "
        "are one sentence",
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
        default=1440,
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

    # Deferred: onnxruntime is the heaviest thing this CLI touches, and `--help`
    # should not pay for it.
    from anime_tools.ocr import OcrWeightsMissing, load_ocr, resolve_onnx_device

    # Not `_device.resolve_device`: its torch probe would cost this run 1.8x for an
    # answer onnxruntime already has (:func:`~anime_tools.ocr.resolve_onnx_device`).
    device = resolve_onnx_device(args.device)
    print(f"Loading PP-OCRv6 ({device})...", flush=True)
    try:
        engine = load_ocr(
            device=device,
            min_score=args.min_score,
            min_chars=args.min_chars,
            skip_en=args.skip_en,
            join_cjk=args.join_cjk,
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
        read_iter_fn=engine.read_iter,
        path_pattern=args.path_pattern,
        apply=args.apply,
        progress=make_progress(25, first=True),
    )

    report_path = write_stage_report(
        report_dir,
        {
            "min_score": args.min_score,
            "min_chars": args.min_chars,
            "skip_en": bool(args.skip_en),
            "join_cjk": bool(args.join_cjk),
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
