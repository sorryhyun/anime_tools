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
from anime_tools.stages.cli._args import (
    add_path_pattern_arg,
    add_report_dir_arg,
)
from anime_tools.stages.requests import OcrRequest

DEFAULT_REPORT_DIR = OcrRequest.report_dir


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


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_ocr

    try:
        run_ocr(OcrRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
