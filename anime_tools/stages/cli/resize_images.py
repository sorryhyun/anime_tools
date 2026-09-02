"""Resize the caption master into the bucket-resolution tree every stage reads.

Every other stage walks ``--dst``, so an image that exists only under
``image_dataset/`` is invisible to them. Each image lands in the ``--target_res``
tier that resizes it the least, keeping its native aspect inside that tier's
token band; the geometry matches the trainer's ``make preprocess-resize``, so
whichever side runs first, the other one skips.

Always writes; there is no dry run, and an image already at its target bucket is
skipped without a re-decode.
"""

from __future__ import annotations

import argparse

from anime_tools.buckets import ALLOWED_TARGET_RES
from anime_tools.stages.cli._args import (
    add_dataset_args,
    add_report_dir_arg,
)
from anime_tools.stages.requests import ResizeRequest
from anime_tools.stages.resize import (
    CROP_ANCHORS,
    DEFAULT_CROP_ANCHOR,
    DEFAULT_MIN_PIXELS,
)

DEFAULT_REPORT_DIR = ResizeRequest.report_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # The one stage whose glob matches against --src: it walks the master to
    # populate --dst.
    add_dataset_args(p, dst_help="Resized image dir", pattern_root="--src")
    p.add_argument(
        "--target_res",
        "--target-res",
        dest="target_res",
        type=int,
        nargs="+",
        default=None,
        metavar="EDGE",
        help=(
            "Free-fit tiers (allowed: "
            + " ".join(str(e) for e in ALLOWED_TARGET_RES)
            + "). Each image lands in the tier that resizes it the least. "
            "Default (unset) = a single 1024 tier. Must match the trainer's "
            "configured target_res or both sides keep re-resizing each other."
        ),
    )
    p.add_argument(
        "--min_pixels",
        "--min-pixels",
        dest="min_pixels",
        type=int,
        default=DEFAULT_MIN_PIXELS,
        help=(
            f"Skip images below this pixel count (default: {DEFAULT_MIN_PIXELS:,} "
            "= 0.5MP; 0 disables). Smaller images would be upscaled to fill a tier."
        ),
    )
    p.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Walk subfolders, mirroring the layout under --dst (default: on)",
    )
    p.add_argument(
        "--copy_captions",
        "--copy-captions",
        dest="copy_captions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also copy .txt / .caption sidecars next to the resized image. Off "
            "by default: the revised caption is written by the correct stage, "
            "which would otherwise be overwritten by the raw master."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-resize even images already at their target bucket",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes (default: 4; 1 runs inline)",
    )
    p.add_argument(
        "--resize_crop_anchor",
        "--resize-crop-anchor",
        dest="resize_crop_anchor",
        choices=tuple(CROP_ANCHORS),
        default=DEFAULT_CROP_ANCHOR,
        help="Anchor for the residual cover-crop (default: center)",
    )
    p.add_argument(
        "--resize_crop_margins",
        "--resize-crop-margins",
        dest="resize_crop_margins",
        nargs=4,
        type=float,
        default=None,
        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"),
        help="Percent margins cropped from the source before resize (default: 0s)",
    )
    p.add_argument(
        "--freefit_max_ratio",
        "--freefit-max-ratio",
        dest="freefit_max_ratio",
        type=float,
        default=4.0,
        help=(
            "Aspect-ratio clamp (default 4.0 = 1:4 / 4:1). Beyond-clamp images "
            "cover-crop to the limit; also keeps the token band solvable."
        ),
    )
    add_report_dir_arg(p, DEFAULT_REPORT_DIR)
    return p


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_resize

    try:
        run_resize(ResizeRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
