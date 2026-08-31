#!/usr/bin/env python3
"""Resize the caption master into the bucket-resolution tree every stage reads.

``autotag`` / ``position`` / ``correct`` all walk ``--dst`` (the resized tree),
because the tagger should see the pixels training sees. So an image that only
exists under ``image_dataset/`` is invisible to them — it is listed in the GUI
sidebar, but a run scoped to it matches nothing and writes nothing. This stage
is what puts it there.

Free-fit geometry: each image lands in the ``--target_res`` tier that resizes it
the least, keeping its native aspect inside that tier's token band, so the crop
is under one 16px patch. Identical to the trainer's ``make preprocess-resize``
(same tier, same solver, same ``anima_resize_*`` PNG keys), so whichever side
runs first, the other one skips.

Always writes — there is no dry run, because the pass is idempotent: an output
already at its target bucket is skipped without a re-decode, so a re-run is
near-free and only a tier change re-resizes.

    python -m anime_tools.stages.cli.resize_images
    python -m anime_tools.stages.cli.resize_images --target_res 1024 1536
"""

from __future__ import annotations

import argparse
import json

from anime_tools._env import resolve_path
from anime_tools.buckets import ALLOWED_TARGET_RES
from anime_tools.stages.resize import (
    CROP_ANCHORS,
    DEFAULT_CROP_ANCHOR,
    DEFAULT_MIN_PIXELS,
    ResizeOptions,
    run_resize_images,
)

DEFAULT_REPORT_DIR = "post_image_dataset/captions/resize"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="image_dataset", help="Caption master dir")
    p.add_argument(
        "--dst", default="post_image_dataset/resized", help="Resized image dir"
    )
    p.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help="fnmatch glob (| to OR-combine) on the path relative to --src",
    )
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
            "by default: the derived caption is written by the correct stage, "
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
    p.add_argument(
        "--report_dir",
        "--report-dir",
        dest="report_dir",
        default=DEFAULT_REPORT_DIR,
        help=f"Where report.json lands (default: {DEFAULT_REPORT_DIR})",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.target_res:
        bad = [e for e in args.target_res if e not in ALLOWED_TARGET_RES]
        if bad:
            raise SystemExit(
                f"--target_res {bad} not in allowed tiers {list(ALLOWED_TARGET_RES)}"
            )

    src = resolve_path(args.src)
    dst = resolve_path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"source dir not found: {src}")

    options = ResizeOptions.build(
        target_res=args.target_res,
        crop_anchor=args.resize_crop_anchor,
        crop_margins=args.resize_crop_margins,
        max_ratio=args.freefit_max_ratio,
    )

    def progress(index: int, total: int, detail: str) -> None:
        print(f"  [{index}/{total}] {detail}", flush=True)

    stats = run_resize_images(
        src=src,
        dst=dst,
        options=options,
        path_pattern=str(args.path_pattern or "*"),
        recursive=bool(args.recursive),
        min_pixels=int(args.min_pixels),
        copy_captions=bool(args.copy_captions),
        overwrite=bool(args.overwrite),
        workers=int(args.workers),
        progress=progress,
    )

    report_dir = resolve_path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "src": str(src),
        "dst": str(dst),
        "path_pattern": str(args.path_pattern or "*"),
        "target_res": list(options.target_res),
        "crop_anchor": options.crop_anchor,
        "crop_margins": list(options.crop_margins),
        "max_ratio": options.max_ratio,
        "min_pixels": int(args.min_pixels),
        "overwrite": bool(args.overwrite),
        "stats": {
            "seen": stats.seen,
            "written": stats.written,
            "skipped_current": stats.skipped_current,
            "skipped_small": stats.skipped_small,
            "failed": stats.failed,
        },
        "buckets": dict(sorted(stats.buckets.items())),
        "failures": stats.failures,
    }
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Resized: {stats.written} written, "
        f"{stats.skipped_current} already current, "
        f"{stats.skipped_small} below {args.min_pixels:,} px, "
        f"{stats.failed} failed ({stats.seen} images seen)"
    )
    for line in stats.failures:
        print(f"  fail: {line}")
    if stats.buckets:
        print("Bucket distribution:")
        for reso, count in sorted(stats.buckets.items()):
            w, h = (int(v) for v in reso.split("x"))
            print(f"  {reso:>10}: {count:>3d} images  ({(w // 16) * (h // 16)} tokens)")


if __name__ == "__main__":
    main()
