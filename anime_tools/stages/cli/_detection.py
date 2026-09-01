"""The SAM3 detection flag block, and the options it builds.

``position_captions`` and ``audit_multiview`` run the *same* detector:
:func:`add_detection_args` declares the flags, :func:`detection_options` reads
the namespace back into the detection half of ``PositionCaptionOptions``.
Declaring one without the other is what lets them disagree.
"""

from __future__ import annotations

import argparse

from anime_tools.masking._sam3 import SUBJECT_PROMPT, add_prompt_embed_arg

# Flags a stage may leave out because it pins the value itself (the audit pins
# ``min_instances=2``). Read only when declared, so an omitted flag falls
# through to the dataclass default.
OPTIONAL_FLAGS = ("blank_crops", "min_instances", "strict_count")


def add_detection_args(
    p: argparse.ArgumentParser,
    *,
    score_threshold_help: str | None = None,
    part_prompts_help: str,
    blank_crops: bool = False,
    min_instances: bool = False,
    strict_count: bool = False,
    name_confidence: bool = False,
) -> None:
    """The ``detection`` argument group both SAM3 stages run under.

    The four booleans add a flag the *other* stage does not take, in the slot it
    occupies in that parser's field order — and therefore in the GUI form.
    """
    g = p.add_argument_group("detection")
    g.add_argument(
        "--prompt", default=SUBJECT_PROMPT, help="SAM3 text prompt for a subject"
    )
    # Declared in masking/_sam3.py beside --checkpoint; both are ⚙ Settings
    # stage defaults.
    add_prompt_embed_arg(g)
    g.add_argument(
        "--score_threshold", type=float, default=0.5, help=score_threshold_help
    )
    g.add_argument(
        "--retry_score_threshold",
        type=float,
        default=0.35,
        help="Retry threshold when detection undershoots the expected count. "
        "This is SAM3's own confidence floor, not a post-filter — see "
        "build_detect_fn",
    )
    g.add_argument(
        "--part_prompts",
        "--part-prompts",
        dest="part_prompts",
        default="",
        help=part_prompts_help,
    )
    g.add_argument(
        "--part_score_threshold",
        "--part-score-threshold",
        dest="part_score_threshold",
        type=float,
        default=0.5,
        help="Confidence floor for a body-part box (kept separate from the "
        "subject threshold — part prompts are the looser concept)",
    )
    g.add_argument(
        "--part_containment_threshold",
        "--part-containment-threshold",
        dest="part_containment_threshold",
        type=float,
        default=0.7,
        help="Drop a part box this nested inside an already-kept box. Unlike "
        "--containment_threshold this is safe to leave on: a part inside a "
        "subject is that subject's own body, never a second subject",
    )
    g.add_argument("--iou_threshold", type=float, default=0.65)
    g.add_argument(
        "--containment_threshold",
        "--containment-threshold",
        dest="containment_threshold",
        type=float,
        default=1.01,
        help="Suppress a box this nested inside a kept one (intersection over "
        "the smaller box). Off by default (>1.0 disables): a real second "
        "subject is as nested as a group box — enabling it cost 32 real "
        "subjects to save 12 group boxes",
    )
    g.add_argument(
        "--mask_containment_threshold",
        "--mask-containment-threshold",
        dest="mask_containment_threshold",
        type=float,
        default=0.8,
        help="Suppress a detection whose MASK is this nested inside a kept "
        "one. On by default, unlike its box counterpart: a second girl in "
        "front of the first nests identically by box but her mask is disjoint. "
        ">1.0 disables (the pre-2026-08-19 behaviour)",
    )
    g.add_argument(
        "--dedupe_fill_ratio",
        "--dedupe-fill-ratio",
        dest="dedupe_fill_ratio",
        type=float,
        default=2.0,
        help="Mask-quality tie-break inside an NMS-matched pair; 0 = off "
        "(score-only survivor). See docs/multiview_audit.md.",
    )
    g.add_argument(
        "--min_area_frac",
        "--min-area-frac",
        dest="min_area_frac",
        type=float,
        default=0.005,
        help="Drop detections smaller than this fraction of the image — an "
        "inset (a character on a phone screen) is not a bindable subject",
    )
    g.add_argument("--pad", type=float, default=0.06, help="bbox padding fraction")
    if blank_crops:
        g.add_argument(
            "--no_blank_crops",
            "--no-blank-crops",
            dest="blank_crops",
            action="store_false",
            help="Skip mask-blanking (probe B: this is what caused the hair-color misses)",
        )
    g.add_argument(
        "--row_tol",
        type=float,
        default=0.25,
        help="Minimum fractional overlap (of the narrower box extent) for two "
        "subjects to share a row — and a column, on magazine layouts where a "
        "full-height subject bridges a stack of panels",
    )
    if min_instances:
        g.add_argument("--min_instances", type=int, default=2)
    g.add_argument("--max_instances", type=int, default=8)
    if strict_count:
        g.add_argument(
            "--no_strict_count",
            "--no-strict-count",
            dest="strict_count",
            action="store_false",
            help="Propose clauses even when detection disagrees with the girls-count",
        )
    if name_confidence:
        g.add_argument("--name_confidence", type=float, default=0.5)


def detection_options(args: argparse.Namespace, **overrides) -> dict[str, object]:
    """The detection half of ``PositionCaptionOptions``, as keyword arguments.

    Extras go through ``**overrides`` (``detection_options(args,
    min_instances=2)``). Flags in :data:`OPTIONAL_FLAGS` are read only when the
    parser declared them.
    """
    opts: dict[str, object] = {
        "prompt": args.prompt,
        "score_threshold": args.score_threshold,
        "retry_score_threshold": args.retry_score_threshold,
        # Comma string on the CLI, tuple in options.
        "part_prompts": tuple(
            t.strip() for t in args.part_prompts.split(",") if t.strip()
        ),
        "part_score_threshold": args.part_score_threshold,
        "part_containment_threshold": args.part_containment_threshold,
        "iou_threshold": args.iou_threshold,
        "containment_threshold": args.containment_threshold,
        "mask_containment_threshold": args.mask_containment_threshold,
        "dedupe_fill_ratio": args.dedupe_fill_ratio,
        "min_area_frac": args.min_area_frac,
        "pad": args.pad,
        "row_tol": args.row_tol,
        "max_instances": args.max_instances,
    }
    for name in OPTIONAL_FLAGS:
        if hasattr(args, name):
            opts[name] = getattr(args, name)
    opts.update(overrides)
    return opts
