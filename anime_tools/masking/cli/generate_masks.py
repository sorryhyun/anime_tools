"""SAM3 subject masks, written to ``workspace/masks_sam/``.

``--prompts`` names what is masked OUT (ignored in the loss); ``--focus-prompts`` names
what is kept, everything else masked out. Give both and the focus region survives minus
the ignore regions. The subject prompt is served by a learned soft prompt by default
(``--prompt_embed``); pass ``none`` for the plain text prompt.
"""

import argparse

from anime_tools import workspace as WS
from anime_tools._device import add_device_arg
from anime_tools.masking._masks import (
    add_force_arg,
    add_mask_dir_args,
    add_walk_args,
    add_workers_arg,
)
from anime_tools.masking._sam3 import (
    SUBJECT_PROMPT,
    add_checkpoint_arg,
    add_prompt_embed_arg,
    prompt_list,
)
from anime_tools.masking.requests import SamMaskRequest

__all__ = ["build_parser", "main", "prompt_list"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_mask_dir_args(parser, mask_default=WS.MASKS_SAM)
    parser.add_argument(
        "--prompts",
        type=str,
        default="",
        help="Comma-separated SAM3 text prompts to mask OUT — these regions are "
        "ignored in the loss (e.g. `speech bubble,text`)",
    )
    parser.add_argument(
        "--focus-prompts",
        dest="focus_prompts",
        type=str,
        default=SUBJECT_PROMPT,
        help="Comma-separated prompts to keep ONLY: everything outside them is "
        f"masked out. Default `{SUBJECT_PROMPT}` (the subject), so a bare run "
        "isolates the subject from her background; pass `none` to keep nothing "
        "in and use --prompts alone",
    )
    add_prompt_embed_arg(parser)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence floor for a detection (default: 0.5)",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=5,
        help="Mask dilation in pixels, 0 = off (default: 5)",
    )
    add_force_arg(parser)
    add_checkpoint_arg(parser)
    add_device_arg(parser)
    add_workers_arg(parser, help="I/O workers for loading/saving (default: 4)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Images to process in parallel (default: 1)",
    )
    add_walk_args(parser)
    return parser


def main() -> None:
    from anime_tools.masking.sam import run_sam_masks

    parser = build_parser()
    try:
        req = SamMaskRequest.from_namespace(parser.parse_args())
    except ValueError as e:
        parser.error(str(e))
    run_sam_masks(req)


if __name__ == "__main__":
    main()
