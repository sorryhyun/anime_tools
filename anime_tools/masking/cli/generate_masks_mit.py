"""Text masks, written to ``workspace/masks_mit/``: SAM3 prompts, a per-stroke UNet++, or
both.

``--use-sam`` grounds SAM3 on ``--sam-prompts``; ``--use-mit`` runs the UNet++ text
segmenter behind comictextdetector's text-block gate (``--ctd-gate``). A balloon is a
shape and a letter is a stroke, so neither switch subsumes the other and both off is the
one argv the stage refuses. The two are unioned before the single dilation.
"""

from __future__ import annotations

import argparse

from anime_tools import workspace as WS
from anime_tools._device import add_device_arg
from anime_tools.masking._masks import (
    add_force_arg,
    add_mask_dir_args,
    add_walk_args,
    add_workers_arg,
    gated_group,
)
from anime_tools.masking._sam3 import add_checkpoint_arg, prompt_list
from anime_tools.masking.requests import MitMaskRequest, prompts_flag

__all__ = ["DEFAULT_SAM_PROMPTS", "build_parser", "detectors", "main", "prompt_list"]

DEFAULT_SAM_PROMPTS = prompts_flag(MitMaskRequest.sam_prompts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_mask_dir_args(parser, mask_default=WS.MASKS_MIT)

    sam = gated_group(
        parser,
        "SAM3 prompts",
        gate="use_sam",
        default=False,
        help="Ground SAM3 on --sam-prompts and mask what it finds. Off by "
        "default: it is a second set of weights to load, and it answers a "
        "different question than the segmenter below — turn it on for "
        "balloons, which are a shape rather than a stroke",
    )
    sam.add_argument(
        "--sam-prompts",
        dest="sam_prompts",
        type=str,
        default=DEFAULT_SAM_PROMPTS,
        help=f"Comma-separated SAM3 text prompts for the regions to mask OUT "
        f"(default `{DEFAULT_SAM_PROMPTS}`; e.g. `speech bubble,sign,"
        f"watermark`). Same polarity as `generate_masks --prompts`: everything "
        f"named here is ignored in the loss",
    )
    sam.add_argument(
        "--sam-threshold",
        dest="sam_threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence floor for a detection (default: 0.5)",
    )
    add_checkpoint_arg(sam)

    mit = gated_group(
        parser,
        "MIT text segmentation",
        gate="use_mit",
        default=True,
        help="Run the UNet++ text segmenter — the stroke-accurate half, and "
        "the only one that finds lettering outside a balloon",
    )
    mit.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model.pth (downloads from HuggingFace if not specified)",
    )
    mit.add_argument(
        "--text-threshold",
        type=float,
        default=0.8,
        help="Text segmentation threshold (default: %(default)s)",
    )
    mit.add_argument(
        "--ctd-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep only mask components overlapping a comictextdetector text "
            "block — drops UNet++ false positives on halos/decorative line art "
            "(--no-ctd-gate = raw UNet++ masks, restores pre-2026-07 behavior). "
            "The net is the download catalog's `ctd_onnx` row; a missing one "
            "warns and leaves the masks ungated"
        ),
    )

    add_force_arg(parser)
    add_device_arg(parser)
    parser.add_argument(
        "--dilate",
        type=int,
        default=3,
        help="Mask dilation in pixels, applied once to the union (default: %(default)s)",
    )
    add_workers_arg(parser)
    add_walk_args(parser)
    return parser


def request(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> MitMaskRequest:
    """The parsed flags as a request — or exit saying why not.

    Both drawers shut is a run that would load no model, walk the whole tree and write
    nothing — caught here, before the first weight is read.
    """
    try:
        return MitMaskRequest.from_namespace(args)
    except ValueError as e:
        parser.error(str(e))


def detectors(parser: argparse.ArgumentParser, args) -> tuple[bool, tuple[str, ...]]:
    """``(run the segmenter, the SAM3 prompts)``."""
    req = request(parser, args)
    return req.use_mit, req.active_sam_prompts


def main() -> None:
    from anime_tools.masking.mit import run_mit_masks

    parser = build_parser()
    run_mit_masks(request(parser, parser.parse_args()))


if __name__ == "__main__":
    main()
