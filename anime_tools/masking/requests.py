"""The three mask stages as request objects — the surface the CLIs, the GUI and
the trainer share.

Torch-free: run one through :func:`anime_tools.masking.run_sam_masks` /
:func:`run_mit_masks` / :func:`run_merge_masks`, which import the models, or
hand ``to_argv()`` to a subprocess. Every field is a flag of the matching CLI,
whose parser is generated from the class (:meth:`Request.parser`): the help,
the default and the drawer a flag sits in are written here, once. Flags are
hyphenated (``--image-dir``), the masking CLIs' canonical form, and take the
underscore spelling as an alias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anime_tools import workspace as WS
from anime_tools._device import DEVICE_HELP
from anime_tools._request import HELP, POSITIONAL, READ, WRITE, Request, arg
from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT, DEFAULT_SUBJECT_PROMPT_EMBED
from anime_tools.masking._sam3 import (
    CHECKPOINT_HELP,
    PROMPT_EMBED_HELP,
    SUBJECT_PROMPT,
    prompt_list,
)

__all__ = [
    "DEFAULT_SAM_PROMPTS",
    "MaskWalkRequest",
    "MergeMasksRequest",
    "MitMaskRequest",
    "SamMaskRequest",
    "prompts_flag",
]

DEFAULT_SAM_PROMPTS = ("speech bubble",)
"""What the text stage's ``--use-sam`` means by *text* until told otherwise: a balloon
is a closed shape, and its interior is as untrainable as the lettering in it."""

WALK_HELP = (
    "Walk subfolders under --image-dir. Mask output mirrors the source "
    "subdir structure under --mask-dir."
)
PATTERN_HELP = (
    "fnmatch glob (| to OR-combine) on each image's path relative to "
    "--image-dir, restricting which images get masked. Same semantics "
    "as the training path_pattern."
)


def prompts_flag(prompts: tuple[str, ...]) -> str:
    """The inverse of :func:`prompt_list`: no prompts is spelled ``none``, because a
    blank flag would read back as the default."""
    return ",".join(prompts) if prompts else "none"


def _prompts(default: tuple[str, ...], **meta) -> tuple[str, ...]:
    return field(
        default=default, metadata={READ: prompt_list, WRITE: prompts_flag, **meta}
    )


def _mask_dir(default: str) -> str:
    """This generator's own tree, never the merged ``masks`` root: both name a mask
    ``{stem}_mask.png`` at the same relative path, so a shared directory would have
    the second run overwrite the first."""
    return arg(
        default,
        help=f"Output mask directory for this generator alone (default: {default}); "
        "`merge_masks` unions it with the other's into the masks root",
    )


@dataclass(frozen=True, kw_only=True)
class MaskWalkRequest(Request):
    """What both generators share: the walk ``_masks.mask_run`` reads by attribute."""

    image_dir: str = arg(help="Image directory")
    mask_dir: str
    """This generator's own tree, never the merged ``masks`` root: both name a mask
    ``{stem}_mask.png`` at the same relative path."""
    force: bool = arg(False, help="Regenerate existing masks")
    workers: int = arg(4, help="I/O workers for loading/saving (default: 4)")
    recursive: bool = arg(False, help=WALK_HELP)
    path_pattern: str | None = arg(None, help=PATTERN_HELP)
    device: str | None = arg(None, help=DEVICE_HELP)
    """``None`` resolves at run time (``_device.resolve_device``)."""


@dataclass(frozen=True, kw_only=True)
class SamMaskRequest(MaskWalkRequest):
    """SAM3 subject masks, written to ``workspace/masks_sam/``.

    ``--prompts`` names what is masked OUT (ignored in the loss); ``--focus-prompts``
    names what is kept, everything else masked out. Give both and the focus region
    survives minus the ignore regions. The subject prompt is served by a learned soft
    prompt by default (``--prompt_embed``); pass ``none`` for the plain text prompt.
    """

    mask_dir: str = _mask_dir(WS.MASKS_SAM)
    prompts: tuple[str, ...] = _prompts(
        (),
        help="Comma-separated SAM3 text prompts to mask OUT — these regions are "
        "ignored in the loss (e.g. `speech bubble,text`)",
    )
    focus_prompts: tuple[str, ...] = _prompts(
        (SUBJECT_PROMPT,),
        help="Comma-separated prompts to keep ONLY: everything outside them is masked "
        f"out. Default `{SUBJECT_PROMPT}` (the subject), so a bare run isolates the "
        "subject from her background; pass `none` to keep nothing in and use "
        "--prompts alone",
    )
    prompt_embed: str = arg(
        DEFAULT_SUBJECT_PROMPT_EMBED, flag="--prompt_embed", help=PROMPT_EMBED_HELP
    )
    """The learned soft prompt standing in for the subject phrase; ``none`` for text.
    Spelled with an underscore like the detection stages', so ⚙ Settings can fill
    all three from one value."""
    threshold: float = arg(
        0.5, help="SAM3 confidence floor for a detection (default: 0.5)"
    )
    dilate: int = arg(5, help="Mask dilation in pixels, 0 = off (default: 5)")
    checkpoint: str = arg(DEFAULT_SAM3_CHECKPOINT, help=CHECKPOINT_HELP)
    batch_size: int = arg(1, help="Images to process in parallel (default: 1)")

    def __post_init__(self) -> None:
        if not self.prompts and not self.focus_prompts:
            raise ValueError("nothing to mask: pass --prompts and/or --focus-prompts")


SAM_DRAWER = "SAM3 prompts"
MIT_DRAWER = "MIT text segmentation"


@dataclass(frozen=True, kw_only=True)
class MitMaskRequest(MaskWalkRequest):
    """Text masks, written to ``workspace/masks_mit/``: SAM3 prompts, a per-stroke
    UNet++, or both.

    ``--use-sam`` grounds SAM3 on ``--sam-prompts``; ``--use-mit`` runs the UNet++ text
    segmenter behind comictextdetector's text-block gate (``--ctd-gate``). A balloon is
    a shape and a letter is a stroke, so neither switch subsumes the other and both off
    is the one argv the stage refuses. The two are unioned before the single dilation.
    """

    mask_dir: str = _mask_dir(WS.MASKS_MIT)
    use_sam: bool = arg(
        False,
        gate="use_sam",
        group=SAM_DRAWER,
        help="Ground SAM3 on --sam-prompts and mask what it finds. Off by default: it "
        "is a second set of weights to load, and it answers a different question "
        "than the segmenter below — turn it on for balloons, which are a shape "
        "rather than a stroke",
    )
    sam_prompts: tuple[str, ...] = _prompts(
        DEFAULT_SAM_PROMPTS,
        gate="use_sam",
        help="Comma-separated SAM3 text prompts for the regions to mask OUT (default "
        f"`{prompts_flag(DEFAULT_SAM_PROMPTS)}`; e.g. `speech bubble,sign,"
        "watermark`). Same polarity as `generate_masks --prompts`: everything named "
        "here is ignored in the loss",
    )
    sam_threshold: float = arg(
        0.5, gate="use_sam", help="SAM3 confidence floor for a detection (default: 0.5)"
    )
    checkpoint: str = arg(DEFAULT_SAM3_CHECKPOINT, gate="use_sam", help=CHECKPOINT_HELP)
    use_mit: bool = arg(
        True,
        gate="use_mit",
        group=MIT_DRAWER,
        help="Run the UNet++ text segmenter — the stroke-accurate half, and the only "
        "one that finds lettering outside a balloon",
    )
    model_path: str | None = arg(
        None,
        gate="use_mit",
        help="Path to model.pth (downloads from HuggingFace if not specified)",
    )
    text_threshold: float = arg(
        0.8, gate="use_mit", help="Text segmentation threshold (default: 0.8)"
    )
    ctd_gate: bool = arg(
        True,
        gate="use_mit",
        help="keep only mask components overlapping a comictextdetector text block "
        "— drops UNet++ false positives on halos/decorative line art (--no-ctd-gate "
        "= raw UNet++ masks, restores pre-2026-07 behavior). The net is the download "
        "catalog's `ctd_onnx` row; a missing one warns and leaves the masks ungated",
    )
    """Keep only UNet++ components overlapping a comictextdetector text block."""
    dilate: int = arg(
        3, help="Mask dilation in pixels, applied once to the union (default: 3)"
    )

    @property
    def active_sam_prompts(self) -> tuple[str, ...]:
        """The prompts SAM3 is grounded on: none while the drawer is shut."""
        return self.sam_prompts if self.use_sam else ()

    def __post_init__(self) -> None:
        if self.use_sam and not self.sam_prompts:
            raise ValueError(
                "--use-sam with no --sam-prompts: nothing for SAM3 to ground on"
            )
        if not self.use_mit and not self.active_sam_prompts:
            raise ValueError("nothing to detect: pass --use-mit and/or --use-sam")


@dataclass(frozen=True, kw_only=True)
class MergeMasksRequest(Request):
    """Merge masks from multiple sources by taking the pixel-wise minimum (union of
    masked regions).

    Keys merges by ``(rel_dir, name)``, so masks at the same relative path across
    inputs collide; the nested layout is preserved under ``--output-dir``. A missing
    input directory is skipped, not an error — running only one generator is a valid
    half of this.
    """

    mask_dirs: tuple[str, ...] = field(
        default=(WS.MASKS_SAM, WS.MASKS_MIT),
        metadata={
            POSITIONAL: True,
            READ: tuple,
            WRITE: list,
            HELP: "Input mask directories to merge (default: the two generators' "
            f"own trees, {WS.MASKS_SAM} {WS.MASKS_MIT})",
        },
    )
    output_dir: str = arg(
        WS.MASKS, help=f"Output directory for merged masks (default: {WS.MASKS})"
    )
