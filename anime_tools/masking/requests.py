"""The three mask stages as request objects — the surface the CLIs, the GUI and
the trainer share (``docs/api_first_plan.md``).

Torch-free: run one through :func:`anime_tools.masking.run_sam_masks` /
:func:`run_mit_masks` / :func:`run_merge_masks`, which import the models, or
hand ``to_argv()`` to a subprocess. Every field is an argparse ``dest`` of the
matching CLI, and the defaults are the CLI's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anime_tools import workspace as WS
from anime_tools._request import FLAG, POSITIONAL, READ, WRITE, Request
from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT, DEFAULT_SUBJECT_PROMPT_EMBED
from anime_tools.masking._sam3 import SUBJECT_PROMPT, prompt_list

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


def prompts_flag(prompts: tuple[str, ...]) -> str:
    """The inverse of :func:`prompt_list`: no prompts is spelled ``none``, because a
    blank flag would read back as the default."""
    return ",".join(prompts) if prompts else "none"


def _prompts(default: tuple[str, ...], **meta) -> tuple[str, ...]:
    return field(
        default=default, metadata={READ: prompt_list, WRITE: prompts_flag, **meta}
    )


@dataclass(frozen=True, kw_only=True)
class MaskWalkRequest(Request):
    """What both generators share: the walk ``_masks.mask_run`` reads by attribute."""

    image_dir: str
    mask_dir: str
    """This generator's own tree, never the merged ``masks`` root: both name a mask
    ``{stem}_mask.png`` at the same relative path."""
    force: bool = False
    """Regenerate masks that already exist."""
    workers: int = 4
    recursive: bool = False
    path_pattern: str | None = None
    device: str | None = None
    """``cuda`` / ``cpu``; ``None`` resolves at run time (``_device.resolve_device``)."""


@dataclass(frozen=True, kw_only=True)
class SamMaskRequest(MaskWalkRequest):
    """SAM3 subject masks (``python -m anime_tools.masking.cli.generate_masks``).

    ``prompts`` names what is masked OUT; ``focus_prompts`` what is kept, everything
    else masked out. Give both and the focus region survives minus the ignore regions.
    """

    mask_dir: str = WS.MASKS_SAM
    prompts: tuple[str, ...] = _prompts(())
    focus_prompts: tuple[str, ...] = _prompts((SUBJECT_PROMPT,))
    prompt_embed: str = field(
        default=DEFAULT_SUBJECT_PROMPT_EMBED, metadata={FLAG: "--prompt_embed"}
    )
    """The learned soft prompt standing in for the subject phrase; ``none`` for text."""
    threshold: float = 0.5
    dilate: int = 5
    checkpoint: str = DEFAULT_SAM3_CHECKPOINT
    batch_size: int = 1

    def __post_init__(self) -> None:
        if not self.prompts and not self.focus_prompts:
            raise ValueError("nothing to mask: pass --prompts and/or --focus-prompts")


@dataclass(frozen=True, kw_only=True)
class MitMaskRequest(MaskWalkRequest):
    """Text masks (``python -m anime_tools.masking.cli.generate_masks_mit``): SAM3
    prompts (``use_sam``), the UNet++ stroke segmenter (``use_mit``), or both, unioned
    before one dilation. Both off is refused."""

    mask_dir: str = WS.MASKS_MIT
    use_sam: bool = False
    sam_prompts: tuple[str, ...] = _prompts(DEFAULT_SAM_PROMPTS)
    sam_threshold: float = 0.5
    checkpoint: str = DEFAULT_SAM3_CHECKPOINT
    use_mit: bool = True
    model_path: str | None = None
    text_threshold: float = 0.8
    ctd_gate: bool = True
    """Keep only UNet++ components overlapping a comictextdetector text block."""
    dilate: int = 3

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
    """Union the generators' trees into the ``masks`` root
    (``python -m anime_tools.masking.cli.merge_masks``). A missing input is skipped."""

    mask_dirs: tuple[str, ...] = field(
        default=(WS.MASKS_SAM, WS.MASKS_MIT),
        metadata={POSITIONAL: True, READ: tuple, WRITE: list},
    )
    output_dir: str = WS.MASKS
