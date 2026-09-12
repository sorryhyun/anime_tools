"""The two mask stages as request objects — the surface the CLIs, the GUI and
the trainer share.

Torch-free: run one through :func:`anime_tools.masking.run_sam_masks` /
:func:`run_merge_masks`, which import the models, or
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
from anime_tools._request import (
    HELP,
    KIND,
    METAVAR,
    NARGS,
    POSITIONAL,
    READ,
    WRITE,
    Request,
    arg,
)
from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT, DEFAULT_SUBJECT_PROMPT_EMBED
from anime_tools.masking._prompts import CHECKPOINT_HELP

__all__ = [
    "MASK_KINDS",
    "MASK_ROLES",
    "MaskPrompt",
    "MaskWalkRequest",
    "MergeMasksRequest",
    "SamMaskRequest",
]

MASK_ROLES = ("keep", "ignore")
"""What a mask prompt's region does to the loss: ``keep`` trains only inside the
union of every keep region, ``ignore`` masks its region out of that (or out of
the whole image, when nothing is kept)."""

MASK_KINDS = ("text", "soft")
"""How a mask prompt reaches SAM3: ``text`` through its text encoder, ``soft`` as a
learned prompt file (``.safetensors``) that stands in for the encoder's output."""

WALK_HELP = (
    "Walk subfolders under --image-dir. Mask output mirrors the source "
    "subdir structure under --mask-dir."
)
PATTERN_HELP = (
    "fnmatch glob (| to OR-combine) on each image's path relative to "
    "--image-dir, restricting which images get masked. Same semantics "
    "as the training path_pattern."
)


@dataclass(frozen=True)
class MaskPrompt:
    """One entry of the mask stage's list: a region SAM3 finds, and what the loss
    does with it. Spelled ``role:kind:value`` on the command line — ``value`` is
    everything after the second colon, so a Windows path keeps its drive."""

    role: str
    kind: str
    value: str

    def __post_init__(self) -> None:
        if self.role not in MASK_ROLES:
            raise ValueError(
                f"mask role {self.role!r}: expected one of {', '.join(MASK_ROLES)}"
            )
        if self.kind not in MASK_KINDS:
            raise ValueError(
                f"mask kind {self.kind!r}: expected one of {', '.join(MASK_KINDS)}"
            )
        if not self.value.strip():
            raise ValueError(f"mask {self.role}:{self.kind} names no prompt")
        # `none` turns --prompt_embed into text elsewhere; here the text is its own
        # kind, so a soft entry has to name a file.
        if self.kind == "soft" and self.value.strip().lower() in ("none", "off"):
            raise ValueError(f"mask {self.spec()}: a soft prompt names a file")

    @classmethod
    def parse(cls, spec: str) -> MaskPrompt:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"mask {spec!r}: expected ROLE:KIND:VALUE")
        role, kind, value = (p.strip() for p in parts)
        return cls(role=role, kind=kind, value=value)

    def spec(self) -> str:
        return f"{self.role}:{self.kind}:{self.value}"


SUBJECT_MASK = MaskPrompt("keep", "soft", DEFAULT_SUBJECT_PROMPT_EMBED)
"""The default list's one entry: keep the subject, found by the learned soft
prompt for ``girl`` (the text prompt when that file is not downloaded)."""


def _read_masks(values) -> tuple[MaskPrompt, ...]:
    return tuple(MaskPrompt.parse(v) for v in values or ())


def _write_masks(masks: tuple[MaskPrompt, ...]) -> list[str]:
    return [m.spec() for m in masks]


def _mask_dir(default: str) -> str:
    """The generator's own tree, never the merged ``masks`` root: a hand-made tree
    merged in beside it names a mask ``{stem}_mask.png`` at the same relative path,
    so sharing the root would have the second run overwrite the first."""
    return arg(
        default,
        help=f"Output mask directory for this generator alone (default: {default}); "
        "`merge_masks` unions it with any other tree into the masks root",
    )


@dataclass(frozen=True, kw_only=True)
class MaskWalkRequest(Request):
    """The walk ``_masks.mask_run`` reads by attribute — kept apart from the SAM3
    knobs so another generator can wrap the same loop."""

    image_dir: str = arg(help="Image directory")
    mask_dir: str
    """The generator's own tree, never the merged ``masks`` root."""
    force: bool = arg(False, help="Regenerate existing masks")
    workers: int = arg(4, help="I/O workers for loading/saving (default: 4)")
    recursive: bool = arg(False, help=WALK_HELP)
    path_pattern: str | None = arg(None, help=PATTERN_HELP)
    device: str | None = arg(None, help=DEVICE_HELP)
    """``None`` resolves at run time (``_device.resolve_device``)."""


@dataclass(frozen=True, kw_only=True)
class SamMaskRequest(MaskWalkRequest):
    """SAM3 masks, written to ``workspace/masks_sam/``.

    ``--masks`` is the list of regions, each ``ROLE:KIND:VALUE``. A ``keep`` region
    is what trains — everything outside the union of them is masked out; an
    ``ignore`` region is masked out of that (or out of the whole image when
    nothing is kept). A ``text`` prompt goes through SAM3's text encoder
    (``ignore:text:speech bubble``); a ``soft`` one is a learned prompt file that
    stands in for it (``keep:soft:networks/calibration/sam3_girl_prompt.safetensors``,
    the default, which falls back to the text prompt ``girl`` when not downloaded).
    """

    mask_dir: str = _mask_dir(WS.MASKS_SAM)
    masks: tuple[MaskPrompt, ...] = field(
        default=(SUBJECT_MASK,),
        metadata={
            NARGS: "*",
            METAVAR: "ROLE:KIND:VALUE",
            KIND: "masks",
            READ: _read_masks,
            WRITE: _write_masks,
            HELP: "Mask regions, one ROLE:KIND:VALUE each — role keep|ignore, kind "
            "text (a SAM3 prompt) | soft (a learned prompt .safetensors). Default: "
            "keep the subject via the shipped soft prompt",
        },
    )
    threshold: float = arg(
        0.5, help="SAM3 confidence floor for a detection (default: 0.5)"
    )
    dilate: int = arg(5, help="Mask dilation in pixels, 0 = off (default: 5)")
    checkpoint: str = arg(DEFAULT_SAM3_CHECKPOINT, help=CHECKPOINT_HELP)

    def __post_init__(self) -> None:
        if not self.masks:
            raise ValueError("nothing to mask: --masks names no region")

    @property
    def keep(self) -> tuple[MaskPrompt, ...]:
        return tuple(m for m in self.masks if m.role == "keep")

    @property
    def ignore(self) -> tuple[MaskPrompt, ...]:
        return tuple(m for m in self.masks if m.role == "ignore")


@dataclass(frozen=True, kw_only=True)
class MergeMasksRequest(Request):
    """Merge masks from multiple sources by taking the pixel-wise minimum (union of
    masked regions).

    Keys merges by ``(rel_dir, name)``, so masks at the same relative path across
    inputs collide; the nested layout is preserved under ``--output-dir``. A missing
    input directory is skipped, not an error — the default input is the one
    generator's tree, and a second tree (hand-painted masks, another tool's
    output) is simply listed beside it.
    """

    mask_dirs: tuple[str, ...] = field(
        default=(WS.MASKS_SAM,),
        metadata={
            POSITIONAL: True,
            READ: tuple,
            WRITE: list,
            HELP: "Input mask directories to merge (default: the SAM3 generator's "
            f"own tree, {WS.MASKS_SAM}; list a hand-made tree beside it)",
        },
    )
    output_dir: str = arg(
        WS.MASKS, help=f"Output directory for merged masks (default: {WS.MASKS})"
    )
