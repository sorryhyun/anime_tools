"""What a SAM3 stage means by *a prompt*: the subject's text prompt, the learned
soft prompt that stands in for it, and the two flags that name them.

Split out of :mod:`_sam3` so a request object can take its help and its defaults
from here without the model module's ``import numpy`` side effect — the GUI
resolves every request class to build its form. Torch-free and numpy-free:
``safetensors`` is imported inside :func:`load_soft_prompt`.
"""

from __future__ import annotations

from pathlib import Path

# Shipped SAM3 soft prompt for the subject pass (textual inversion of ``anime
# girl``): keeps ``anime girl``'s recall with ``girl``'s junk profile. Part
# prompts stay textual. The path comes from the download catalog, which is
# torch-free, and the CLIs import it from here.
from anime_tools.downloads import DEFAULT_SUBJECT_PROMPT_EMBED

SUBJECT_PROMPT = "girl"
"""The text prompt every SAM3 stage means by *the subject*, and the phrase the shipped
soft prompt is the textual inversion of: ``--prompt``'s default, and what
``--prompt_embed`` stands in for."""


CHECKPOINT_HELP = "SAM3 weights"
PROMPT_EMBED_HELP = (
    "learned soft prompt (.safetensors) used in place of the "
    f"{SUBJECT_PROMPT!r} text prompt for the subject pass; every other "
    f"prompt stays textual. Default = the shipped "
    f"{DEFAULT_SUBJECT_PROMPT_EMBED}; pass `none` for the plain text prompt"
)
"""The help for ``--checkpoint`` / ``--prompt_embed``, wherever they are declared:
the stage requests carry them as field metadata, the probe CLIs take them through
``_sam3.add_checkpoint_arg`` / ``add_prompt_embed_arg``. Both name a
file a ⚙ Settings → Models row writes and are
:data:`anime_tools.gui.stages.SETTING_FIELDS` dests filled once from Settings,
which only works while every stage spells them identically."""


_NO_PROMPTS = {"none", "off"}


def prompt_list(spec: str) -> tuple[str, ...]:
    """A comma-separated prompt flag as the tuple of prompts it names.

    ``none`` / ``off`` mean *no prompts*. Emptying the field is not enough to say it: the
    GUI omits a flag whose value is blank, so a cleared prompt field would come back as
    its default.
    """
    if spec.strip().lower() in _NO_PROMPTS:
        return ()
    return tuple(t.strip() for t in spec.split(",") if t.strip())


_PROMPT_EMBED_OFF = {"", "none", "off", "text"}


def resolve_prompt_embed(spec: str | None) -> Path | None:
    """``None``/``none``/``off``/``""`` -> text prompt; else the resolved file.

    A missing *default* file degrades to the text prompt with a warning (a
    relocated checkout without the artifact); an explicit missing path raises.
    """
    import warnings

    from anime_tools._env import resolve_path

    if spec is None or spec.strip().lower() in _PROMPT_EMBED_OFF:
        return None
    path = resolve_path(spec)
    if path.exists():
        return path
    if spec == DEFAULT_SUBJECT_PROMPT_EMBED:
        warnings.warn(
            f"shipped soft prompt missing at {path}; falling back to the text "
            "prompt (get it with `python -m anime_tools.downloads soft_prompt`)",
            stacklevel=2,
        )
        return None
    raise FileNotFoundError(f"--prompt_embed {spec!r} not found at {path}")


# Keys a SAM3 soft prompt is stored under: SAM3 encodes a text prompt into this
# triple and the rest of the model only ever sees it.
SOFT_PROMPT_KEYS = ("language_features", "language_mask", "language_embeds")


def load_soft_prompt(path: str | Path, device: str | None = None) -> dict:
    """The three prompt tensors from a saved soft prompt, on ``device`` (auto
    when ``None``)."""
    from safetensors.torch import load_file

    from anime_tools._device import resolve_device

    tensors = load_file(str(path), device=resolve_device(device))
    return {k: tensors[k] for k in SOFT_PROMPT_KEYS}


def prompt_embed_sha256(path: Path | None) -> str | None:
    import hashlib

    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "CHECKPOINT_HELP",
    "PROMPT_EMBED_HELP",
    "SOFT_PROMPT_KEYS",
    "SUBJECT_PROMPT",
    "load_soft_prompt",
    "prompt_embed_sha256",
    "prompt_list",
    "resolve_prompt_embed",
]
