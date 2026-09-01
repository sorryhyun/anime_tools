"""Anima Tagger ComfyUI nodes.

* ``AnimaTaggerLoader`` — load an ``AnimaTagger`` checkpoint and emit a reusable
  ``ANIMA_TAGGER`` socket.
* ``AnimaTaggerCaption`` — take an ``ANIMA_TAGGER`` + ``IMAGE``, return the
  predicted caption as a ``STRING``.

``ANIMA_TAGGER`` is a plain string in ComfyUI's type system, so the
AnimaDirectEdit node in ``comfyui-anima-directedit`` consumes the same socket
with no code-level dependency on this package.

*What a tagger checkpoint is* — the repo it ships from, which files each backend
requires, the backend test, the fetch — belongs to ``anime_tools`` and is
imported from it. What stays here is the ComfyUI shell: the dropdown, the
IMAGE→PIL conversion, and ``HOME``, the one question ComfyUI answers
differently.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import comfy.model_management
import numpy as np
import torch
from PIL import Image

try:
    from anime_tools.tagger import AnimaTagger
    from anime_tools.tagger.dbv4_meta import DEFAULT_TAGGER_DIR, TAGGER_HF_REPO

    # ``_is_dbv4_dir`` is private, but it is the package's one answer to "is this
    # checkpoint dbv4-backed?", and both the dropdown and the loader's log line
    # ask it. A second spelling here is exactly the fork this module used to be.
    from anime_tools.tagger.tagger import _is_dbv4_dir, ensure_tagger_checkpoint
except ImportError as e:  # pragma: no cover — install-time guidance
    # No extra to name: `anime_tools` has had none since 0.3 (torch, sam3 and
    # the rest are plain dependencies), so this is the spelling that works and
    # the one this node's own pyproject.toml declares.
    raise ImportError(
        "comfyui-anima-tagger needs the `anime-tools` package: "
        "pip install 'anime-tools @ git+https://github.com/sorryhyun/anime_tools'"
    ) from e

logger = logging.getLogger(__name__)

# Relative ``tagger_dir`` values resolve under ``HOME``: ``ANIME_TOOLS_HOME`` /
# ``ANIMA_HOME`` when set, else the ComfyUI base directory, else this node's
# parent dir. Deliberately *not* ``_env.curation_home()``, whose last resort is
# the CWD — under ComfyUI that is wherever the server happened to be launched
# from, while ``folder_paths.base_path`` is the install the user keeps models in.
HERE = Path(__file__).resolve().parent


def _home() -> Path:
    for key in ("ANIME_TOOLS_HOME", "ANIMA_HOME"):
        v = os.environ.get(key)
        if v:
            return Path(v).expanduser().resolve()
    try:
        import folder_paths  # ComfyUI

        return Path(folder_paths.base_path).resolve()
    except Exception:  # noqa: BLE001 — outside ComfyUI
        return HERE.parents[1]


HOME = _home()

# The shipped default (``DEFAULT_TAGGER_DIR``) is the **dbv4-backed** checkpoint:
# the ``dbv4/`` subfolder of ``sorryhyun/anima-tagger``, an external caformer
# trunk projected onto our vocab plus our sidecar head. It uses NO PE encoder,
# which is why this node exposes no PE inputs; a legacy PE-head checkpoint still
# loads and resolves its encoders through the registry default, also with no knob
# here. Which files either shape requires, and where they come from, is
# ``ensure_tagger_checkpoint``'s business rather than this module's.


def _list_tagger_dirs() -> list[str]:
    """Loadable tagger-checkpoint directories under ``models/captioners/``.

    A directory qualifies when its ``config.json`` declares the ``dbv4`` backend
    or carries an ``aux_encoder`` field (legacy dual-encoder) — the only two
    architectures that still load, so the dropdown never offers a checkpoint
    ``AnimaTagger`` would reject. Paths are relative to ``HOME``, which the
    ``load()`` resolver prepends. Empty when nothing is installed; the caller
    seeds the default dir so auto-fetch still works.
    """
    if not HOME.exists():
        return []
    base = HOME / "models" / "captioners"
    if not base.is_dir():
        return []
    out: list[str] = []
    for p in sorted(base.iterdir()):
        cfg = p / "config.json"
        if not cfg.exists():
            continue
        try:
            with open(cfg, encoding="utf-8") as f:
                cfg_d = json.load(f)
        except (OSError, ValueError):
            continue
        # The dbv4 half of the test is the package's, so the dropdown and the
        # loader cannot disagree about what a dbv4 checkpoint is; only the legacy
        # ``aux_encoder`` shape, which the package has no name for, is read here.
        if not _is_dbv4_dir(p) and not cfg_d.get("aux_encoder"):
            continue
        out.append(str(p.relative_to(HOME)))
    return out


def _dropdown(default: str, found: list[str]) -> list[str]:
    """Build a dropdown list with ``default`` pinned first, deduped.

    The default heads the list even when absent from ``found``, so it stays
    selectable as the auto-fetch target.
    """
    return [default] + [p for p in found if p != default]


def _comfy_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE [B, H, W, C] in [0,1] -> PIL.RGB (first batch element)."""
    arr = image_tensor[0].clamp(0, 1).cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


class AnimaTaggerLoader:
    """Load an AnimaTagger checkpoint as a reusable graph asset.

    ComfyUI memoizes loader outputs by inputs, so the tagger persists across
    invocations as long as ``tagger_dir`` doesn't change.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tagger_dir": (
                    _dropdown(DEFAULT_TAGGER_DIR, _list_tagger_dirs()),
                    {
                        "tooltip": (
                            "AnimaTagger checkpoint directory, picked from "
                            "the checkpoints discovered under "
                            "models/captioners/ (any dir with a config.json). "
                            "The first row is the default dbv4-backed "
                            "checkpoint; if it's missing required files it's "
                            f"auto-fetched from {TAGGER_HF_REPO} on first use."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("ANIMA_TAGGER",)
    RETURN_NAMES = ("tagger",)
    FUNCTION = "load"
    CATEGORY = "anima"
    DESCRIPTION = (
        "Load an AnimaTagger checkpoint. Output socket is consumed by "
        "AnimaTaggerCaption (image -> caption) and AnimaDirectEdit. ComfyUI "
        "memoizes the output, so re-running the graph reuses the same "
        f"instance without reloading. The tagger checkpoint ({TAGGER_HF_REPO}) "
        "is auto-downloaded if its path doesn't exist yet; the dbv4 backend "
        "needs no PE vision encoder, so there is nothing else to pick."
    )

    def load(self, tagger_dir: str, **_legacy):
        """``**_legacy`` swallows ``pe_ckpt`` / ``pe_aux_ckpt`` from workflows
        saved before those inputs were removed, so an old graph still runs."""
        tdir = Path(tagger_dir.strip())
        if not tdir.is_absolute():
            tdir = HOME / tdir
        # The package's preflight, not a copy of it: for a dbv4 checkpoint
        # ``ensure_tagger_checkpoint`` ends by fetching the **gated GPL-3.0**
        # caformer backbone, so the token/terms failure — and the multi-GB
        # download — happen here, while the loader node is the one running, and
        # not in the middle of the first predict, which under ComfyUI is a
        # queued prompt stalling on an error nothing on the graph explains.
        # It also downloads through ``anime_tools._hf``, which turns a gated
        # 401/403 into a FileNotFoundError naming the accept-terms recovery.
        # Repo and subfolder are its defaults (``sorryhyun/anima-tagger``,
        # ``dbv4/``) — the same ones ``DEFAULT_TAGGER_DIR`` above is named for.
        ensure_tagger_checkpoint(tdir)
        device = comfy.model_management.get_torch_device()
        logger.info(
            "AnimaTaggerLoader: loading %s on %s (backend=%s)",
            tdir,
            device,
            "dbv4" if _is_dbv4_dir(tdir) else "pe",
        )
        # PE checkpoints stay unset: inert on dbv4, encoder-registry default
        # (-> HF fallback) on a legacy PE-head checkpoint.
        tagger = AnimaTagger(ckpt_dir=tdir, device=device)
        return (tagger,)


class AnimaTaggerCaption:
    """Run an AnimaTagger over an image and emit the caption as a STRING."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tagger": (
                    "ANIMA_TAGGER",
                    {"tooltip": "AnimaTagger from AnimaTaggerLoader."},
                ),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption"
    CATEGORY = "anima"
    DESCRIPTION = (
        "Predict an Anima-formatted caption for an image using AnimaTagger. "
        "Output is a comma-separated tag string in Anima's canonical slot "
        "order (rating, count, characters, copyrights, @artists, generals) "
        "with underscores replaced by spaces - drop-in for any STRING input "
        "(CLIPTextEncode, prompt_src_override on AnimaDirectEdit, etc.)."
    )

    def caption(self, tagger: AnimaTagger, image: torch.Tensor):
        pil = _comfy_image_to_pil(image)
        text = tagger.predict_caption(pil)
        logger.info("AnimaTaggerCaption: %r", text)
        return (text,)


NODE_CLASS_MAPPINGS = {
    "AnimaTaggerLoader": AnimaTaggerLoader,
    "AnimaTaggerCaption": AnimaTaggerCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaTaggerLoader": "Anima Tagger Loader",
    "AnimaTaggerCaption": "Anima Tagger Caption",
}
