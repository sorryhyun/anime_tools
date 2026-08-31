"""Anima Tagger ComfyUI nodes.

* ``AnimaTaggerLoader`` — load an ``AnimaTagger`` checkpoint and emit a reusable
  ``ANIMA_TAGGER`` socket.
* ``AnimaTaggerCaption`` — take an ``ANIMA_TAGGER`` + ``IMAGE``, return the
  predicted caption as a ``STRING``.

``ANIMA_TAGGER`` is a plain string in ComfyUI's type system, so the
AnimaDirectEdit node in ``comfyui-anima-directedit`` consumes the same socket
with no code-level dependency on this package.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch

# Relative ``tagger_dir`` values resolve under ``HOME``: ``ANIME_TOOLS_HOME`` /
# ``ANIMA_HOME`` when set, else the ComfyUI base directory, else this node's
# parent dir.
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

# Auto-fetched when ``tagger_dir`` is missing required files, so the node works
# out of the box while ``tagger_dir`` can still name a custom checkpoint.
_HF_TAGGER_REPO = "sorryhyun/anima-tagger"
_REQUIRED_FILES = ("config.json", "model.safetensors", "vocab.json", "rules.yaml")
# dbv4-backed checkpoints ship no weights of ours — the GPL-3.0 caformer
# backbone is fetched from its own gated upstream repo by the library (needs
# `timm` + an HF token that accepted the terms). Only vocab/rules/thresholds +
# the sidecar head live on our repo.
_DBV4_REQUIRED_FILES = ("config.json", "vocab.json", "rules.yaml")
_DBV4_OPTIONAL_FILES = (
    "thresholds.safetensors",
    "groups.yaml",
    "sidecar.safetensors",
    "sidecar.json",
)


def _is_dbv4_dir(tdir: Path) -> bool:
    try:
        import json

        with open(tdir / "config.json", encoding="utf-8") as f:
            return json.load(f).get("backend") == "dbv4"
    except (OSError, ValueError):
        return False


_OPTIONAL_FILES = ("thresholds.safetensors", "groups.yaml")

# The shipped default is the **dbv4-backed** checkpoint (``dbv4/`` subfolder of
# ``sorryhyun/anima-tagger``): an external caformer trunk projected onto our
# vocab plus our sidecar head. It uses NO PE encoder, which is why this node
# exposes no PE inputs; a legacy PE-head checkpoint still loads and resolves its
# encoders through the registry default, also with no knob here.
_HF_SUBFOLDER = "dbv4"
_DEFAULT_TAGGER_DIR = "models/captioners/anima-tagger-dbv4"


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
        if cfg_d.get("backend") != "dbv4" and not cfg_d.get("aux_encoder"):
            continue
        out.append(str(p.relative_to(HOME)))
    return out


def _dropdown(default: str, found: list[str]) -> list[str]:
    """Build a dropdown list with ``default`` pinned first, deduped.

    The default heads the list even when absent from ``found``, so it stays
    selectable as the auto-fetch target.
    """
    return [default] + [p for p in found if p != default]


def _ensure_tagger_dir(tdir: Path, hf_subfolder: str = "") -> None:
    """If ``tdir`` is missing any required tagger file, fetch the whole
    checkpoint from ``sorryhyun/anima-tagger`` into it.

    ``hf_subfolder`` prefixes the repo path so different versions can be pulled
    from one repo; downloads are flattened into ``tdir`` so the loader's
    directory contract stays uniform across versions. Optional files are
    best-effort — a 404 just means the published checkpoint doesn't ship one.
    """
    if all((tdir / f).exists() for f in _REQUIRED_FILES):
        return
    if all((tdir / f).exists() for f in _DBV4_REQUIRED_FILES) and _is_dbv4_dir(tdir):
        return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    logger.info(
        "AnimaTaggerLoader: %s is missing required files - fetching %s%s (one-time).",
        tdir,
        _HF_TAGGER_REPO,
        f"/{hf_subfolder}" if hf_subfolder else "",
    )
    tdir.mkdir(parents=True, exist_ok=True)

    def _fetch_flat(fname: str) -> Path:
        repo_path = f"{hf_subfolder}/{fname}" if hf_subfolder else fname
        downloaded = Path(
            hf_hub_download(
                repo_id=_HF_TAGGER_REPO,
                filename=repo_path,
                local_dir=str(tdir),
            )
        )
        dest = tdir / fname
        if downloaded.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), str(dest))
        return dest

    # config.json first — it decides whether model.safetensors (PE head) is
    # needed, or just data + sidecar (dbv4 backend).
    _fetch_flat("config.json")
    if _is_dbv4_dir(tdir):
        required, optional = _DBV4_REQUIRED_FILES, _DBV4_OPTIONAL_FILES
    else:
        required, optional = _REQUIRED_FILES, _OPTIONAL_FILES
    for fname in required:
        if fname != "config.json":
            _fetch_flat(fname)
    for fname in optional:
        try:
            _fetch_flat(fname)
        except EntryNotFoundError:
            logger.debug(
                "optional file %s not present on %s%s",
                fname,
                _HF_TAGGER_REPO,
                f"/{hf_subfolder}" if hf_subfolder else "",
            )


import comfy.model_management
from PIL import Image

try:
    from anime_tools.tagger import AnimaTagger
except ImportError as e:  # pragma: no cover — install-time guidance
    raise ImportError(
        "comfyui-anima-tagger needs the `anime-tools[tagger]` package: "
        "pip install 'anime-tools[tagger] @ git+https://github.com/sorryhyun/anime_tools'"
    ) from e

logger = logging.getLogger(__name__)


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
                    _dropdown(_DEFAULT_TAGGER_DIR, _list_tagger_dirs()),
                    {
                        "tooltip": (
                            "AnimaTagger checkpoint directory, picked from "
                            "the checkpoints discovered under "
                            "models/captioners/ (any dir with a config.json). "
                            "The first row is the default dbv4-backed "
                            "checkpoint; if it's missing required files it's "
                            f"auto-fetched from {_HF_TAGGER_REPO} on first use."
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
        f"instance without reloading. The tagger checkpoint ({_HF_TAGGER_REPO}) "
        "is auto-downloaded if its path doesn't exist yet; the dbv4 backend "
        "needs no PE vision encoder, so there is nothing else to pick."
    )

    def load(self, tagger_dir: str, **_legacy):
        """``**_legacy`` swallows ``pe_ckpt`` / ``pe_aux_ckpt`` from workflows
        saved before those inputs were removed, so an old graph still runs."""
        tdir = Path(tagger_dir.strip())
        if not tdir.is_absolute():
            tdir = HOME / tdir
        _ensure_tagger_dir(tdir, hf_subfolder=_HF_SUBFOLDER)
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
