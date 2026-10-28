"""The one place a checkpoint's ``config.json`` becomes a backbone object."""

from __future__ import annotations

import torch

from anime_tools.tagger.dbv4_backend import Dbv4Backend
from anime_tools.tagger.dbv4_meta import (
    BACKBONES,
    DEFAULT_DBV4_ARCH,
    DEFAULT_DBV4_IMG_SIZE,
    DEFAULT_DBV4_REPO,
    DEFAULT_PIXAI_IMG_SIZE,
    DEFAULT_PIXAI_REPO,
    DEFAULT_PIXAI_REVISION,
)
from anime_tools.tagger.pixai_backend import PixaiBackend

Backend = Dbv4Backend | PixaiBackend


def build_backend(
    cfg_d: dict,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Backend:
    """``config.json`` → the backend it names (lazy: weights load on first use)."""
    kind = cfg_d.get("backend")
    if kind not in BACKBONES:
        raise ValueError(
            f"unsupported tagger backend {kind!r}: expected one of {sorted(BACKBONES)}"
        )
    d = dict(cfg_d.get(kind) or {})
    if kind == "pixai":
        return PixaiBackend(
            repo=d.get("repo", DEFAULT_PIXAI_REPO),
            revision=d.get("revision", DEFAULT_PIXAI_REVISION),
            img_size=int(d.get("img_size", DEFAULT_PIXAI_IMG_SIZE)),
            device=device,
            dtype=dtype,
        )
    return Dbv4Backend(
        repo=d.get("repo", DEFAULT_DBV4_REPO),
        arch=d.get("arch", DEFAULT_DBV4_ARCH),
        img_size=int(d.get("img_size", DEFAULT_DBV4_IMG_SIZE)),
        device=device,
        dtype=dtype,
        revision=d.get("revision"),
    )
