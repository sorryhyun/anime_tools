"""Tagger feature-cache path resolution (``post_image_dataset/anima_tagger/``).

The two path helpers survive the 2026-08-27 archive of the PE-head trainer
(``_archive/anima_tagger_training/scripts/caches.py``) because the dbv4
sidecar cache (``<root>/dbv4/<arch>_hidden.safetensors``, written by
``anime_tools/tagger/cli/train_sidecar.py``) and the legacy per-encoder token
caches (``<root>/tokens-<encoder>/``, read by ``bench/readback`` and
``bench/tagger_external`` for pre-dbv4 checkpoints) share the same root.
Change the layout here, propagate everywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["cache_dir_for", "feature_cache_root"]


def feature_cache_root(args: argparse.Namespace) -> Path:
    """Root dir for tagger feature caches.

    Honors ``--feature_cache_dir``; when unset, defaults to
    ``post_image_dataset/anima_tagger/``. Decoupled from ``--out_dir`` (the
    checkpoint + vocab home) so the bulky dataset-derived caches live next to
    the other dataset caches and are shared across checkpoints.
    """
    explicit = getattr(args, "feature_cache_dir", None)
    if explicit:
        return Path(explicit)
    return Path("post_image_dataset") / "anima_tagger"


def cache_dir_for(feature_root: Path, pool_kind: str, encoder: str) -> Path:
    """Per-encoder subdir under :func:`feature_cache_root` for ``pool_kind``
    (``map`` → ``tokens-<encoder>/``, ``mean`` → ``pooled-<encoder>/``)."""
    sub = "tokens" if pool_kind == "map" else "pooled"
    return feature_root / f"{sub}-{encoder}"
