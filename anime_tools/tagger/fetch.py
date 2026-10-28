"""Getting a tagger checkpoint onto disk — torch-free.

Two ensures, both idempotent and both safe to call before anything is loaded:
:func:`ensure_tagger_checkpoint` for the checkpoint dir itself (``config.json``
decides which file set is required), and :func:`ensure_tagger_backbone` for the
upstream backbone weights, which never land in the checkpoint dir at all —
they live in the HF hub cache (the gated dbv4 one under the user's own token).

Where each file comes from is :mod:`anime_tools.downloads`' and
:mod:`anime_tools.tagger.dbv4_meta`'s to say; this module is the fetch itself,
so a caller can preflight a checkpoint without importing the model.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from anime_tools.tagger.dbv4_meta import (
    DBV4_OPTIONAL_FILES,
    DBV4_REQUIRED_FILES,
    TAGGER_HF_REPO,
    TAGGER_HF_SUBFOLDER,
    TAGGER_OPTIONAL_FILES,
    TAGGER_REQUIRED_FILES,
    backend_of,
)

__all__ = ["ensure_tagger_backbone", "ensure_tagger_checkpoint", "is_backbone_dir"]

logger = logging.getLogger(__name__)


def is_backbone_dir(ckpt_dir: Path) -> bool:
    """Does the checkpoint at ``ckpt_dir`` name a known backbone (pixai / dbv4)?

    The package's one answer to the question, since it decides which file set is
    required and whether a backbone has to be fetched. An unreadable or absent
    ``config.json`` is False rather than an error: the caller is about to fetch.
    """
    return backend_of(ckpt_dir) is not None


def ensure_tagger_checkpoint(
    ckpt_dir: str | Path,
    repo: str = TAGGER_HF_REPO,
    subfolder: str = TAGGER_HF_SUBFOLDER,
    *,
    backbone: bool = True,
) -> Path:
    """Fetch the tagger checkpoint into ``ckpt_dir`` if any required file is missing.

    Files are flattened into ``ckpt_dir`` regardless of source layout; optional
    files are best-effort. With ``backbone=True`` a backbone-backed checkpoint
    also runs :func:`ensure_tagger_backbone`, so the upstream weights are
    fetched here rather than lazily on the first predict.
    """
    ckpt_dir = Path(ckpt_dir)
    if all((ckpt_dir / f).exists() for f in TAGGER_REQUIRED_FILES):
        return ckpt_dir
    if all((ckpt_dir / f).exists() for f in DBV4_REQUIRED_FILES) and is_backbone_dir(
        ckpt_dir
    ):
        if backbone:
            ensure_tagger_backbone(ckpt_dir)
        return ckpt_dir
    from huggingface_hub.utils import EntryNotFoundError

    from anime_tools._hf import hf_download

    logger.info(
        "AnimaTagger: %s missing required files — fetching %s%s (one-time).",
        ckpt_dir,
        repo,
        f"/{subfolder}" if subfolder else "",
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_flat(fname: str) -> Path:
        repo_path = f"{subfolder}/{fname}" if subfolder else fname
        downloaded = Path(
            hf_download(
                what="AnimaTagger weights",
                repo_id=repo,
                filename=repo_path,
                local_dir=str(ckpt_dir),
            )
        )
        dest = ckpt_dir / fname
        if downloaded.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), str(dest))
        return dest

    # config.json first: it decides which file set is required.
    _fetch_flat("config.json")
    if is_backbone_dir(ckpt_dir):
        required, optional = DBV4_REQUIRED_FILES, DBV4_OPTIONAL_FILES
    else:
        required, optional = TAGGER_REQUIRED_FILES, TAGGER_OPTIONAL_FILES
    for fname in required:
        if fname != "config.json":
            _fetch_flat(fname)
    for fname in optional:
        if (ckpt_dir / fname).exists():
            continue
        try:
            _fetch_flat(fname)
        except (EntryNotFoundError, FileNotFoundError):
            logger.debug("optional tagger file %s not present on %s", fname, repo)
    if backbone and is_backbone_dir(ckpt_dir):
        ensure_tagger_backbone(ckpt_dir)
    return ckpt_dir


def ensure_tagger_backbone(ckpt_dir: str | Path) -> str:
    """Preflight the backbone named by the checkpoint at ``ckpt_dir``.

    The backbone (``config.json[<backend>]["repo"]`` at its ``revision``) only
    ever lands in the HF hub cache. Probes the cache offline first, then
    fetches through ``hf_download``, which turns a gated 401/403 into a
    ``FileNotFoundError`` naming the accept-terms recovery. Returns the repo
    id; ``ANIMA_TAGGER_NO_AUTOFETCH=1`` fails instead of fetching.
    """
    from anime_tools.tagger.dbv4_meta import backbone_cached, backbone_for, fetch_hint

    ckpt_dir = Path(ckpt_dir)
    backbone = backbone_for(ckpt_dir)
    if not is_backbone_dir(ckpt_dir) or backbone_cached(backbone):
        return backbone.repo
    hint = fetch_hint(backbone)
    if os.environ.get("ANIMA_TAGGER_NO_AUTOFETCH"):
        raise FileNotFoundError(
            f"AnimaTagger backbone {backbone.repo} is not in the HF cache and "
            f"ANIMA_TAGGER_NO_AUTOFETCH is set. Run "
            f"`python -m anime_tools.downloads tagger_backbone` ({hint})."
        )
    from anime_tools._hf import hf_download

    logger.info(
        "AnimaTagger: backbone %s not cached — fetching (%s%s; one-time).",
        backbone.repo,
        backbone.spec.license,
        ", gated, under your HF token" if backbone.spec.gated else "",
    )
    for fname in backbone.spec.files:
        hf_download(
            what=f"AnimaTagger backbone ({backbone.repo})",
            hint=hint,
            repo_id=backbone.repo,
            filename=fname,
            revision=backbone.revision,
        )
    return backbone.repo
