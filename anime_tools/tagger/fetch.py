"""Getting a tagger checkpoint onto disk — torch-free.

Two ensures, both idempotent and both safe to call before anything is loaded:
:func:`ensure_tagger_checkpoint` for the checkpoint dir itself (``config.json``
decides which file set is required), and :func:`ensure_tagger_backbone` for the
gated upstream dbv4 weights, which never land in the checkpoint dir at all —
they live in the HF hub cache under the user's own token.

Where each file comes from is :mod:`anime_tools.downloads`' and
:mod:`anime_tools.tagger.dbv4_meta`'s to say; this module is the fetch itself,
so a caller can preflight a checkpoint without importing the model.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from anime_tools._json import read_json
from anime_tools.tagger.dbv4_meta import (
    DBV4_OPTIONAL_FILES,
    DBV4_REQUIRED_FILES,
    TAGGER_HF_REPO,
    TAGGER_HF_SUBFOLDER,
    TAGGER_OPTIONAL_FILES,
    TAGGER_REQUIRED_FILES,
)

__all__ = ["ensure_tagger_backbone", "ensure_tagger_checkpoint", "is_dbv4_dir"]

logger = logging.getLogger(__name__)


def is_dbv4_dir(ckpt_dir: Path) -> bool:
    """Does the checkpoint at ``ckpt_dir`` name the dbv4 backend?

    The package's one answer to the question, since it decides which file set is
    required and whether a backbone has to be fetched. An unreadable or absent
    ``config.json`` is False rather than an error: the caller is about to fetch.
    """
    try:
        return read_json(ckpt_dir / "config.json").get("backend") == "dbv4"
    except (OSError, ValueError):
        return False


def ensure_tagger_checkpoint(
    ckpt_dir: str | Path,
    repo: str = TAGGER_HF_REPO,
    subfolder: str = TAGGER_HF_SUBFOLDER,
    *,
    backbone: bool = True,
) -> Path:
    """Fetch the tagger checkpoint into ``ckpt_dir`` if any required file is missing.

    Files are flattened into ``ckpt_dir`` regardless of source layout; optional
    files are best-effort. With ``backbone=True`` a dbv4 checkpoint also runs
    :func:`ensure_tagger_backbone`, so the gated upstream weights are fetched
    here rather than lazily on the first predict.
    """
    ckpt_dir = Path(ckpt_dir)
    if all((ckpt_dir / f).exists() for f in TAGGER_REQUIRED_FILES):
        return ckpt_dir
    if all((ckpt_dir / f).exists() for f in DBV4_REQUIRED_FILES) and is_dbv4_dir(
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
    if is_dbv4_dir(ckpt_dir):
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
    if backbone and is_dbv4_dir(ckpt_dir):
        ensure_tagger_backbone(ckpt_dir)
    return ckpt_dir


def ensure_tagger_backbone(ckpt_dir: str | Path) -> str:
    """Preflight the gated dbv4 backbone for the checkpoint at ``ckpt_dir``.

    The backbone (``config.json["dbv4"]["repo"]``, GPL-3.0) is gated and only
    ever lands in the HF hub cache under the user's own token. Probes the cache
    offline first, then fetches through ``hf_download``, which turns a gated
    401/403 into a ``FileNotFoundError`` naming the accept-terms recovery.
    Returns the repo id; ``ANIMA_TAGGER_NO_AUTOFETCH=1`` fails instead of
    fetching.
    """
    from anime_tools.tagger.dbv4_meta import (
        DBV4_BACKBONE_FILES,
        backbone_cached,
        backbone_repo_for,
        gated_hint,
    )

    ckpt_dir = Path(ckpt_dir)
    repo = backbone_repo_for(ckpt_dir)
    if not is_dbv4_dir(ckpt_dir) or backbone_cached(repo):
        return repo
    if os.environ.get("ANIMA_TAGGER_NO_AUTOFETCH"):
        raise FileNotFoundError(
            f"AnimaTagger backbone {repo} is not in the HF cache and "
            f"ANIMA_TAGGER_NO_AUTOFETCH is set. Run "
            f"`python -m anime_tools.downloads tagger_backbone` "
            f"({gated_hint(repo)})."
        )
    from anime_tools._hf import hf_download

    logger.info(
        "AnimaTagger: backbone %s not cached — fetching under your HF token "
        "(gated, GPL-3.0; one-time).",
        repo,
    )
    for fname in DBV4_BACKBONE_FILES:
        hf_download(
            what=f"AnimaTagger backbone ({repo})",
            hint=gated_hint(repo),
            repo_id=repo,
            filename=fname,
        )
    return repo
