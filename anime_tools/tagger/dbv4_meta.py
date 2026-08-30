"""Torch-free facts about the dbv4 tagger backbone.

The Anima Tagger is a thin head over an off-the-shelf danbooru tagger. Our
half (vocab / rules / thresholds / sidecar) ships from ``sorryhyun/anima-tagger``
and lands under ``models/captioners/``; the backbone ships from a **gated,
GPL-3.0** upstream repo and lands in the HuggingFace hub cache under the
user's own token (which is also their record of accepting the terms).

Three surfaces need the same facts without importing torch — the task runner
(``scripts/tasks/downloads.py``), the GUI system dialog, and the loader —
so they live here rather than in :mod:`anime_tools.tagger.dbv4_backend`.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_DBV4_REPO = "animetimm/caformer_b36.dbv4-full"
DEFAULT_DBV4_ARCH = "caformer_b36"
DEFAULT_DBV4_IMG_SIZE = 384

# Everything :class:`Dbv4Backend` pulls from the backbone repo.
DBV4_BACKBONE_FILES = ("model.safetensors", "selected_tags.csv", "meta.json")


def gated_hint(repo: str) -> str:
    """Recovery text for a failed backbone fetch (token / terms)."""
    return f"hf auth login, then accept the terms at https://huggingface.co/{repo}"


def backbone_repo_for(ckpt_dir: str | Path) -> str:
    """Backbone repo the checkpoint at ``ckpt_dir`` was built against.

    Reads ``config.json["dbv4"]["repo"]``; falls back to the default when the
    checkpoint is absent or doesn't name one. A checkpoint built against
    another ``animetimm/*.dbv4-full`` variant must resolve *that* backbone.
    """
    try:
        with open(Path(ckpt_dir) / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return DEFAULT_DBV4_REPO
    repo = (cfg.get("dbv4") or {}).get("repo")
    return repo if isinstance(repo, str) and repo else DEFAULT_DBV4_REPO


def backbone_cached(repo: str) -> bool:
    """True when every backbone file is already in the local hub cache (offline)."""
    from anime_tools._hf import hf_file_cached

    return all(hf_file_cached(repo, f) for f in DBV4_BACKBONE_FILES)
