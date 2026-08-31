"""Torch-free facts about the dbv4 tagger backbone.

The Anima Tagger is a thin head over an off-the-shelf danbooru tagger. Our
half (vocab / rules / thresholds / sidecar) ships from ``sorryhyun/anima-tagger``
and lands under ``models/captioners/``; the backbone ships from a **gated,
GPL-3.0** upstream repo and lands in the HuggingFace hub cache under the
user's own token (which is also their record of accepting the terms).

Three surfaces need the same facts without importing torch — the download
catalog (:mod:`anime_tools.downloads`, which the GUI's Settings dialog lists),
the ComfyUI loader, and :mod:`anime_tools.tagger.tagger` itself — so they live
here rather than in :mod:`anime_tools.tagger.dbv4_backend`, which imports
torch. ``tagger.py`` re-exports every name below, so the historical
``from anime_tools.tagger.tagger import DEFAULT_TAGGER_DIR`` still works.
"""

from __future__ import annotations

from pathlib import Path

from anime_tools._json import read_json

DEFAULT_DBV4_REPO = "animetimm/caformer_b36.dbv4-full"
DEFAULT_DBV4_ARCH = "caformer_b36"
DEFAULT_DBV4_IMG_SIZE = 384

# Everything :class:`Dbv4Backend` pulls from the backbone repo.
DBV4_BACKBONE_FILES = ("model.safetensors", "selected_tags.csv", "meta.json")

# --- our half of the tagger: the checkpoint dir, and where it ships from -----
# Auto-fetch repo when ckpt_dir is missing required files (mirrors the ComfyUI
# loader). The live checkpoint lives under the `dbv4/` subfolder (2026-08-27:
# the external caformer_b36 backend + our sidecar — vocab / rules / groups /
# thresholds / sidecar only, the GPL-3.0 backbone weights come from the
# upstream gated repo); `v5/` is the last in-house PE dual-encoder head, kept
# as the fallback. `v3/` and the legacy v2 root files were deleted 2026-08-29.
TAGGER_HF_REPO = "sorryhyun/anima-tagger"
TAGGER_HF_SUBFOLDER = "dbv4"
TAGGER_REQUIRED_FILES = ("config.json", "model.safetensors", "vocab.json", "rules.yaml")
TAGGER_OPTIONAL_FILES = ("thresholds.safetensors", "groups.yaml")
# dbv4-backed checkpoints carry no model.safetensors (weights come from the
# gated upstream repo); the sidecar pair is optional.
DBV4_REQUIRED_FILES = ("config.json", "vocab.json", "rules.yaml")
DBV4_OPTIONAL_FILES = TAGGER_OPTIONAL_FILES + ("sidecar.safetensors", "sidecar.json")
DEFAULT_TAGGER_DIR = "models/captioners/anima-tagger-dbv4"


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
        cfg = read_json(Path(ckpt_dir) / "config.json")
    except (OSError, ValueError):
        return DEFAULT_DBV4_REPO
    repo = (cfg.get("dbv4") or {}).get("repo")
    return repo if isinstance(repo, str) and repo else DEFAULT_DBV4_REPO


def backbone_cached(repo: str) -> bool:
    """True when every backbone file is already in the local hub cache (offline)."""
    from anime_tools._hf import hf_file_cached

    return all(hf_file_cached(repo, f) for f in DBV4_BACKBONE_FILES)
