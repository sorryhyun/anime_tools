"""Torch-free facts about the tagger backbones.

Our half of the tagger (vocab / rules / thresholds / sidecar) ships from
``sorryhyun/anima-tagger`` into ``models/captioners/``; the backbone ships from
its upstream repo into the HuggingFace hub cache. Two backbones exist:

* ``pixai`` (the default) — ``pixai-labs/pixai-tagger-v1.0``, Apache-2.0,
  public; its model code (``tagger_pipeline.py``) is loaded through
  transformers' remote code at a pinned commit.
* ``dbv4`` — ``animetimm/caformer_b36.dbv4-full``, GPL-3.0 and gated, fetched
  under the user's own token.

A checkpoint names its backbone in ``config.json["backend"]`` and carries a
``config.json[<backend>]`` block (``repo`` / ``revision`` / ``img_size`` …).
``tagger.py`` re-exports every name below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anime_tools._json import read_json
from anime_tools.contract import (
    DBV4_BACKBONE_FILES,
    DBV4_OPTIONAL_FILES,
    DBV4_REQUIRED_FILES,
    PIXAI_BACKBONE_FILES,
    TAGGER_OPTIONAL_FILES,
    TAGGER_REQUIRED_FILES,
)

__all__ = [
    "DBV4_BACKBONE_FILES",
    "DBV4_OPTIONAL_FILES",
    "DBV4_REQUIRED_FILES",
    "PIXAI_BACKBONE_FILES",
    "TAGGER_OPTIONAL_FILES",
    "TAGGER_REQUIRED_FILES",
]

DEFAULT_DBV4_REPO = "animetimm/caformer_b36.dbv4-full"
DEFAULT_DBV4_ARCH = "caformer_b36"
DEFAULT_DBV4_IMG_SIZE = 384

DEFAULT_PIXAI_REPO = "pixai-labs/pixai-tagger-v1.0"
DEFAULT_PIXAI_REVISION = "9fe10addf9326e292da8a85a98ea74cd91b41771"
DEFAULT_PIXAI_IMG_SIZE = 1008


@dataclass(frozen=True)
class BackboneSpec:
    kind: str
    default_repo: str
    default_revision: str | None
    files: tuple[str, ...]
    license: str
    gated: bool


BACKBONES: dict[str, BackboneSpec] = {
    "pixai": BackboneSpec(
        "pixai",
        DEFAULT_PIXAI_REPO,
        DEFAULT_PIXAI_REVISION,
        PIXAI_BACKBONE_FILES,
        "apache-2.0",
        gated=False,
    ),
    "dbv4": BackboneSpec(
        "dbv4",
        DEFAULT_DBV4_REPO,
        None,
        DBV4_BACKBONE_FILES,
        "gpl-3.0",
        gated=True,
    ),
}
DEFAULT_BACKEND = "pixai"

# Our half of the tagger: auto-fetched when ckpt_dir is missing required files.
# The pixai checkpoint sits under ``pixai/``; the repo root still holds the
# dbv4 one, which every anime_tools <= 0.8 fetches. The file sets are contract
# (the trainer probes them too), so they live in ``anime_tools.contract`` and
# are re-exported here.
TAGGER_HF_REPO = "sorryhyun/anima-tagger"
TAGGER_HF_SUBFOLDER = "pixai"
DEFAULT_TAGGER_DIR = "models/captioners/anima-tagger-pixai"


@dataclass(frozen=True)
class Backbone:
    kind: str
    repo: str
    revision: str | None

    @property
    def spec(self) -> BackboneSpec:
        return BACKBONES[self.kind]


def gated_hint(repo: str) -> str:
    """Recovery text for a failed backbone fetch (token / terms)."""
    return f"hf auth login, then accept the terms at https://huggingface.co/{repo}"


def fetch_hint(backbone: Backbone) -> str:
    if backbone.spec.gated:
        return gated_hint(backbone.repo)
    return "python -m anime_tools.downloads tagger_backbone"


def backend_of(ckpt_dir: str | Path) -> str | None:
    """``config.json["backend"]`` when it names a known backbone, else ``None``."""
    try:
        kind = read_json(Path(ckpt_dir) / "config.json").get("backend")
    except (OSError, ValueError):
        return None
    return kind if kind in BACKBONES else None


def backbone_for(ckpt_dir: str | Path) -> Backbone:
    """Backbone the checkpoint at ``ckpt_dir`` was built against.

    Reads ``config.json[<backend>]``; an absent or unreadable checkpoint
    answers the default backbone.
    """
    try:
        cfg = read_json(Path(ckpt_dir) / "config.json")
    except (OSError, ValueError):
        cfg = {}
    kind = cfg.get("backend")
    if kind not in BACKBONES:
        kind = DEFAULT_BACKEND
    spec = BACKBONES[kind]
    block = cfg.get(kind) or {}
    repo = block.get("repo")
    if not (isinstance(repo, str) and repo):
        return Backbone(kind, spec.default_repo, spec.default_revision)
    return Backbone(kind, repo, block.get("revision"))


def backbone_repo_for(ckpt_dir: str | Path) -> str:
    return backbone_for(ckpt_dir).repo


def backbone_cached(backbone: Backbone) -> bool:
    """True when every backbone file is already in the local hub cache (offline)."""
    from anime_tools._hf import hf_file_cached

    return all(
        hf_file_cached(backbone.repo, f, revision=backbone.revision)
        for f in backbone.spec.files
    )
