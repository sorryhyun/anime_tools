"""anime_tools — dataset curation for anime diffusion training.

Sub-packages (import what you need; the top level stays torch-free):
``anime_tools.captions`` (grammar, taxonomy + correction, sidecars, index),
``anime_tools.tagger`` (the Anima Tagger and its CLIs), ``anime_tools.stages``
(the caption-master stages).

The trainer (``anima_lora``) depends on this package; this package never
imports the trainer (``docs/contract.md``).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

try:
    __version__ = _dist_version("anime_tools")
except PackageNotFoundError:  # a checkout on sys.path without an install
    __version__ = "0+unknown"

__all__ = ["AnimaTagger", "__version__"]


def __getattr__(name: str):
    # Lazy (PEP 562): AnimaTagger pulls torch/timm; keep ``import anime_tools`` cheap.
    if name == "AnimaTagger":
        from anime_tools.tagger.tagger import AnimaTagger

        return AnimaTagger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
