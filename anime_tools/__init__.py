"""anime_tools — dataset curation for anime diffusion training.

Sub-packages (import what you need; the top level stays torch-free):
``anime_tools.captions`` (the caption grammar, tag taxonomy + correction,
variants sidecars, the caption index), ``anime_tools.tagger`` (the Anima
Tagger and its CLIs), ``anime_tools.stages`` (the caption-master stages).

The trainer (``anima_lora``) depends on this package; this package never
imports the trainer (``docs/contract.md``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["AnimaTagger", "__version__"]


def __getattr__(name: str):
    # Lazy (PEP 562): AnimaTagger pulls torch/timm; keep ``import anime_tools`` cheap.
    if name == "AnimaTagger":
        from anime_tools.tagger.tagger import AnimaTagger

        return AnimaTagger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
