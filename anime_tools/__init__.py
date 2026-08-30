"""anime_tools — dataset curation for anime diffusion training.

Sub-packages (import what you need; the top level stays torch-free):

* ``anime_tools.captions`` — the caption grammar (``parse_caption`` /
  ``compose_caption``), tag taxonomy + correction, variants sidecars, the
  caption index. Pure Python.
* ``anime_tools.tagger`` — the Anima Tagger (dbv4 backend) + its CLIs
  (``python -m anime_tools.tagger.cli --mode …``). Needs the ``tagger`` extra.
* ``anime_tools.stages`` — the caption-master stages (autotag, position
  clauses, correction, multiview audit) and their CLIs.

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
