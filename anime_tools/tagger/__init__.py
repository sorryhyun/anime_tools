"""Anima Tagger — a thin head over the external dbv4 caformer tagger.

``AnimaTagger`` is exposed lazily (it imports torch/timm); ``dbv4_meta`` is
torch-free and safe to import from download tooling / GUIs.
"""

__all__ = ["DEFAULT_TAGGER_DIR", "AnimaTagger"]


def __getattr__(name: str):
    if name in ("AnimaTagger", "DEFAULT_TAGGER_DIR"):
        from anime_tools.tagger import tagger as _t

        return getattr(_t, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
