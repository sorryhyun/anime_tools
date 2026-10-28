"""Anima Tagger — a thin head over an external backbone tagger (pixai; dbv4).

``AnimaTagger`` is exposed lazily because it imports torch; ``dbv4_meta``
stays torch-free.
"""

__all__ = ["DEFAULT_TAGGER_DIR", "AnimaTagger"]


def __getattr__(name: str):
    if name in ("AnimaTagger", "DEFAULT_TAGGER_DIR"):
        from anime_tools.tagger import tagger as _t

        return getattr(_t, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
