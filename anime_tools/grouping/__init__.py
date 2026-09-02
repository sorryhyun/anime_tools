"""Dataset grouping by PE-Spatial visual similarity.

``features`` — per-image feature cache (``$NEAR_TWIN_CACHE``); ``matching`` —
dense grid matching; ``groups`` — per-artist connected components →
``groups.json``; ``embedder`` — the default PE-Spatial embedder. The surface is
``run_groups(GroupRequest(...))`` (:mod:`requests` is torch-free) with the CLIs
in ``cli/`` as shells; both names are exposed lazily (PEP 562).
"""

__all__ = ["GroupRequest", "run_groups"]


def __getattr__(name: str):
    if name == "GroupRequest":
        from anime_tools.grouping.requests import GroupRequest

        return GroupRequest
    if name == "run_groups":
        from anime_tools.grouping.groups import run_groups

        return run_groups
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
