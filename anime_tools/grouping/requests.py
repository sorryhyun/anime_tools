"""The grouping stage as a request object (``docs/api_first_plan.md``).

Torch-free: run one through :func:`anime_tools.grouping.run_groups`, which
loads the embedder, or hand ``to_argv()`` to a subprocess. Every field is an
argparse ``dest`` of ``grouping/cli/build_groups.py`` with its default.
"""

from __future__ import annotations

from dataclasses import dataclass

from anime_tools import workspace as WS
from anime_tools._request import Request
from anime_tools.grouping.groups import (
    DEFAULT_CELL_MATCH_MIN,
    DEFAULT_GRID,
    DEFAULT_MATCH_FRAC_MIN,
    DEFAULT_RATIO,
    DEFAULT_SIM_MIN,
)

__all__ = ["DEFAULT_EMBEDDER", "GroupRequest"]

DEFAULT_EMBEDDER = "anime_tools.grouping.embedder:pe_spatial_embedder"
"""The PE-Spatial factory, named by string so nothing here imports torch."""


@dataclass(frozen=True, kw_only=True)
class GroupRequest(Request):
    """Group near-identical images per artist into ``groups.json``
    (``python -m anime_tools.grouping.cli.build_groups``). A curation aid for
    the GUI's Dataset tab; writes nothing else."""

    source_dir: str = WS.RESIZED
    """The image tree to group — the resized one, the pixels training sees."""
    out: str = f"{WS.GROUPS}/groups.json"
    cell_match_min: float = DEFAULT_CELL_MATCH_MIN
    """Per-cell cosine for an inlier grid-cell match."""
    match_frac_min: float = DEFAULT_MATCH_FRAC_MIN
    """Inlier fraction to connect two images (higher = tighter)."""
    sim_min: float = DEFAULT_SIM_MIN
    """Stage-A CLS-cosine prefilter (loose; the grid match is the gate)."""
    grid: int = DEFAULT_GRID
    ratio: float = DEFAULT_RATIO
    min_size: int = 2
    """Drop groups smaller than this (1 keeps singletons)."""
    embedder: str | None = None
    """``module:callable(device=...)`` returning an ``Embedder``; ``None`` is
    :data:`DEFAULT_EMBEDDER`."""
    batch_size: int = 16
    num_workers: int = 4
    device: str | None = None

    def __post_init__(self) -> None:
        if self.embedder is not None and ":" not in self.embedder:
            raise ValueError(
                f"--embedder must be `module:callable`, got {self.embedder!r}"
            )
