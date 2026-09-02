"""The grouping stage as a request object.

Torch-free: run one through :func:`anime_tools.grouping.run_groups`, which
loads the embedder, or hand ``to_argv()`` to a subprocess. Every field is a
flag of ``grouping/cli/build_groups.py``, whose parser is generated from the
class (:meth:`Request.parser`); flags are hyphenated (``--source-dir``) and take
the underscore spelling as an alias.
"""

from __future__ import annotations

from dataclasses import dataclass

from anime_tools import workspace as WS
from anime_tools._device import DEVICE_HELP
from anime_tools._request import Request, arg
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
    """Group dataset images by PE-Spatial visual similarity → groups.json manifest.

    A curation tool, not a preprocess/training step: it clusters near-identical
    images per artist so the GUI Dataset tab can filter by group, and writes nothing
    else. Two images group when ``match_frac >= --match-frac-min`` at per-cell floor
    ``--cell-match-min``. Re-runs reuse the shared PE-Spatial feature cache, so
    retuning the thresholds is cheap.
    """

    source_dir: str = arg(
        WS.RESIZED,
        help="Image tree to group. Defaults to the resized tree, which is the pixel "
        "data every other stage reads (and what training sees)",
    )
    out: str = arg(
        f"{WS.GROUPS}/groups.json", help="Manifest path the GUI Dataset tab reads"
    )
    cell_match_min: float = arg(
        DEFAULT_CELL_MATCH_MIN, help="per-cell cosine for an inlier grid-cell match"
    )
    match_frac_min: float = arg(
        DEFAULT_MATCH_FRAC_MIN,
        help="inlier fraction to connect two images (higher = tighter)",
    )
    sim_min: float = arg(
        DEFAULT_SIM_MIN,
        help="Stage-A CLS-cosine prefilter (loose; the grid match is the gate)",
    )
    grid: int = arg(DEFAULT_GRID, help="pooled grid edge (G×G cells)")
    ratio: float = arg(
        DEFAULT_RATIO, help="ratio-test distinctiveness (lower = stricter)"
    )
    min_size: int = arg(2, help="drop groups smaller than this (1 keeps singletons)")
    embedder: str | None = arg(
        None,
        help="dotted factory `module:callable(device=...)` returning a grouping "
        f"Embedder (default: the package's PE-Spatial, `{DEFAULT_EMBEDDER}`).",
    )
    batch_size: int = arg(16, help="embed batch size")
    num_workers: int = arg(4, help="DataLoader image-decode workers")
    device: str | None = arg(None, help=DEVICE_HELP)

    def __post_init__(self) -> None:
        if self.embedder is not None and ":" not in self.embedder:
            raise ValueError(
                f"--embedder must be `module:callable`, got {self.embedder!r}"
            )
