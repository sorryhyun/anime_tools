"""Group dataset images by PE-Spatial visual similarity → groups.json manifest.

A curation tool, not a preprocess/training step: it clusters near-identical
images per artist so the GUI Dataset tab can filter by group, and writes nothing
else. Two images group when ``match_frac >= --match-frac-min`` at per-cell floor
``--cell-match-min``. Re-runs reuse the shared PE-Spatial feature cache, so
retuning the thresholds is cheap.
"""

import argparse

from anime_tools import workspace as WS
from anime_tools._device import add_device_arg
from anime_tools.grouping.groups import (
    DEFAULT_CELL_MATCH_MIN,
    DEFAULT_GRID,
    DEFAULT_MATCH_FRAC_MIN,
    DEFAULT_RATIO,
    DEFAULT_SIM_MIN,
)
from anime_tools.grouping.requests import DEFAULT_EMBEDDER, GroupRequest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-dir",
        default=WS.RESIZED,
        help=(
            "Image tree to group. Defaults to the resized tree, which is the "
            "pixel data every other stage reads (and what training sees)"
        ),
    )
    p.add_argument(
        "--out",
        default=f"{WS.GROUPS}/groups.json",
        help="Manifest path the GUI Dataset tab reads",
    )
    p.add_argument(
        "--cell-match-min",
        type=float,
        default=DEFAULT_CELL_MATCH_MIN,
        help="per-cell cosine for an inlier grid-cell match",
    )
    p.add_argument(
        "--match-frac-min",
        type=float,
        default=DEFAULT_MATCH_FRAC_MIN,
        help="inlier fraction to connect two images (higher = tighter)",
    )
    p.add_argument(
        "--sim-min",
        type=float,
        default=DEFAULT_SIM_MIN,
        help="Stage-A CLS-cosine prefilter (loose; the grid match is the gate)",
    )
    p.add_argument(
        "--grid", type=int, default=DEFAULT_GRID, help="pooled grid edge (G×G cells)"
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help="ratio-test distinctiveness (lower = stricter)",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="drop groups smaller than this (1 keeps singletons)",
    )
    p.add_argument(
        "--embedder",
        default=None,
        help=(
            "dotted factory `module:callable(device=...)` returning a grouping "
            f"Embedder (default: the package's PE-Spatial, `{DEFAULT_EMBEDDER}`)."
        ),
    )
    p.add_argument("--batch-size", type=int, default=16, help="embed batch size")
    p.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader image-decode workers"
    )
    add_device_arg(p)
    return p


def main(argv: list[str] | None = None) -> None:
    from anime_tools.grouping.groups import run_groups

    run_groups(GroupRequest.from_argv(build_parser(), argv))


if __name__ == "__main__":
    main()
