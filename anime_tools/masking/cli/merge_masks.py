"""Merge masks from multiple sources by taking the pixel-wise minimum (union of masked regions).

Keys merges by ``(rel_dir, name)``, so masks at the same relative path across inputs
collide; the nested layout is preserved under ``--output-dir``.
"""

import argparse

from anime_tools import workspace as WS
from anime_tools.masking.requests import MergeMasksRequest

DEFAULT_INPUTS = list(MergeMasksRequest.mask_dirs)
"""The two generators' own ``--mask-dir`` defaults, in the order they run.

A missing input directory is skipped, not an error — running only one generator is a
valid half of this.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mask_dirs",
        nargs="*",
        default=DEFAULT_INPUTS,
        help=f"Input mask directories to merge (default: {' '.join(DEFAULT_INPUTS)})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=WS.MASKS,
        help=f"Output directory for merged masks (default: {WS.MASKS})",
    )
    return parser


def main() -> None:
    from anime_tools.masking.merge import run_merge_masks

    run_merge_masks(MergeMasksRequest.from_namespace(build_parser().parse_args()))


if __name__ == "__main__":
    main()
