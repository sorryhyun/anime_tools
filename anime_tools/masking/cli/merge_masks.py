"""Merge masks from multiple sources by taking the pixel-wise minimum (union of masked regions).

Walks each input mask directory recursively and keys merges by
``(rel_dir, name)`` so masks at the same relative path across inputs collide.
The output preserves the same nested layout under ``--output-dir``.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from anime_tools import workspace as WS
from anime_tools._env import resolve_path
from anime_tools.masking._masks import iter_masks

DEFAULT_INPUTS = [WS.MASKS_SAM, WS.MASKS_MIT]
"""The two generators' own ``--mask-dir`` defaults, in the order they run.

A merge whose inputs are not the trees the generators wrote merges nothing, so
the three paths are declared once in :mod:`anime_tools.workspace` and read back
here. A missing input directory is skipped, not an error — running only one
generator is a valid half of this.
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
    args = build_parser().parse_args()

    # Home-anchored, so the defaults name the trees the generators wrote
    # however the operator got here.
    mask_dirs = [resolve_path(d) for d in args.mask_dirs]
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_rel: dict[tuple[str, str], list[Path]] = {}
    for d in mask_dirs:
        if not d.exists():
            continue
        for rel_str, p in iter_masks(d):
            by_rel.setdefault((rel_str, p.name), []).append(p)

    if not by_rel:
        print("No masks found.")
        return

    merged = 0
    for (rel_str, name), sources in tqdm(sorted(by_rel.items()), desc="Merging masks"):
        if len(sources) == 1:
            arr = np.array(Image.open(sources[0]))
        else:
            # lower alpha = more masking
            arr = np.array(Image.open(sources[0]))
            for src in sources[1:]:
                other = np.array(
                    Image.open(src).resize((arr.shape[1], arr.shape[0]), Image.NEAREST)
                )
                arr = np.minimum(arr, other)

        target_dir = output_dir / rel_str if rel_str else output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, mode="L").save(target_dir / name)
        merged += 1

    print(f"Merged {merged} masks into {output_dir}/")


if __name__ == "__main__":
    main()
