"""Merge masks from multiple sources by the pixel-wise minimum (union of masked regions)
— :func:`run_merge_masks` over a :class:`~anime_tools.masking.requests.MergeMasksRequest`.

Keys merges by ``(rel_dir, name)``, so masks at the same relative path across inputs
collide; the nested layout is preserved under ``output_dir``. The CLI
(``cli/merge_masks.py``) is a shell over this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from anime_tools._env import resolve_path
from anime_tools.masking._masks import iter_masks
from anime_tools.masking.requests import MergeMasksRequest


def run_merge_masks(req: MergeMasksRequest) -> int:
    """Returns how many masks were written under ``req.output_dir``."""
    # Home-anchored, so the defaults name the trees the generators wrote however the
    # operator got here.
    mask_dirs = [resolve_path(d) for d in req.mask_dirs]
    output_dir = resolve_path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_rel: dict[tuple[str, str], list[Path]] = {}
    for d in mask_dirs:
        if not d.exists():
            continue
        for rel_str, p in iter_masks(d):
            by_rel.setdefault((rel_str, p.name), []).append(p)

    if not by_rel:
        print("No masks found.")
        return 0

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

    return merged
