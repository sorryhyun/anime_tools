"""Per-image analysis beside a SAM3 stage's report: what the stage saw in one
image, kept after the run that saw it.

``report.json`` is one run's rows, and a GUI run scoped to one image rewrites it
with that image alone, so it cannot answer "what did the last run say about
*this* image". Each image the position sweep (or its audit phase) looks at
instead leaves two files under ``<report_dir>/analysis/``, mirroring the resized
tree by image stem:

- ``<stem>.json`` — the stage's row for the image (the proposal, or the audit's
  finding) plus ``labels``, which row list the mask's indices refer to.
- ``<stem>.png`` — the instance label map: 8-bit L at the resized image's size,
  ``0`` background and ``i + 1`` where instance ``i`` is.

An image the next run walks and has nothing to say about loses both
(:func:`clear_analysis`), so the pair is always the latest word on the image
rather than the last word anyone had. Torch-free; numpy and PIL are imported
only by the writer.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from anime_tools._json import write_json

__all__ = ["ANALYSIS_SUBDIR", "analysis_paths", "clear_analysis", "write_analysis"]

ANALYSIS_SUBDIR = "analysis"


def analysis_paths(root: Path, image_rel: str) -> tuple[Path, Path]:
    """``(json, png)`` for the image at ``image_rel`` (posix, relative to the
    resized tree) under the analysis tree ``root``."""
    stem = PurePosixPath(image_rel).with_suffix("")
    base = root.joinpath(*stem.parts)
    return base.with_name(base.name + ".json"), base.with_name(base.name + ".png")


def clear_analysis(root: Path | None, image_rel: str) -> None:
    if root is None:
        return
    for path in analysis_paths(root, image_rel):
        path.unlink(missing_ok=True)


def write_analysis(
    root: Path | None,
    image_rel: str,
    record: dict[str, Any],
    detections: Sequence[Any],
    size: tuple[int, int],
) -> None:
    """Write one image's record and the label map of ``detections`` (anything
    with a ``box`` and an optional boolean ``mask``), in the order the record's
    ``labels`` list names. A detection without a mask paints its box, so every
    index in the record has pixels behind it."""
    if root is None:
        return
    import numpy as np
    from PIL import Image

    w, h = size
    label = np.zeros((h, w), dtype=np.uint8)
    for index, det in enumerate(detections[:255]):
        mask = getattr(det, "mask", None)
        if mask is not None:
            region = np.asarray(mask)
            if region.ndim == 3:
                region = region[0]
            region = region > 0.5 if region.dtype != bool else region
            if region.shape != (h, w):
                region = (
                    np.asarray(
                        Image.fromarray(region.astype(np.uint8) * 255).resize(
                            (w, h), Image.NEAREST
                        )
                    )
                    > 127
                )
        else:
            x0, y0, x1, y1 = (round(v) for v in det.box)
            region = np.zeros((h, w), dtype=bool)
            region[max(y0, 0) : max(y1, 0), max(x0, 0) : max(x1, 0)] = True
        label[region] = index + 1
    json_path, png_path = analysis_paths(root, image_rel)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label).save(png_path)
    write_json(json_path, {"image": image_rel, **record})
