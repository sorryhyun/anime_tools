"""The OCR package's one image decode.

``cv2.imdecode`` over ``np.fromfile`` rather than ``cv2.imread``: the latter
cannot open a non-ASCII path on Windows, and dataset folders are often named in
Japanese. cv2 and numpy load inside the call, so importing this stays cheap.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["imread"]


def imread(path: str | Path, *, gray: bool = False):
    """``path`` decoded as BGR ``uint8`` (``(H, W)`` single-channel with
    ``gray``), or ``None`` when OpenCV cannot decode it."""
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
