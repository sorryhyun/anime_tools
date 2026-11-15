"""Axis-aligned box geometry — the one spelling of area, IoU and containment.

Stdlib-only leaf. A box is any ``(x0, y0, x1, y1, …)`` sequence (a trailing
score is ignored); ints and floats both work. Every ratio here is ``0.0`` when
its denominator is empty, so a degenerate box never divides by zero.

Two containments, because the callers ask two questions:

* :func:`containment` — intersection over the *smaller* box: how nested a pair
  is, whichever way round (the subject NMS, the OCR line dedupe).
* :func:`inside` — intersection over ``inner``'s own area: how much of one
  given box sits in another (the AnimeText block-and-columns denest).
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["area", "containment", "inside", "intersection", "iou"]


def area(box: Sequence[float]) -> float:
    """Width × height, a negative extent counting as zero."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a: Sequence[float], b: Sequence[float]) -> float:
    """Area of the overlap of ``a`` and ``b``."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union."""
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the *smaller* box — how nested the pair is.

    IoU is blind to nesting: a box wholly inside another scores tiny on IoU
    while scoring 1.0 here. Symmetric in ``a`` and ``b``.
    """
    smallest = min(area(a), area(b))
    return intersection(a, b) / smallest if smallest > 0 else 0.0


def inside(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Share of ``inner``'s area that lies within ``outer``."""
    own = area(inner)
    return intersection(inner, outer) / own if own > 0 else 0.0
