"""``anime_tools._boxes`` — the one box geometry — and the old names that
re-export it."""

from __future__ import annotations

import pytest

from anime_tools import _boxes as B


def test_area_iou_and_the_two_containments():
    outer, inner, apart = (0, 0, 10, 10), (2, 2, 4, 6), (20, 20, 30, 30)
    assert B.area(outer) == 100 and B.area((5, 5, 1, 1)) == 0
    assert B.intersection(outer, inner) == 8
    assert B.iou(outer, inner) == pytest.approx(8 / 100)
    assert B.iou(outer, apart) == 0.0
    # over the smaller box, either way round
    assert B.containment(outer, inner) == B.containment(inner, outer) == 1.0
    # over the first box's own area
    assert B.inside(inner, outer) == 1.0
    assert B.inside(outer, inner) == pytest.approx(0.08)
    # a trailing score is ignored; a degenerate box divides by nothing
    assert B.inside((2, 2, 4, 6, 0.9), outer) == 1.0
    assert B.containment((3, 3, 3, 3), outer) == 0.0
    assert B.iou((3, 3, 3, 3), (3, 3, 3, 3)) == 0.0


def test_the_old_names_are_the_same_geometry():
    from anime_tools.ocr import animetext, reread
    from anime_tools.stages import position_captions as pc

    a, b = (0, 0, 10, 10), (5, 0, 15, 10)
    assert pc.box_iou(a, b) == B.iou(a, b)
    assert pc.box_area(a) == B.area(a)
    assert pc.box_containment(a, b) == B.containment(a, b)
    assert reread.overlap(a, b) == (B.iou(a, b), B.containment(a, b))
    assert animetext.containment(a, b) == B.inside(a, b)
    assert reread.NESTED_CONTAINMENT == animetext.NEST_TH
