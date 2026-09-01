"""``masking/_masks.py``: the plan every generator runs before its model loads.

Both mask generators used to hand-roll the walk → mirror → skip loop, which is
the piece that decides *where a mask lands* and *whether an existing one is
overwritten*. Neither had a test; both wrote to the same layout the merge step
and the GUI read back by convention. These pin that convention from the write
side, so the mirror rule and the ``--force`` semantics are checked rather than
mirrored by eye.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anime_tools.masking._masks import (
    iter_masks,
    mask_name,
    mask_path_for,
    plan_mask_jobs,
    write_ignore_mask,
    write_mask,
)


def make_image(path: Path, size: tuple[int, int] = (8, 6)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (128, 128, 128)).save(path)
    return path


@pytest.fixture
def tree(tmp_path: Path) -> tuple[Path, Path]:
    """Two artists plus a root-level image — the shape a real dataset has."""
    images = tmp_path / "images"
    make_image(images / "a.png")
    make_image(images / "artist_x" / "one.png")
    make_image(images / "artist_x" / "two.jpg")
    make_image(images / "artist_y" / "one.png")  # same stem, other folder
    return images, tmp_path / "masks"


def test_mask_path_mirrors_the_source_subdir(tree):
    images, masks = tree
    assert mask_path_for(images / "artist_x" / "one.png", images, masks) == (
        masks / "artist_x" / "one_mask.png"
    )
    # A root-level image lands flat, not under a "." directory.
    assert mask_path_for(images / "a.png", images, masks) == masks / "a_mask.png"
    assert mask_name("one") == "one_mask.png"


def test_plan_covers_the_tree_and_creates_its_directories(tree):
    images, masks = tree
    jobs = plan_mask_jobs(images, masks, recursive=True)
    assert [p.relative_to(masks).as_posix() for _, p in jobs] == [
        "a_mask.png",
        "artist_x/one_mask.png",
        "artist_x/two_mask.png",
        "artist_y/one_mask.png",
    ]
    # Same stem in two folders is not a collision — the mirror disambiguates.
    assert len({p for _, p in jobs}) == len(jobs)
    # Every output directory exists, so a caller's write loop is a plain save.
    assert all(p.parent.is_dir() for _, p in jobs)


def test_plan_is_flat_without_recursive(tree):
    images, masks = tree
    jobs = plan_mask_jobs(images, masks)
    assert [p.name for _, p in jobs] == ["a_mask.png"]


def test_plan_honours_the_path_pattern(tree):
    images, masks = tree
    jobs = plan_mask_jobs(images, masks, recursive=True, pattern="artist_y/*")
    assert [p.relative_to(masks).as_posix() for _, p in jobs] == [
        "artist_y/one_mask.png"
    ]


def test_existing_masks_are_skipped_unless_forced(tree):
    images, masks = tree
    done = mask_path_for(images / "artist_x" / "one.png", images, masks)
    done.parent.mkdir(parents=True, exist_ok=True)
    write_mask(done, np.ones((6, 8), dtype=np.uint8))

    kept = plan_mask_jobs(images, masks, recursive=True)
    assert done not in {p for _, p in kept}

    forced = plan_mask_jobs(images, masks, recursive=True, force=True)
    assert done in {p for _, p in forced}


def test_ignore_mask_inverts_the_detection(tmp_path: Path):
    detected = np.zeros((4, 4), dtype=np.uint8)
    detected[1:3, 1:3] = 1
    out = tmp_path / "x_mask.png"
    write_ignore_mask(out, detected)

    arr = np.array(Image.open(out))
    assert Image.open(out).mode == "L"
    # detected=1 -> alpha=0 (ignored in the loss); elsewhere 255 (trained on).
    assert arr[1, 1] == 0 and arr[2, 2] == 0
    assert arr[0, 0] == 255 and arr[3, 3] == 255


def test_iter_masks_keys_by_relative_dir(tree):
    images, masks = tree
    for image_path, mask_path in plan_mask_jobs(images, masks, recursive=True):
        write_mask(mask_path, np.ones((2, 2), dtype=np.uint8))
        assert image_path.exists()

    assert [(rel, p.name) for rel, p in iter_masks(masks)] == [
        ("", "a_mask.png"),
        ("artist_x", "one_mask.png"),
        ("artist_x", "two_mask.png"),
        ("artist_y", "one_mask.png"),
    ]


def test_the_merge_reads_exactly_what_the_two_generators_write():
    """One fact split across three CLIs.

    Each generator writes its *own* tree — sharing one would have the second
    run overwrite the first, since both name a mask ``{stem}_mask.png`` at the
    same relative path — and ``merge_masks`` unions them into the ``masks``
    root. So the merge's default inputs have to be the generators' default
    outputs; they are all written in terms of :mod:`anime_tools.workspace`, and
    this is the pairing that keeps them there.
    """
    from anime_tools import workspace as WS
    from anime_tools.masking.cli import generate_masks, generate_masks_mit, merge_masks

    def default(module, dest):
        return next(a.default for a in module.build_parser()._actions if a.dest == dest)

    sam = default(generate_masks, "mask_dir")
    mit = default(generate_masks_mit, "mask_dir")
    assert sam != mit, "the two generators would overwrite each other"
    assert default(merge_masks, "mask_dirs") == [sam, mit]
    assert default(merge_masks, "output_dir") == WS.MASKS
    assert WS.MASKS not in (sam, mit), "a generator writes the merged root"
