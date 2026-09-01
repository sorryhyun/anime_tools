"""``masking/_masks.py``: the plan every generator runs before its model loads.

Both mask generators used to hand-roll the walk → mirror → skip loop, which is
the piece that decides *where a mask lands* and *whether an existing one is
overwritten*. Neither had a test; both wrote to the same layout the merge step
and the GUI read back by convention. These pin that convention from the write
side, so the mirror rule and the ``--force`` semantics are checked rather than
mirrored by eye.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anime_tools.masking._masks import (
    coverage_pct,
    iter_masks,
    mask_name,
    mask_path_for,
    mask_run,
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


# ---- the run scaffolding -----------------------------------------------


def run_args(images: Path, masks: Path, **over) -> argparse.Namespace:
    """The flags ``mask_run`` reads back — every one of them declared in the
    same module, which is the point of reading them there."""
    return argparse.Namespace(
        **{
            "image_dir": str(images),
            "mask_dir": str(masks),
            "recursive": True,
            "path_pattern": None,
            "force": False,
            "workers": 2,
            **over,
        }
    )


def test_the_run_resolves_both_roots_and_plans_the_walk(tree, capsys):
    images, masks = tree
    with mask_run(run_args(images, masks)) as run:
        assert (run.image_dir, run.mask_dir) == (images, masks)
        assert masks.is_dir(), "the output root exists before the first write"
        assert run.total == 4
        assert [p.relative_to(masks).as_posix() for _, p in run.items] == [
            "a_mask.png",
            "artist_x/one_mask.png",
            "artist_x/two_mask.png",
            "artist_y/one_mask.png",
        ]
        # The pool is the run's, so `write_mask(..., pool=run.pool)` is the
        # generators' whole I/O story.
        run.pool.submit(write_mask, run.items[0][1], np.ones((2, 2), np.uint8)).result()
        run.advance()
        run.note(run.items[0][0], "41.2%")

    assert capsys.readouterr().out.strip() == f"Masks saved to {masks}/"


def test_the_run_narrows_by_the_same_walk_flags(tree):
    images, masks = tree
    with mask_run(run_args(images, masks, recursive=False)) as run:
        assert [p.name for _, p in run.items] == ["a_mask.png"]
    with mask_run(run_args(images, masks, path_pattern="artist_y/*")) as run:
        assert [p.name for _, p in run.items] == ["one_mask.png"]


def test_nothing_to_do_is_a_sentence_not_a_directory(tmp_path, capsys):
    """An empty plan is not an error and not an early return: the caller's loop
    runs zero times, and the closing line says so rather than naming a tree
    nothing landed in."""
    images = tmp_path / "images"
    images.mkdir()
    masks = tmp_path / "masks"

    seen = []
    with mask_run(run_args(images, masks)) as run:
        assert run.total == 0
        seen = [item for item in run.items]

    assert seen == []
    assert capsys.readouterr().out.strip() == "No images to process."


def test_a_run_that_raised_does_not_sign_off(tree, capsys):
    """The closing line is past the ``finally``: the pool and the bar still
    close, but a traceback is not followed by "Masks saved to"."""
    images, masks = tree
    with pytest.raises(RuntimeError), mask_run(run_args(images, masks)) as run:
        assert run.total == 4
        raise RuntimeError("model exploded")
    assert capsys.readouterr().out.strip() == ""


def test_coverage_is_the_masks_own_denominator(tmp_path):
    """Polarity-blind and shape-blind: the three call sites mean different
    things by the number, and none of them passes a ``(w, h)`` to transpose."""
    m = np.zeros((4, 8), dtype=np.uint8)  # non-square, so a transpose would show
    assert coverage_pct(m) == 0.0
    m[:, :2] = 1
    assert coverage_pct(m) == pytest.approx(25.0)
    assert coverage_pct(1 - m) == pytest.approx(75.0)


# ---- the drawer seam ---------------------------------------------------


def test_a_gated_group_names_its_switch_where_the_gui_reads_it():
    """``gated_group`` and ``gui.stages.fields_of`` are one fact in two files.

    The GUI draws a *drawer* — a group folded away behind one checkbox — off an
    attribute stamped on the argparse group, and ``gui/stages.py`` spells that
    attribute rather than importing it, so it stays free of every stage's
    dependencies. Two spellings would mean a group that renders as a flat
    fieldset with a stray boolean in it, which is not an error anywhere.
    """
    import argparse

    from anime_tools.gui import stages as S
    from anime_tools.masking._masks import GATE_ATTR, gated_group

    assert S.GATE_ATTR == GATE_ATTR

    parser = argparse.ArgumentParser()
    group = gated_group(parser, "T", gate="use_thing", default=False, help="h")
    assert getattr(group, GATE_ATTR) == "use_thing"
    group.add_argument("--knob", type=int, default=1)

    fields = {f.dest: f for f in S.fields_of(parser)}
    # The gate names itself, which is how the form tells the switch from what it
    # switches; everything else in the group names the gate.
    assert fields["use_thing"].gate == "use_thing"
    assert fields["knob"].gate == "use_thing"
    assert fields["use_thing"].negate == "--no-use-thing"


def test_the_text_stage_runs_two_detectors_behind_two_switches():
    """The panel's shape *is* the CLI's: one drawer per detector, and the knobs
    that only mean something while it runs live inside it. A flag that drifted
    out of its group would show up on the form whether or not its detector was
    on."""
    from anime_tools.masking.cli import generate_masks_mit as mit

    parser = mit.build_parser()
    gated = {
        a.dest
        for g in parser._action_groups
        for a in g._group_actions
        if getattr(g, "gui_gate", None)
    }
    by_gate: dict[str, set[str]] = {}
    for g in parser._action_groups:
        gate = getattr(g, "gui_gate", None)
        if gate:
            by_gate[gate] = {a.dest for a in g._group_actions}

    assert by_gate["use_sam"] == {
        "use_sam",
        "sam_prompts",
        "sam_threshold",
        "checkpoint",
    }
    assert by_gate["use_mit"] == {"use_mit", "model_path", "text_threshold", "ctd_gate"}
    # The walk, the dilation and the output tree belong to neither detector.
    assert not gated & {"image_dir", "mask_dir", "dilate", "recursive", "path_pattern"}

    args = parser.parse_args(["--image-dir", "i"])
    # The segmenter is the stage's historical behaviour and stays on; SAM3 is a
    # second set of weights, so it is opt-in — with the one prompt it is for
    # already typed into the drawer.
    assert (args.use_mit, args.ctd_gate) == (True, True)
    assert args.use_sam is False
    assert mit.prompt_list(args.sam_prompts) == ("speech bubble",)


def test_the_text_stage_refuses_a_run_with_no_detector():
    """Both drawers shut is a run that loads nothing, walks the tree and writes
    nothing — the one shape of this form that means "do not run me"."""
    from anime_tools.masking.cli import generate_masks_mit as mit

    parser = mit.build_parser()

    def check(argv):
        return mit.detectors(parser, parser.parse_args(["--image-dir", "i", *argv]))

    assert check([]) == (True, ())
    assert check(["--use-sam"]) == (True, ("speech bubble",))
    assert check(["--no-use-mit", "--use-sam", "--sam-prompts", "text,sign"]) == (
        False,
        ("text", "sign"),
    )
    with pytest.raises(SystemExit):
        check(["--no-use-mit"])
    # `none` is how a prompt field says "none of them" — the GUI would send a
    # blank one back as its default.
    with pytest.raises(SystemExit):
        check(["--use-sam", "--sam-prompts", "none"])
