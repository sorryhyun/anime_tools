"""The stage CLIs' shared argparse blocks, pinned.

``anime_tools.gui.stages`` builds the GUI form by introspecting each stage's
``build_parser()``, so an argparse detail is a UI contract: a dropped ``dest=``
renames a form field, a lost ``--foo-bar`` alias breaks a saved command line,
and a drifted default silently changes what a run does.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pytest

from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT
from anime_tools.masking._sam3 import SUBJECT_PROMPT
from anime_tools.masking.cli import generate_masks, generate_masks_mit
from anime_tools.stages.cli import (
    audit_apply_curated,
    audit_multiview,
    autotag_captions,
    correct_captions,
    position_captions,
    resize_images,
)
from anime_tools.stages.cli._detection import OPTIONAL_FLAGS, detection_options
from anime_tools.stages.instance_detection import DEFAULT_SUBJECT_PROMPT_EMBED
from anime_tools.stages.position_captions import PositionCaptionOptions

SAM3_STAGES = {
    "position": position_captions,
    "audit": audit_multiview,
}
CAPTION_STAGES = {
    "autotag": autotag_captions,
    **SAM3_STAGES,
}
ALL_STAGES = {
    **CAPTION_STAGES,
    "resize": resize_images,
    "correct": correct_captions,
}


# Declared in the detection group but read off the namespace by ``build_detect_fn``
# rather than through the options object.
DETECTOR_ONLY = frozenset({"prompt_embed"})


def actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {a.dest: a for a in parser._actions if a.dest != "help"}


def detection_dests(module) -> set[str]:
    """The dests in a stage parser's ``detection`` argument group."""
    parser = module.build_parser()
    group = next(g for g in parser._action_groups if g.title == "detection")
    return {a.dest for a in group._group_actions}


@pytest.fixture(scope="module")
def parsers() -> dict[str, dict[str, argparse.Action]]:
    return {name: actions(mod.build_parser()) for name, mod in ALL_STAGES.items()}


# ---- one spelling per flag, everywhere it appears ----------------------


@pytest.mark.parametrize(
    ("dest", "flags", "default"),
    [
        ("path_pattern", ("--path_pattern", "--path-pattern"), "*"),
        ("report_dir", ("--report_dir", "--report-dir"), None),
        ("from_report", ("--from_report", "--from-report"), None),
        ("tagger_dir", ("--tagger_dir", "--tagger-dir"), None),
        ("checkpoint", ("--checkpoint",), DEFAULT_SAM3_CHECKPOINT),
        ("prompt_embed", ("--prompt_embed",), DEFAULT_SUBJECT_PROMPT_EMBED),
        ("device", ("--device",), None),
        ("src", ("--src",), "image_dataset"),
        ("dst", ("--dst",), "workspace/resized"),
    ],
)
def test_shared_flags_keep_one_spelling(parsers, dest, flags, default):
    """Every stage that takes one of these takes it identically.

    ``--report_dir``'s default is per-stage, so only its spelling is pinned;
    ``correct_captions`` requires its roots rather than defaulting them.
    """
    seen = 0
    for name, acts in parsers.items():
        if dest not in acts:
            continue
        seen += 1
        action = acts[dest]
        assert tuple(action.option_strings) == flags, (
            f"{name} spells {dest} differently"
        )
        if default is not None and not (name == "correct" and dest in ("src", "dst")):
            assert action.default == default, f"{name} defaults {dest} differently"
    assert seen >= 2, f"{dest} is no longer shared — drop it from this test"


def test_every_caption_stage_is_dry_run_by_default(parsers):
    """Nothing is written without ``--apply``."""
    for name in CAPTION_STAGES:
        apply_action = parsers[name]["apply"]
        assert apply_action.option_strings == ["--apply"]
        assert apply_action.default is False


def test_report_dir_defaults_are_distinct(parsers):
    """Each stage reports into its own directory, so one stage's ``--from_report``
    cannot read another's report."""
    defaults = [parsers[name]["report_dir"].default for name in CAPTION_STAGES]
    assert len(set(defaults)) == len(defaults)


def test_the_curated_apply_reads_the_report_the_audit_writes():
    """``audit_apply_curated``'s ``--report`` default is the audit's
    ``--report_dir`` default plus the file written there."""
    report = actions(audit_apply_curated.build_parser())["report"]
    assert report.default == f"{audit_multiview.DEFAULT_REPORT_DIR}/report.json"


# ---- the two SAM3 stages run the same detector -------------------------


def test_the_two_sam3_stages_declare_identical_detection_flags(parsers):
    """The audit hands its own namespace to the position CLI's ``build_detect_fn``,
    so only :data:`OPTIONAL_FLAGS` and help text may differ.
    """
    position, audit = parsers["position"], parsers["audit"]
    shared = detection_dests(position_captions) & detection_dests(audit_multiview)
    assert shared, "the detection group vanished from one of them"
    for dest in sorted(shared - set(OPTIONAL_FLAGS)):
        a, b = position[dest], audit[dest]
        assert a.option_strings == b.option_strings, f"{dest}: spelling drifted"
        assert a.default == b.default, f"{dest}: default drifted"
        assert a.type == b.type, f"{dest}: type drifted"


def test_the_audit_pins_the_flags_it_does_not_ask_for(parsers):
    """The audit declares none of :data:`OPTIONAL_FLAGS` and takes the dataclass
    defaults; the position stage declares all of them."""
    for dest in OPTIONAL_FLAGS:
        assert dest not in parsers["audit"], f"the audit now declares {dest}"
        assert dest in parsers["position"], f"the position stage lost {dest}"


def test_the_sam3_mask_stages_share_the_catalog_flags():
    """The masking CLIs share the ⚙ Settings stage defaults declared in
    ``masking/_sam3.py``. The text stage takes ``--checkpoint`` alone, since
    ``--prompt_embed`` stands in for the subject phrase only.
    """
    masks, position, text = (
        actions(generate_masks.build_parser()),
        actions(position_captions.build_parser()),
        actions(generate_masks_mit.build_parser()),
    )
    for dest in ("checkpoint", "prompt_embed"):
        assert masks[dest].option_strings == position[dest].option_strings, dest
        assert masks[dest].default == position[dest].default, dest
    assert text["checkpoint"].option_strings == position["checkpoint"].option_strings
    assert text["checkpoint"].default == position["checkpoint"].default
    assert "prompt_embed" not in text


def test_the_sam3_mask_stage_defaults_to_the_phrase_its_soft_prompt_encodes():
    """The default ``--focus-prompts`` is the phrase ``--prompt_embed`` encodes."""
    args = generate_masks.build_parser().parse_args(
        ["--image-dir", "i", "--mask-dir", "m"]
    )
    assert generate_masks.prompt_list(args.focus_prompts) == (SUBJECT_PROMPT,)
    assert args.prompt_embed == DEFAULT_SUBJECT_PROMPT_EMBED


def test_a_prompt_flag_is_a_comma_separated_list():
    assert generate_masks.prompt_list(" speech bubble , text ,") == (
        "speech bubble",
        "text",
    )
    assert generate_masks.prompt_list("") == ()
    # The GUI drops a blank field back to the default, so "none of them" needs a word.
    assert generate_masks.prompt_list("none") == ()


# ---- the flags and the options they build stay in step -----------------


def test_detection_options_reads_every_detection_flag_the_parser_declares():
    """Every detection-group dest reaches the options object; one that does not
    would parse fine and do nothing.
    """
    parser = position_captions.build_parser()
    declared = detection_dests(position_captions) - DETECTOR_ONLY
    built = set(detection_options(parser.parse_args([])))
    assert declared - built == set(), f"declared but never read: {declared - built}"


def _twist(value):
    """A value that is definitely not the one passed in."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return (value + " hips").strip()
    raise AssertionError(f"no twist for {value!r}")


def test_every_option_field_is_wired_to_the_flag_that_names_it():
    """Field by field: twist the namespace value, rebuild, and require the option
    to move — a field that does not is one ``build_options_from_args`` drops.
    """
    parser = position_captions.build_parser()
    dests = {a.dest for a in parser._actions}
    fields = dataclasses.fields(PositionCaptionOptions)
    assert {f.name for f in fields} <= dests, "an option field has no flag naming it"

    for f in fields:
        args = parser.parse_args([])
        before = position_captions.build_options_from_args(args)
        setattr(args, f.name, _twist(getattr(args, f.name)))
        after = position_captions.build_options_from_args(args)
        assert getattr(after, f.name) != getattr(before, f.name), (
            f"--{f.name} never reaches PositionCaptionOptions.{f.name}"
        )


def test_the_audit_builds_its_options_off_the_shared_detection_half():
    """The audit's options come from ``detection_options(args, min_instances=2)``:
    the shared values arrive and the pinned ones win.
    """
    audit_args = audit_multiview.build_parser().parse_args([])
    position_args = position_captions.build_parser().parse_args([])
    shared = detection_options(audit_args, min_instances=2)
    options = PositionCaptionOptions(**shared)

    assert options.min_instances == 2
    assert options.mask_containment_threshold == 0.8
    # Every non-pinned detection value matches the position stage's, since both
    # namespaces come from the same declaration.
    for name, value in detection_options(position_args).items():
        if name in OPTIONAL_FLAGS:
            continue
        assert shared[name] == value, f"{name} differs between the two stages"


def test_the_device_flag_has_no_copies_left():
    """``--device`` is declared only in ``_device.py``.

    It is a :data:`anime_tools.gui.stages.AUTO_FIELDS` dest — neither shown on
    the form nor put on the argv — so the child resolves it through
    :func:`anime_tools._device.resolve_device`, which only works while every CLI
    defaults it to ``None``.
    """
    from anime_tools import _device

    package = Path(_device.__file__).parent
    carriers = sorted(
        p.relative_to(package).as_posix()
        for p in package.rglob("*.py")
        if '"--device"' in p.read_text(encoding="utf-8")
    )
    assert carriers == ["_device.py"]
