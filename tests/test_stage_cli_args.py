"""The stage CLIs' shared argparse blocks, pinned.

``anime_tools.gui.stages`` builds the GUI form by introspecting each stage's
``build_parser()``, so an argparse detail is a UI contract: a dropped ``dest=``
renames a form field, a lost ``--foo-bar`` alias breaks a saved command line,
and a drifted default silently changes what a run does. Before
``stages/cli/_args.py`` and ``_detection.py`` existed those declarations were
retyped once per parser — and the multiview audit's copy had already lost every
long-form alias, most of its help, and (a live bug) a whole option field.

These tests are the standing version of the before/after schema snapshot that
guarded the consolidation: they assert the shared blocks still produce one
spelling everywhere, and that the flags and the options they build stay in step.
"""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT
from anime_tools.stages.cli import (
    audit_multiview,
    autotag_captions,
    correct_captions,
    position_captions,
    resize_images,
)
from anime_tools.stages.cli._detection import OPTIONAL_FLAGS, detection_options
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


# Declared in the detection group but read off the namespace by
# ``build_detect_fn`` rather than through the options object: the soft prompt is
# a tensor the detector loads, not a threshold the pipeline runs under.
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
        ("device", ("--device",), None),
        ("src", ("--src",), "image_dataset"),
        ("dst", ("--dst",), "post_image_dataset/resized"),
    ],
)
def test_shared_flags_keep_one_spelling(parsers, dest, flags, default):
    """Every stage that takes one of these takes it identically.

    ``--report_dir``'s default is per-stage (it names the stage's own report
    directory), so only its spelling is pinned; ``correct_captions`` requires
    its roots rather than defaulting them, and is exempt on that one point.
    ``--checkpoint`` is declared once in ``masking/_sam3.py``, beside the
    ``load_sam3`` it feeds, so the download button and every loader name the
    same file.
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
    """``--apply`` is the whole safety model: nothing is written without it."""
    for name in CAPTION_STAGES:
        apply_action = parsers[name]["apply"]
        assert apply_action.option_strings == ["--apply"]
        assert apply_action.default is False


def test_report_dir_defaults_are_distinct(parsers):
    """Each stage reports into its own directory — a shared default would have
    one stage's ``--from_report`` replay read another's report."""
    defaults = [parsers[name]["report_dir"].default for name in CAPTION_STAGES]
    assert len(set(defaults)) == len(defaults)


# ---- the two SAM3 stages run the same detector -------------------------


def test_the_two_sam3_stages_declare_identical_detection_flags(parsers):
    """The audit imports ``build_detect_fn`` from the position CLI and hands it
    its own namespace, so a flag that differs between them is a detector that
    behaves differently for no stated reason. Only the flags one stage pins for
    itself (:data:`OPTIONAL_FLAGS`) and its help text may differ.
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
    """``min_instances`` is what the audit *is* (two subjects on a `1girl`
    caption), and it never blanks crops or relaxes a strict count — so it
    declares none of the three and takes the dataclass default."""
    for dest in OPTIONAL_FLAGS:
        assert dest not in parsers["audit"], f"the audit now declares {dest}"
        assert dest in parsers["position"], f"the position stage lost {dest}"


# ---- the flags and the options they build stay in step -----------------


def test_detection_options_reads_every_detection_flag_the_parser_declares():
    """A knob added to ``add_detection_args`` but not to ``detection_options``
    parses fine and does nothing — the failure mode this pairing exists to make
    impossible. Every detection-group dest must reach the options object.
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
    """The two halves together cover the whole dataclass.

    Field by field: change the namespace value, rebuild, and require the option
    to move. A field that does not move is one ``build_options_from_args``
    stopped passing through — the flag still parses, the run still succeeds,
    and the knob silently does nothing.
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
    """``detection_options(args, min_instances=2, …)`` — the audit's whole
    options construction. It used to be an inline rebuild that had already
    dropped ``mask_containment_threshold``; this pins that the shared values
    arrive and the pinned ones win.
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
