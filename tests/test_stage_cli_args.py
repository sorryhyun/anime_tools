"""The stage CLIs' shared flags, pinned.

Every parser is generated from its request class (``anime_tools._request``),
so a flag's spelling, default and group live in one field's metadata — and the
GUI form is drawn from the same list. What this file pins is the *shared*
part: a dest that several stages take is spelled and defaulted identically, so
one ⚙ Settings value can fill it everywhere.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pytest

from anime_tools._request import args_of
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
from anime_tools.stages.instance_detection import DEFAULT_SUBJECT_PROMPT_EMBED
from anime_tools.stages.position_captions import PositionCaptionOptions
from anime_tools.stages.requests import (
    DETECTION,
    POSITION_ONLY_FLAGS,
    AuditRequest,
    DetectionRequest,
    PositionRequest,
)

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


# The audit's detection group carries ``name_confidence``, which the position
# stage declares under clause composition: a dest, not a detector knob.
AUDIT_GROUP_EXTRAS = frozenset({"name_confidence"})


def actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {a.dest: a for a in parser._actions if a.dest != "help"}


def detection_dests(cls) -> set[str]:
    """The dests a stage request puts in its ``detection`` group."""
    return {a.name for a in args_of(cls) if a.group == DETECTION}


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
        (
            "prompt_embed",
            ("--prompt_embed", "--prompt-embed"),
            DEFAULT_SUBJECT_PROMPT_EMBED,
        ),
        ("device", ("--device",), None),
        ("src", ("--src",), "image_dataset"),
        ("dst", ("--dst",), "workspace/resized"),
    ],
)
def test_shared_flags_keep_one_spelling(parsers, dest, flags, default):
    """Every stage that takes one of these takes it identically: the canonical
    spelling first, the other separator as an alias.

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


def test_every_flag_takes_both_separators():
    """A flag with a separator in it is accepted either way, so the caption
    stages' underscores and the masking CLIs' hyphens are one convention."""
    for name, mod in {**ALL_STAGES, "masks": generate_masks}.items():
        for a in actions(mod.build_parser()).values():
            canon = a.option_strings[0] if a.option_strings else ""
            if "_" in canon or "-" in canon[2:]:
                assert len(a.option_strings) >= 2, (name, canon)
                body = canon[2:]
                other = "--" + (
                    body.replace("_", "-") if "_" in body else body.replace("-", "_")
                )
                assert other in a.option_strings, (name, canon)


def test_every_caption_stage_is_dry_run_by_default(parsers):
    """Nothing is written without ``--apply``."""
    for name in CAPTION_STAGES:
        apply_action = parsers[name]["apply"]
        assert apply_action.option_strings[0] == "--apply"
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


def test_the_help_is_the_field_metadata():
    """``--help`` prints what the request field says, ``%`` included, and the
    parser's description is the class docstring."""
    parser = position_captions.build_parser()
    acts = actions(parser)
    novel = next(a for a in args_of(PositionRequest) if a.name == "max_novel_tags")
    assert "46% novel" in novel.help
    assert acts["max_novel_tags"].help == novel.help.replace("%", "%%")
    assert "46% novel" in parser.format_help()
    assert parser.description.startswith("Rewrite multi-subject captions")


# ---- the two SAM3 stages run the same detector -------------------------


def test_the_two_sam3_stages_declare_identical_detection_flags(parsers):
    """Both read their detection group into one ``DetectionRequest``, so only
    :data:`POSITION_ONLY_FLAGS` and help text may differ.
    """
    position, audit = parsers["position"], parsers["audit"]
    shared = detection_dests(PositionRequest) & detection_dests(AuditRequest)
    assert shared, "the detection group vanished from one of them"
    for dest in sorted(shared - set(POSITION_ONLY_FLAGS)):
        a, b = position[dest], audit[dest]
        assert a.option_strings == b.option_strings, f"{dest}: spelling drifted"
        assert a.default == b.default, f"{dest}: default drifted"
        assert a.type == b.type, f"{dest}: type drifted"


def test_the_audit_pins_the_flags_it_does_not_ask_for(parsers):
    """The audit declares none of :data:`POSITION_ONLY_FLAGS` and takes the
    dataclass defaults; the position stage declares all of them."""
    for dest in POSITION_ONLY_FLAGS:
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


# ---- the flags and the request that reads them stay in step -------------


def test_the_detection_group_is_exactly_the_detection_request():
    """The ``detection`` group of either SAM3 stage is the nested
    ``DetectionRequest`` block, plus the stage's own detection knobs
    (:data:`POSITION_ONLY_FLAGS`, the audit's ``name_confidence``)."""
    request_fields = {f.name for f in dataclasses.fields(DetectionRequest)}
    assert detection_dests(PositionRequest) - set(POSITION_ONLY_FLAGS) == request_fields
    assert detection_dests(AuditRequest) - AUDIT_GROUP_EXTRAS == request_fields


def test_every_option_field_has_a_flag_naming_it():
    """A ``PositionCaptionOptions`` field is a position-stage dest (its wiring
    is pinned request-side in ``test_stage_requests``)."""
    dests = {a.dest for a in position_captions.build_parser()._actions}
    assert {f.name for f in dataclasses.fields(PositionCaptionOptions)} <= dests


def test_the_audit_builds_its_options_off_the_shared_detection_half():
    """The audit's options are the position stage's detector verbatim with
    ``min_instances`` pinned, whichever parser the flags came through."""
    audit_req = AuditRequest.from_namespace(
        audit_multiview.build_parser().parse_args([])
    )
    position_req = position_captions.PositionRequest.from_namespace(
        position_captions.build_parser().parse_args([])
    )
    options = audit_req.options()
    assert options.min_instances == 2
    assert options.mask_containment_threshold == 0.8
    assert audit_req.detection == position_req.detection
    for name in (f.name for f in dataclasses.fields(DetectionRequest)):
        if name in ("checkpoint", "prompt_embed"):
            continue
        assert getattr(options, name) == getattr(position_req.options(), name), name


def test_the_device_flag_has_no_copies_left():
    """``--device`` is declared only in ``_device.py``.

    It is a :data:`anime_tools.gui.stages.AUTO_FIELDS` dest — neither shown on
    the form nor put on the argv — so the child resolves it through
    :func:`anime_tools._device.resolve_device`, which only works while every
    stage defaults it to ``None``. The request classes spell it as a field
    (``device: str | None = arg(None, help=DEVICE_HELP)``), never as a flag.
    """
    from anime_tools import _device

    package = Path(_device.__file__).parent
    carriers = sorted(
        p.relative_to(package).as_posix()
        for p in package.rglob("*.py")
        if '"--device"' in p.read_text(encoding="utf-8")
    )
    assert carriers == ["_device.py"]
