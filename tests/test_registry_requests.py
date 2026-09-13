"""Every registered stage's request object is its CLI's flags, both ways.

Driven by ``stages/registry.py``: a stage added there without a case here
fails, and every case is run through both the shell's ``build_parser()`` and
the class's own ``Request.parser()``, which must be the same parser. The
package-specific pins (what a default argv spells, validation, the detection
block, the workspace defaults) stay in ``test_stage_requests.py`` and
``test_masking_requests.py``.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys

import pytest

from anime_tools._request import Request, args_of
from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT, DEFAULT_SUBJECT_PROMPT_EMBED
from anime_tools.grouping.requests import GroupRequest
from anime_tools.masking.requests import MaskPrompt, MergeMasksRequest, SamMaskRequest
from anime_tools.stages.registry import BY_ID, STAGES, Stage
from anime_tools.stages.requests import (
    AuditRequest,
    AutotagRequest,
    CorrectRequest,
    DetectionRequest,
    DropGroupRequest,
    ExportRequest,
    MultiviewRequest,
    OcrRequest,
    PositionRequest,
    ResizeRequest,
)

DETECTION = DetectionRequest(
    prompt="woman",
    prompt_embed="none",
    checkpoint="sam.pt",
    score_threshold=0.6,
    retry_score_threshold=0.3,
    part_prompts=("hips", "thighs"),
    part_score_threshold=0.4,
    part_containment_threshold=0.6,
    iou_threshold=0.5,
    containment_threshold=0.9,
    mask_containment_threshold=1.5,
    dedupe_fill_ratio=0.0,
    min_area_frac=0.01,
    pad=0.1,
    row_tol=0.3,
    max_instances=4,
)

MULTIVIEW = MultiviewRequest(
    multiview_threshold=0.4,
    identity_confidence=0.8,
    suggest_counts=True,
    apply_verdicts=("multiple views", "extra-character"),
    apply_confidence=("strong", "weak"),
    sheets=False,
)

CASES: dict[str, Request] = {
    "resize": ResizeRequest(
        src="s",
        dst="d",
        path_pattern="a/*",
        target_res=(1024, 768),
        min_pixels=0,
        recursive=False,
        overwrite=True,
        workers=1,
        resize_crop_anchor="top",
        resize_crop_margins=(1.0, 2.0, 3.0, 4.0),
        freefit_max_ratio=2.0,
        report_dir="rep",
    ),
    "autotag": AutotagRequest(
        src="s",
        dst="d",
        path_pattern="a/*",
        mode="merge",
        min_confidence=0.4,
        apply=True,
        report_dir="rep",
        tagger_dir="ckpt",
        device="cpu",
    ),
    "position": PositionRequest(
        src="s",
        apply=True,
        report_dir="rep",
        crops=True,
        tagger_dir="ckpt",
        device="cpu",
        detection=DETECTION,
        blank_crops=False,
        min_instances=3,
        strict_count=False,
        max_clause_tags=6,
        max_novel_tags=0,
        name_confidence=0.7,
        allow_unlisted_names=True,
        discriminative_only=False,
        bag_gated_identity=False,
        multi_view_gate=False,
        bind_view_anatomy=False,
        bind_framing=False,
        rewrite=False,
        bag_relax=1.0,
        bag_word_relax=1.0,
        bag_relax_min_score=0.0,
        attribution_margin=0.0,
        qwen3="q",
        max_tokens=256,
    ),
    "correct": CorrectRequest(
        src="s",
        dst="d",
        tag_csv="t.csv",
        path_pattern="a/*",
        recursive=True,
        caption_insert_no_artist=True,
        caption_trigger_word="trig",
        caption_trigger_at_front=True,
        caption_drop_groups="artist,lighting",
        no_correct=True,
        caption_shuffle_variants=3,
        caption_tag_dropout_rate=0.1,
        caption_tag_randomize_rate=0.2,
        qwen3="q",
        t5_tokenizer_path="t5",
    ),
    "drop_groups": DropGroupRequest(
        src="s",
        dst="d",
        path_pattern="a/*",
        groups=("artist", "clothing"),
        keep_tags=("@trig",),
        category_paths=("효과/연출 > 날씨",),
        tag_csv="t.csv",
        recursive=False,
        apply=True,
        report_dir="rep",
    ),
    "audit": AuditRequest(
        dst="d",
        apply=True,
        crops=True,
        detection=DETECTION,
        multiview=MULTIVIEW,
        name_confidence=0.6,
    ),
    "ocr": OcrRequest(
        dst="d",
        ocr_dir="o",
        path_pattern="a/*",
        min_chars=1,
        skip_en=False,
        strip_symbols=False,
        det_conf=0.4,
        min_det=0.3,
        min_score=0.2,
        min_box_px=8,
        max_boxes=16,
        mask_dir="m",
        comp_min_side=24,
        comp_max=4,
        vl_batch_size=2,
        apply=True,
        report_dir="rep",
        device="cpu",
    ),
    "groups": GroupRequest(
        source_dir="s",
        out="g.json",
        cell_match_min=0.9,
        match_frac_min=0.3,
        sim_min=0.4,
        grid=5,
        ratio=0.7,
        min_size=1,
        embedder="mod:factory",
        batch_size=8,
        num_workers=0,
        device="cpu",
    ),
    "masks_sam": SamMaskRequest(
        image_dir="i",
        mask_dir="m",
        masks=(
            MaskPrompt("ignore", "text", "speech bubble"),
            MaskPrompt("keep", "soft", "C:/soft/girl.safetensors"),
        ),
        threshold=0.7,
        dilate=0,
        checkpoint="w.pt",
        force=True,
        workers=2,
        recursive=True,
        path_pattern="a/*",
        device="cpu",
    ),
    "masks_merge": MergeMasksRequest(mask_dirs=("x", "y"), output_dir="o"),
    # No ``path_pattern``: Export publishes the whole workspace, and is the one
    # stage the GUI cannot narrow to the open image.
    "export": ExportRequest(
        src="s",
        dst="d",
        masks="m",
        master="mm",
        index="i.json",
        out="o",
        apply=True,
        report_dir="rep",
    ),
}
"""One non-default instance per registered stage id."""

REQUIRED: dict[str, list[str]] = {
    "correct": ["--src", "s", "--dst", "d"],
    "masks_sam": ["--image-dir", "i"],
}
"""The stages that require a root rather than defaulting it: the shortest
argv their parser accepts."""

HEAVY = ("torch", "cv2", "sam3", "timm")

stages = pytest.mark.parametrize("stage", STAGES, ids=lambda s: s.id)


def actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {a.dest: a for a in parser._actions if a.dest != "help"}


def shell(stage: Stage):
    return importlib.import_module(stage.module)


@pytest.fixture(scope="module")
def parsers() -> dict[str, dict[str, argparse.Action]]:
    return {s.id: actions(shell(s).build_parser()) for s in STAGES}


# ---- the registry and the cases agree --------------------------------------


def test_every_registered_stage_has_a_case():
    assert set(CASES) == set(BY_ID)
    for stage in STAGES:
        assert type(CASES[stage.id]) is stage.request_class(), stage.id


# ---- round trips -----------------------------------------------------------


@stages
def test_a_request_round_trips_through_its_parser(stage):
    """The shell's ``build_parser()`` and ``Request.parser()`` read one argv
    to one request."""
    req = CASES[stage.id]
    cls = type(req)
    argv = req.to_argv()
    for parser in (shell(stage).build_parser(), cls.parser()):
        assert cls.from_namespace(parser.parse_args(argv)) == req


@stages
def test_the_parser_defaults_are_the_request_defaults(stage):
    """A required-only argv reads back as a default request, and spells back
    as that argv, so a field's default and its flag's default cannot drift."""
    cls = type(CASES[stage.id])
    required = REQUIRED.get(stage.id, [])
    parsed = cls.from_namespace(shell(stage).build_parser().parse_args(required))
    assert parsed == cls(
        **{
            k.lstrip("-").replace("-", "_"): v
            for k, v in zip(required[::2], required[1::2])
        }
    )
    assert parsed.to_argv() == required


@stages
def test_the_shell_parser_is_the_generated_one(stage):
    """A shell adds nothing of its own: dest for dest, the same flags, default,
    type and choices as ``Request.parser()``."""
    own = actions(shell(stage).build_parser())
    generated = actions(type(CASES[stage.id]).parser())
    assert own.keys() == generated.keys()
    for dest, a in own.items():
        b = generated[dest]
        assert (a.option_strings, a.default, a.type, a.choices, a.nargs) == (
            b.option_strings,
            b.default,
            b.type,
            b.choices,
            b.nargs,
        ), dest


# ---- torch stays out of the request half -----------------------------------


def test_the_requests_import_with_torch_poisoned():
    """Every registered stage's request class, its parser and its CLI shell
    import with the model libraries *poisoned* — ``sys.modules[name] = None``
    makes ``import torch`` raise — so a stray top-level import is an
    ``ImportError`` here, not a slow GUI server. The packages' lazy names
    (``from anime_tools.stages import AutotagRequest, run_autotag``) resolve
    under the same poison."""
    code = f"""
import importlib, sys
for name in {HEAVY!r}:
    sys.modules[name] = None
from anime_tools.stages.registry import STAGES
for stage in STAGES:
    cls = stage.request_class()
    cls.parser()
    importlib.import_module(stage.module).build_parser()
import anime_tools.stages, anime_tools.masking, anime_tools.grouping
for name in anime_tools.stages.__all__:
    getattr(anime_tools.stages, name)
for name in anime_tools.masking.__all__:
    if name.endswith("Request"):
        getattr(anime_tools.masking, name)
anime_tools.grouping.GroupRequest
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert r.returncode == 0, r.stderr


def test_resolving_every_request_stays_import_light():
    """Building the GUI's form schema must not import a stage.

    ``registry.py`` names each request lazily so nothing heavy is imported to
    *list* the stages — but the GUI resolves every class to build its form, and
    a request that reached into its own stage module for one default (a
    ``DEFAULT_MIN_PIXELS``, a ``PositionCaptionOptions``) undid that on every
    schema build: measured at 360 modules with numpy, PIL and yaml along. The
    defaults live in leaves instead (``stages/_options.py``,
    ``masking/_prompts.py``), so this is the assertion that keeps them there.
    """
    code = """
import importlib, sys
from anime_tools.stages.registry import STAGES
for stage in STAGES:
    stage.request_class()
heavy = sorted(m for m in ("numpy", "PIL", "yaml", "cv2", "torch") if m in sys.modules)
assert not heavy, heavy
"""
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert r.returncode == 0, r.stderr


def test_the_request_defaults_are_spelled_once():
    """A stage module re-exports its defaults from the leaf, never the reverse.

    The leaf is the home precisely so ``requests.py`` can read a number without
    the stage; an owner that declared its own copy would drift from the flag's
    default silently.
    """
    from anime_tools.stages import _options, multiview_audit, position_captions, resize

    assert resize.DEFAULT_MIN_PIXELS is _options.DEFAULT_MIN_PIXELS
    assert resize.CROP_ANCHORS is _options.CROP_ANCHORS
    assert resize.DEFAULT_CROP_ANCHOR is _options.DEFAULT_CROP_ANCHOR
    assert multiview_audit.MULTIPLE_VIEWS is _options.MULTIPLE_VIEWS
    assert multiview_audit.EXTRA_CHARACTER is _options.EXTRA_CHARACTER
    assert multiview_audit.DEFAULT_MULTIVIEW_PROB is _options.DEFAULT_MULTIVIEW_PROB
    assert (
        multiview_audit.DEFAULT_IDENTITY_CONFIDENCE
        is _options.DEFAULT_IDENTITY_CONFIDENCE
    )
    assert position_captions.PositionCaptionOptions is _options.PositionCaptionOptions


# ---- the runner -------------------------------------------------------------


@stages
def test_every_stage_names_its_runner(stage):
    """``Stage.run`` resolves to the in-process ``run_<stage>(request)`` the
    CLI shell wraps, so a driver can go from the registry to a call without
    importing a runner by name."""
    fn = stage.runner()
    assert callable(fn)
    assert fn.__name__ == stage.run.rpartition(":")[2]
    assert fn.__name__.startswith("run_"), stage.id


# ---- one spelling per shared flag ------------------------------------------


@stages
def test_every_flag_takes_both_separators(stage):
    """A flag with a separator in it is accepted either way, so the caption
    stages' underscores and the masking CLIs' hyphens are one convention."""
    for a in actions(shell(stage).build_parser()).values():
        canon = a.option_strings[0] if a.option_strings else ""
        if "_" in canon or "-" in canon[2:]:
            assert len(a.option_strings) >= 2, (stage.id, canon)
            body = canon[2:]
            other = "--" + (
                body.replace("_", "-") if "_" in body else body.replace("-", "_")
            )
            assert other in a.option_strings, (stage.id, canon)


@pytest.mark.parametrize(
    ("dest", "flags", "default"),
    [
        ("path_pattern", ("--path_pattern", "--path-pattern"), "*"),
        ("report_dir", ("--report_dir", "--report-dir"), None),
        ("from_report", ("--from_report", "--from-report"), None),
        ("tagger_dir", ("--tagger_dir", "--tagger-dir"), None),
        ("device", ("--device",), None),
        ("src", ("--src",), "image_dataset"),
        ("dst", ("--dst",), "workspace/resized"),
    ],
)
def test_shared_flags_keep_one_spelling(parsers, dest, flags, default):
    """Every caption stage that takes one of these takes it identically: the
    canonical spelling first, the other separator as an alias — the GUI fills
    one ⚙ Settings value into all of them (``gui/stages.py::SETTING_FIELDS``).

    ``--report_dir``'s default is per-stage, so only its spelling is pinned.
    """
    seen = 0
    for stage in STAGES:
        acts = parsers[stage.id]
        if dest not in acts or type(CASES[stage.id]).FLAG_SEP != "_":
            continue
        seen += 1
        action = acts[dest]
        assert tuple(action.option_strings) == flags, (
            f"{stage.id} spells {dest} differently"
        )
        if default is not None:
            assert action.default == default, f"{stage.id} defaults {dest} differently"
    assert seen >= 2, f"{dest} is no longer shared — drop it from this test"


@pytest.mark.parametrize(
    ("dest", "flags", "default", "stages"),
    [
        (
            "checkpoint",
            ("--checkpoint",),
            DEFAULT_SAM3_CHECKPOINT,
            {"position", "audit", "masks_sam"},
        ),
        # The mask stage names its soft prompts per entry of `--masks` instead.
        (
            "prompt_embed",
            ("--prompt_embed", "--prompt-embed"),
            DEFAULT_SUBJECT_PROMPT_EMBED,
            {"position", "audit"},
        ),
    ],
)
def test_the_sam3_stages_share_the_catalog_flags(parsers, dest, flags, default, stages):
    """The two ⚙ Settings model values reach every SAM3 stage that takes them,
    masking's hyphenated CLI included, under one spelling and one default."""
    carriers = {s.id for s in STAGES if dest in parsers[s.id]}
    assert carriers == stages
    for stage_id in carriers:
        action = parsers[stage_id][dest]
        assert tuple(action.option_strings) == flags, stage_id
        assert action.default == default, stage_id


def test_every_stage_with_apply_is_dry_run_by_default():
    """Nothing is written without ``--apply``."""
    carriers = []
    for stage in STAGES:
        cls = type(CASES[stage.id])
        apply = next((a for a in args_of(cls) if a.name == "apply"), None)
        if apply is None:
            continue
        carriers.append(stage.id)
        assert apply.flags[0] == "--apply", stage.id
        assert apply.default is False, stage.id
    assert set(carriers) == {
        "autotag",
        "correct",
        "drop_groups",
        "position",
        "audit",
        "ocr",
        "export",
    }


# ---- the generated parser ----------------------------------------------------


def test_the_help_is_the_field_metadata():
    """``--help`` prints what the request field says, ``%`` included, and the
    parser's description is the class docstring.

    A field's help is plain prose — argparse's own ``%`` escaping happens on
    the way into the parser, so a help that spells ``%%`` itself prints a
    doubled sign.
    """
    parser = shell(BY_ID["ocr"]).build_parser()
    conf = next(a for a in args_of(OcrRequest) if a.name == "det_conf")
    assert "~15% more" in conf.help and "%%" not in conf.help
    assert actions(parser)["det_conf"].help == conf.help.replace("%", "%%")
    printed = parser.format_help()
    assert "~15%" in printed and "%%" not in printed

    parser = shell(BY_ID["position"]).build_parser()
    assert parser.description.startswith("Rewrite multi-subject captions")


def test_every_device_flag_is_the_one_flag():
    """Every ``--device`` is ``_device.py``'s: spelling, default and help.

    It is a :data:`anime_tools.gui.stages.AUTO_FIELDS` dest — neither shown on
    the form nor put on the argv — so the child resolves it through
    :func:`anime_tools._device.resolve_device`, which only works while every
    stage defaults it to ``None`` and spells it identically. The request
    classes carry it as a field (``device: str | None = arg(None,
    help=DEVICE_HELP)``); the CLIs that are not request objects take
    :func:`~anime_tools._device.add_device_arg`.

    Asserted off the parsers rather than off the source text, so a refactor
    that preserves the flag preserves the test.
    """
    from anime_tools._device import DEVICE_HELP, add_device_arg

    hand_written = argparse.ArgumentParser()
    add_device_arg(hand_written)
    one = actions(hand_written)["device"]
    assert one.option_strings == ["--device"]
    assert (one.default, one.help) == (None, DEVICE_HELP)

    carriers = 0
    for stage in STAGES:
        case = CASES[stage.id]
        field = next((a for a in args_of(type(case)) if a.name == "device"), None)
        if field is None:
            continue
        carriers += 1
        assert (field.default, field.help) == (None, DEVICE_HELP), stage.id
        act = actions(shell(stage).build_parser())["device"]
        assert act.option_strings == one.option_strings, stage.id
        assert (act.default, act.help) == (one.default, one.help), stage.id
    assert carriers, "no stage takes a device any more"

    # An AUTO_FIELDS dest: the form never shows it and the argv never carries
    # it, which is what leaves the child free to probe.
    from anime_tools.gui.stages import AUTO_FIELDS

    assert "device" in AUTO_FIELDS
