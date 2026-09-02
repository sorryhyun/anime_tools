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
from pathlib import Path

import pytest

from anime_tools._request import Request, args_of
from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT
from anime_tools.grouping.requests import GroupRequest
from anime_tools.masking.requests import (
    MergeMasksRequest,
    MitMaskRequest,
    SamMaskRequest,
)
from anime_tools.stages.instance_detection import DEFAULT_SUBJECT_PROMPT_EMBED
from anime_tools.stages.registry import BY_ID, STAGES, Stage
from anime_tools.stages.requests import (
    AuditRequest,
    AutotagRequest,
    CorrectRequest,
    DetectionRequest,
    ExportRequest,
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

CASES: dict[str, Request] = {
    "resize": ResizeRequest(
        src="s",
        dst="d",
        path_pattern="a/*",
        target_res=(1024, 768),
        min_pixels=0,
        recursive=False,
        copy_captions=True,
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
        from_report="r.json",
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
    "audit": AuditRequest(
        dst="d",
        apply=True,
        apply_verdicts=("multiple views", "extra-character"),
        apply_confidence=("strong", "weak"),
        crops=True,
        sheets=False,
        detection=DETECTION,
        name_confidence=0.6,
        multiview_threshold=0.4,
        identity_confidence=0.8,
        suggest_counts=True,
    ),
    "ocr": OcrRequest(
        dst="d",
        ocr_dir="o",
        path_pattern="a/*",
        min_score=0.5,
        min_chars=1,
        skip_en=False,
        join_cjk=False,
        min_box_px=8,
        max_boxes=16,
        det_limit_side=960,
        batch_size=4,
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
        prompts=("speech bubble", "text"),
        focus_prompts=(),
        prompt_embed="none",
        threshold=0.7,
        dilate=0,
        checkpoint="w.pt",
        batch_size=4,
        force=True,
        workers=2,
        recursive=True,
        path_pattern="a/*",
        device="cpu",
    ),
    "masks_mit": MitMaskRequest(
        image_dir="i",
        use_sam=True,
        sam_prompts=("sign",),
        sam_threshold=0.3,
        use_mit=False,
        model_path="net.pth",
        text_threshold=0.5,
        ctd_gate=False,
        dilate=1,
        recursive=True,
    ),
    "masks_merge": MergeMasksRequest(mask_dirs=("x", "y"), output_dir="o"),
    "export": ExportRequest(
        src="s",
        dst="d",
        path_pattern="a/*",
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
    "masks_mit": ["--image-dir", "i"],
}
"""The stages that require a root rather than defaulting it: the shortest
argv their parser accepts."""

HEAVY = ("torch", "cv2", "sam3", "onnxruntime", "timm")

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
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr


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

    ``--report_dir``'s default is per-stage, so only its spelling is pinned;
    ``correct`` requires its roots rather than defaulting them.
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
        if default is not None and not (
            stage.id == "correct" and dest in ("src", "dst")
        ):
            assert action.default == default, f"{stage.id} defaults {dest} differently"
    assert seen >= 2, f"{dest} is no longer shared — drop it from this test"


@pytest.mark.parametrize(
    ("dest", "flags", "default"),
    [
        ("checkpoint", ("--checkpoint",), DEFAULT_SAM3_CHECKPOINT),
        (
            "prompt_embed",
            ("--prompt_embed", "--prompt-embed"),
            DEFAULT_SUBJECT_PROMPT_EMBED,
        ),
    ],
)
def test_the_sam3_stages_share_the_catalog_flags(parsers, dest, flags, default):
    """The two ⚙ Settings model values reach every SAM3 stage, masking's
    hyphenated CLIs included, under one spelling and one default. The text
    stage takes ``--checkpoint`` alone: ``--prompt_embed`` stands in for the
    subject phrase only."""
    carriers = {s.id for s in STAGES if dest in parsers[s.id]}
    expected = {"position", "audit", "masks_sam"} | (
        {"masks_mit"} if dest == "checkpoint" else set()
    )
    assert carriers == expected
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
    assert set(carriers) == {"autotag", "position", "audit", "ocr", "export"}


# ---- the generated parser ----------------------------------------------------


def test_the_help_is_the_field_metadata():
    """``--help`` prints what the request field says, ``%`` included, and the
    parser's description is the class docstring."""
    parser = shell(BY_ID["position"]).build_parser()
    acts = actions(parser)
    novel = next(a for a in args_of(PositionRequest) if a.name == "max_novel_tags")
    assert "46% novel" in novel.help
    assert acts["max_novel_tags"].help == novel.help.replace("%", "%%")
    assert "46% novel" in parser.format_help()
    assert parser.description.startswith("Rewrite multi-subject captions")


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
    for stage in STAGES:
        device = next(
            (a for a in args_of(type(CASES[stage.id])) if a.name == "device"), None
        )
        if device is not None:
            assert device.default is None, stage.id
