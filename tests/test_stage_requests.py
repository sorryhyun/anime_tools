"""The caption and grouping stages' request objects are the CLIs' flags, both
ways — the P2/P3 shape of ``docs/api_first_plan.md``."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

from anime_tools import workspace as WS
from anime_tools.grouping.cli import build_groups
from anime_tools.grouping.requests import GroupRequest
from anime_tools.stages.cli import (
    audit_multiview,
    autotag_captions,
    correct_captions,
    export_workspace,
    ocr_captions,
    position_captions,
    resize_images,
)
from anime_tools.stages.position_captions import PositionCaptionOptions
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

CASES = [
    (
        autotag_captions,
        AutotagRequest(
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
    ),
    (
        position_captions,
        PositionRequest(
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
    ),
    (position_captions, PositionRequest(flatten=True, apply=True)),
    (
        audit_multiview,
        AuditRequest(
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
    ),
    (
        correct_captions,
        CorrectRequest(
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
    ),
    (
        ocr_captions,
        OcrRequest(
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
    ),
    (
        resize_images,
        ResizeRequest(
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
    ),
    (
        export_workspace,
        ExportRequest(
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
    ),
    (
        build_groups,
        GroupRequest(
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
    ),
]
REQUIRED = {CorrectRequest: ["--src", "s", "--dst", "d"]}
"""The one stage that requires its roots rather than defaulting them."""


def _id(case):
    return getattr(case, "__name__", type(case).__name__)


@pytest.mark.parametrize("module, req", CASES, ids=_id)
def test_a_request_round_trips_through_its_parser(module, req):
    argv = req.to_argv()
    assert type(req).from_namespace(module.build_parser().parse_args(argv)) == req


@pytest.mark.parametrize("module, req", CASES, ids=_id)
def test_the_parser_defaults_are_the_request_defaults(module, req):
    """A default-only argv reads back as a default request, so a field's
    default and its flag's default cannot drift."""
    cls = type(req)
    required = REQUIRED.get(cls, [])
    parsed = cls.from_namespace(module.build_parser().parse_args(required))
    assert parsed == cls(
        **{k.lstrip("-"): v for k, v in zip(required[::2], required[1::2])}
    )
    assert parsed.to_argv() == required


def test_a_default_argv_names_only_what_changed():
    """Underscore spellings, ``store_false`` switches spelled by their one flag,
    the nested detection block inline, ``nargs`` values after one flag."""
    assert AutotagRequest(mode="merge", apply=True).to_argv() == [
        "--apply",
        "--mode",
        "merge",
    ]
    assert PositionRequest(
        detection=DetectionRequest(part_prompts=("hips",)), rewrite=False
    ).to_argv() == ["--part_prompts", "hips", "--no_rewrite"]
    assert OcrRequest(skip_en=False).to_argv() == ["--keep_en"]
    assert ResizeRequest(target_res=(1024,), recursive=False).to_argv() == [
        "--target_res",
        "1024",
        "--no-recursive",
    ]
    assert AuditRequest(apply_confidence=("strong", "weak")).to_argv() == [
        "--apply_confidence",
        "strong,weak",
    ]
    assert GroupRequest(min_size=1).to_argv() == ["--min-size", "1"]


def test_the_stage_defaults_are_the_workspace_layout():
    assert AutotagRequest().dst == WS.RESIZED
    assert OcrRequest().ocr_dir == WS.OCR
    assert ExportRequest().out == WS.EXPORT_ROOT
    assert GroupRequest().source_dir == WS.RESIZED
    reports = {
        cls().report_dir
        for cls in (AutotagRequest, PositionRequest, AuditRequest, OcrRequest)
    }
    assert len(reports) == 4, "two stages report into one directory"


# ---- the detection block --------------------------------------------------


def _twist(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + " hips"
    if isinstance(value, tuple):
        return (*value, "hips")
    raise AssertionError(f"no twist for {value!r}")


def test_every_option_field_is_wired_to_the_request_field_that_names_it():
    """Field by field: twist the request, rebuild the options, and require the
    option to move — a field that does not is one ``options()`` drops."""
    base = PositionRequest()
    for f in dataclasses.fields(PositionCaptionOptions):
        if hasattr(base.detection, f.name):
            twisted = dataclasses.replace(
                base,
                detection=dataclasses.replace(
                    base.detection,
                    **{f.name: _twist(getattr(base.detection, f.name))},
                ),
            )
        else:
            twisted = dataclasses.replace(
                base, **{f.name: _twist(getattr(base, f.name))}
            )
        assert getattr(twisted.options(), f.name) != getattr(base.options(), f.name), (
            f"{f.name} never reaches PositionCaptionOptions"
        )
    assert base.options() == PositionCaptionOptions()


def test_the_audit_pins_two_subjects_and_takes_the_rest_from_its_detector():
    req = AuditRequest(detection=DETECTION, name_confidence=0.6)
    options = req.options()
    assert options.min_instances == AuditRequest.MIN_INSTANCES == 2
    assert options.name_confidence == 0.6
    assert options.part_prompts == ("hips", "thighs")
    assert options.blank_crops is PositionCaptionOptions().blank_crops
    assert options.strict_count is PositionCaptionOptions().strict_count


def test_the_processor_floor_is_the_lowest_threshold_any_pass_asks_for():
    assert DetectionRequest().floor == 0.35
    assert DetectionRequest(part_score_threshold=0.1).floor == 0.1


# ---- validation lives in the request ---------------------------------------


def test_a_request_refuses_what_its_cli_refused():
    with pytest.raises(ValueError, match="--mode"):
        AutotagRequest(mode="replace")
    with pytest.raises(ValueError, match="mutually exclusive"):
        PositionRequest(flatten=True, from_report="r.json")
    with pytest.raises(ValueError, match="--qwen3"):
        CorrectRequest(
            src="s", dst="d", caption_shuffle_variants=2, caption_tag_randomize_rate=0.5
        )
    # One variant never randomizes, so the tokenizers are not required.
    assert not CorrectRequest(
        src="s", dst="d", caption_shuffle_variants=1, caption_tag_randomize_rate=0.5
    ).randomizes
    with pytest.raises(ValueError, match="allowed tiers"):
        ResizeRequest(target_res=(1000,))
    with pytest.raises(ValueError, match="--resize_crop_anchor"):
        ResizeRequest(resize_crop_anchor="corner")
    with pytest.raises(ValueError, match="module:callable"):
        GroupRequest(embedder="just_a_module")


def test_the_cli_reports_a_bad_request_the_way_argparse_does(capsys):
    with pytest.raises(SystemExit) as exc:
        autotag_captions.main(["--mode", "replace"])
    assert exc.value.code == 2
    assert "--mode" in capsys.readouterr().err


# ---- torch stays out of the request half ----------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "import anime_tools.stages.requests, anime_tools.stages.run",
        "from anime_tools.stages import AutotagRequest, PositionRequest, run_autotag",
        "import anime_tools.grouping.requests; from anime_tools.grouping import GroupRequest",
        (
            "from anime_tools.stages.cli import position_captions, audit_multiview, "
            "autotag_captions, correct_captions, ocr_captions, resize_images, "
            "export_workspace"
        ),
        "from anime_tools.grouping.cli import build_groups",
    ],
)
def test_the_requests_import_without_a_model_library(code):
    probe = (
        f"import sys; {code}; "
        "heavy = {'torch', 'cv2', 'sam3', 'onnxruntime', 'timm'} & set(sys.modules); "
        "assert not heavy, heavy"
    )
    r = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr


def test_the_tagger_loads_once_per_process(monkeypatch, tmp_path):
    """Autotag followed by position in one interpreter reads the weights once:
    ``load_anima_tagger`` caches on ``(checkpoint dir, device)``."""
    from anime_tools.stages import _models
    from anime_tools.tagger import tagger as tagger_mod

    built = []

    class FakeDevice:
        """What ``AnimaTagger.device`` is: a ``torch.device``, not a string."""

        def __str__(self):
            return "cpu"

    class FakeTagger:
        def __init__(self, ckpt_dir, device):
            built.append((ckpt_dir, device))
            self.device = FakeDevice()

        def predict_caption(self, image, min_confidence=0.0):
            return "1girl"

    monkeypatch.setattr(tagger_mod, "AnimaTagger", FakeTagger)
    monkeypatch.setattr(tagger_mod, "ensure_tagger_checkpoint", lambda p: p)
    monkeypatch.setattr(_models, "_LOADED", {})

    a, _ = _models.load_anima_tagger(tmp_path, "cpu", quiet=True)
    b, _ = _models.load_anima_tagger(str(tmp_path), "cpu", quiet=True)
    assert a is b and len(built) == 1
    _models.load_anima_tagger(tmp_path / "other", "cpu", quiet=True)
    assert len(built) == 2

    # Autotag goes through the same cache, and what it reports is JSON.
    import json

    from anime_tools.stages.autotag import build_tag_fn

    tag_fn, info = build_tag_fn(tmp_path, "cpu")
    assert len(built) == 2
    assert json.loads(json.dumps(info)) == {
        "tagger_dir": str(tmp_path),
        "device": "cpu",
    }
    assert tag_fn(None) == "1girl"
