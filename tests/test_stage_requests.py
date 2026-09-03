"""What the caption and grouping stages' request objects pin beyond the
registry-wide round trip (``test_registry_requests.py``): the argv a default
spells, the workspace defaults, the detection block's wiring, validation, and
the per-process tagger cache."""

from __future__ import annotations

import dataclasses

import pytest

from anime_tools import workspace as WS
from anime_tools._request import args_of
from anime_tools.grouping.requests import GroupRequest
from anime_tools.stages.cli import (
    audit_apply_curated,
    audit_multiview,
    autotag_captions,
)
from anime_tools.stages.position_captions import PositionCaptionOptions
from anime_tools.stages.requests import (
    DETECTION,
    POSITION_ONLY_FLAGS,
    AuditRequest,
    AutotagRequest,
    CorrectRequest,
    DetectionRequest,
    ExportRequest,
    OcrRequest,
    PositionRequest,
    ResizeRequest,
)

# The audit's detection group carries ``name_confidence``, which the position
# stage declares under clause composition: a dest, not a detector knob.
AUDIT_GROUP_EXTRAS = frozenset({"name_confidence"})


def detection_dests(cls) -> set[str]:
    """The dests a stage request puts in its ``detection`` group."""
    return {a.name for a in args_of(cls) if a.group == DETECTION}


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


def test_the_curated_apply_reads_the_report_the_audit_writes():
    """``audit_apply_curated``'s ``--report`` default is the audit's
    ``--report_dir`` default plus the file written there."""
    parser = audit_apply_curated.build_parser()
    report = next(a for a in parser._actions if a.dest == "report")
    assert report.default == f"{audit_multiview.DEFAULT_REPORT_DIR}/report.json"


# ---- the detection block --------------------------------------------------


def test_the_detection_group_is_exactly_the_detection_request():
    """The ``detection`` group of either SAM3 stage is the nested
    ``DetectionRequest`` block, plus the stage's own detection knobs
    (:data:`POSITION_ONLY_FLAGS`, the audit's ``name_confidence``). The audit
    declares none of the position-only flags and takes their defaults."""
    request_fields = {f.name for f in dataclasses.fields(DetectionRequest)}
    assert detection_dests(PositionRequest) - set(POSITION_ONLY_FLAGS) == request_fields
    assert detection_dests(AuditRequest) - AUDIT_GROUP_EXTRAS == request_fields
    audit_dests = {a.name for a in args_of(AuditRequest)}
    position_dests = {a.name for a in args_of(PositionRequest)}
    assert not set(POSITION_ONLY_FLAGS) & audit_dests
    assert set(POSITION_ONLY_FLAGS) <= position_dests
    assert {
        f.name for f in dataclasses.fields(PositionCaptionOptions)
    } <= position_dests


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
    """The audit's options are the position stage's detector verbatim with
    ``min_instances`` pinned; the two default to the same detection block."""
    detection = DetectionRequest(part_prompts=("hips", "thighs"), score_threshold=0.6)
    req = AuditRequest(detection=detection, name_confidence=0.6)
    options = req.options()
    assert options.min_instances == AuditRequest.MIN_INSTANCES == 2
    assert options.mask_containment_threshold == 0.8
    assert options.name_confidence == 0.6
    assert options.part_prompts == ("hips", "thighs")
    assert options.score_threshold == 0.6
    assert options.blank_crops is PositionCaptionOptions().blank_crops
    assert options.strict_count is PositionCaptionOptions().strict_count
    assert AuditRequest().detection == PositionRequest().detection


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
            self.dbv4_runtime = "torch"

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
        # Which backbone ran, since an exported dbv4.onnx is picked up silently.
        "runtime": "torch",
    }
    assert tag_fn(None) == "1girl"
