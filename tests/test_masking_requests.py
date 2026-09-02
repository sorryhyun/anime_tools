"""The mask stages' request objects are the CLIs' flags, both ways."""

from __future__ import annotations

import subprocess
import sys

import pytest

from anime_tools import workspace as WS
from anime_tools.masking.cli import generate_masks, generate_masks_mit, merge_masks
from anime_tools.masking.requests import (
    MergeMasksRequest,
    MitMaskRequest,
    SamMaskRequest,
)

CASES = [
    (
        generate_masks,
        SamMaskRequest(
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
    ),
    (
        generate_masks_mit,
        MitMaskRequest(
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
    ),
    (merge_masks, MergeMasksRequest(mask_dirs=("x", "y"), output_dir="o")),
]


@pytest.mark.parametrize("module, req", CASES, ids=lambda c: getattr(c, "__name__", ""))
def test_a_request_round_trips_through_its_parser(module, req):
    argv = req.to_argv()
    assert type(req).from_namespace(module.build_parser().parse_args(argv)) == req


@pytest.mark.parametrize("module, req", CASES, ids=lambda c: getattr(c, "__name__", ""))
def test_the_parser_defaults_are_the_request_defaults(module, req):
    """A default-only argv reads back as a default request, so a field's
    default and its flag's default cannot drift."""
    required = ["--image-dir", "i"] if module is not merge_masks else []
    parsed = type(req).from_namespace(module.build_parser().parse_args(required))
    assert parsed == type(req)(**({"image_dir": "i"} if required else {}))
    assert parsed.to_argv() == required


def test_a_default_argv_names_only_what_changed():
    req = SamMaskRequest(image_dir="i", prompts=("text",), focus_prompts=())
    assert req.to_argv() == [
        "--image-dir",
        "i",
        "--prompts",
        "text",
        "--focus-prompts",
        "none",
    ]
    assert MitMaskRequest(image_dir="i", ctd_gate=False).to_argv() == [
        "--image-dir",
        "i",
        "--no-ctd-gate",
    ]
    assert MergeMasksRequest(mask_dirs=("a",)).to_argv() == ["a"]


def test_the_generators_default_to_their_own_trees():
    assert SamMaskRequest(image_dir="i").mask_dir == WS.MASKS_SAM
    assert MitMaskRequest(image_dir="i").mask_dir == WS.MASKS_MIT
    assert MergeMasksRequest().mask_dirs == (WS.MASKS_SAM, WS.MASKS_MIT)
    assert MergeMasksRequest().output_dir == WS.MASKS


def test_a_request_refuses_a_run_that_would_detect_nothing():
    with pytest.raises(ValueError, match="nothing to mask"):
        SamMaskRequest(image_dir="i", focus_prompts=())
    with pytest.raises(ValueError, match="nothing to detect"):
        MitMaskRequest(image_dir="i", use_mit=False)
    with pytest.raises(ValueError, match="no --sam-prompts"):
        MitMaskRequest(image_dir="i", use_sam=True, sam_prompts=())
    # Shut, the SAM drawer's prompts are inert.
    assert MitMaskRequest(image_dir="i", sam_prompts=()).active_sam_prompts == ()


def test_the_requests_import_without_a_model_library():
    code = (
        "import sys, anime_tools.masking, anime_tools.masking.requests; "
        "from anime_tools.masking import SamMaskRequest, MitMaskRequest, MergeMasksRequest; "
        "heavy = {'torch', 'cv2', 'sam3', 'onnxruntime', 'timm'} & set(sys.modules); "
        "assert not heavy, heavy"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
