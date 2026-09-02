"""What the mask stages' request objects pin beyond the registry-wide round
trip (``test_registry_requests.py``): the argv a default spells, the mask
trees, the prompt lists and validation."""

from __future__ import annotations

import pytest

from anime_tools import workspace as WS
from anime_tools.masking._sam3 import SUBJECT_PROMPT, prompt_list
from anime_tools.masking.requests import (
    MergeMasksRequest,
    MitMaskRequest,
    SamMaskRequest,
)
from anime_tools.stages.instance_detection import DEFAULT_SUBJECT_PROMPT_EMBED


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


def test_the_sam3_mask_stage_defaults_to_the_phrase_its_soft_prompt_encodes():
    """The default ``--focus-prompts`` is the phrase ``--prompt_embed`` encodes."""
    req = SamMaskRequest(image_dir="i")
    assert req.focus_prompts == (SUBJECT_PROMPT,)
    assert req.prompt_embed == DEFAULT_SUBJECT_PROMPT_EMBED


def test_a_prompt_flag_is_a_comma_separated_list():
    assert prompt_list(" speech bubble , text ,") == ("speech bubble", "text")
    assert prompt_list("") == ()
    # The GUI drops a blank field back to the default, so "none of them" needs a word.
    assert prompt_list("none") == ()
    assert SamMaskRequest.from_namespace(
        SamMaskRequest.parser().parse_args(["--image-dir", "i", "--prompts", "a, b"])
    ).prompts == ("a", "b")
