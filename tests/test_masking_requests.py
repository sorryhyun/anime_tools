"""What the mask stages' request objects pin beyond the registry-wide round
trip (``test_registry_requests.py``): the argv a default spells, the mask
trees, the mask list and validation."""

from __future__ import annotations

import pytest

from anime_tools import workspace as WS
from anime_tools._request import args_of
from anime_tools.masking.requests import MaskPrompt, MergeMasksRequest, SamMaskRequest
from anime_tools.stages.instance_detection import DEFAULT_SUBJECT_PROMPT_EMBED


def test_a_default_argv_names_only_what_changed():
    req = SamMaskRequest(image_dir="i", masks=(MaskPrompt("ignore", "text", "text"),))
    assert req.to_argv() == ["--image-dir", "i", "--masks", "ignore:text:text"]
    assert SamMaskRequest(image_dir="i").to_argv() == ["--image-dir", "i"]
    assert MergeMasksRequest(mask_dirs=("a",)).to_argv() == ["a"]


def test_the_generator_defaults_to_its_own_tree():
    assert SamMaskRequest(image_dir="i").mask_dir == WS.MASKS_SAM
    assert MergeMasksRequest().mask_dirs == (WS.MASKS_SAM,)
    assert MergeMasksRequest().output_dir == WS.MASKS


def test_a_request_refuses_a_run_that_would_detect_nothing():
    with pytest.raises(ValueError, match="nothing to mask"):
        SamMaskRequest(image_dir="i", masks=())


def test_the_default_keeps_the_subject_through_the_shipped_soft_prompt():
    req = SamMaskRequest(image_dir="i")
    assert req.masks == (MaskPrompt("keep", "soft", DEFAULT_SUBJECT_PROMPT_EMBED),)
    assert req.keep == req.masks and req.ignore == ()


def test_a_mask_is_role_kind_value_and_the_value_keeps_its_colons():
    """A Windows path's drive letter is a colon; everything after the second one
    is the value."""
    m = MaskPrompt.parse("keep:soft:C:/soft/girl.safetensors")
    assert m == MaskPrompt("keep", "soft", "C:/soft/girl.safetensors")
    assert MaskPrompt.parse(m.spec()) == m
    parsed = SamMaskRequest.from_namespace(
        SamMaskRequest.parser().parse_args(
            [
                "--image-dir",
                "i",
                "--masks",
                "keep:text:girl",
                "ignore:text:speech bubble",
            ]
        )
    )
    assert parsed.keep == (MaskPrompt("keep", "text", "girl"),)
    assert parsed.ignore == (MaskPrompt("ignore", "text", "speech bubble"),)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("girl", "ROLE:KIND:VALUE"),
        ("focus:text:girl", "role"),
        ("keep:embed:x", "kind"),
        ("keep:text: ", "names no prompt"),
        ("keep:soft:none", "names a file"),
    ],
)
def test_a_malformed_mask_is_refused(spec, match):
    with pytest.raises(ValueError, match=match):
        MaskPrompt.parse(spec)


def test_the_mask_list_is_its_own_form_kind():
    """The GUI draws a row editor for it rather than a textarea of strings."""
    field = next(a for a in args_of(SamMaskRequest) if a.name == "masks")
    assert field.kind == "masks"
    assert field.default == [f"keep:soft:{DEFAULT_SUBJECT_PROMPT_EMBED}"]
