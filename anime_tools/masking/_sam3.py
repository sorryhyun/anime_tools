"""The one SAM3 entry point: model + processor construction, and the numpy shim.

**The numpy shim is an import side effect of this module**: ``sam3`` pins
``numpy<2`` upstream and still spells ``np.bool``, which numpy 2 removed, so the
alias has to exist *before* ``import sam3``. Importing this module is the one
place that guarantee is made — hence the sam3 imports deferred into the
functions, which also keeps importing this module torch-free (a ``--help`` or a
GUI schema dump costs nothing).
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np

from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT, DEFAULT_SUBJECT_PROMPT_EMBED

# Deliberately a module-level side effect — see the docstring.
if not hasattr(np, "bool"):
    np.bool = np.bool_

SUBJECT_PROMPT = "girl"
"""The text prompt every SAM3 stage means by *the subject*, and the phrase the
shipped soft prompt is the textual inversion of. Named here because it is the
hinge between two flags: it is ``--prompt``'s default, and it is what
``--prompt_embed`` stands in for."""


def add_checkpoint_arg(p: argparse._ActionsContainer) -> None:
    """``--checkpoint`` — SAM3 weights, defaulted from the download catalog.

    Beside :func:`load_sam3` because the flag and the load are the same fact: a
    stage must name the file the ⚙ Settings → Models row *writes*, or the
    download button and the loader disagree. It is also a
    :data:`anime_tools.gui.stages.SETTING_FIELDS` dest, filled once from
    Settings — which only works while every SAM3 CLI spells it identically.
    Takes a group as readily as a parser, like :func:`add_prompt_embed_arg`:
    the text-mask stage declares it inside its ``use_sam`` drawer.
    """
    p.add_argument("--checkpoint", default=DEFAULT_SAM3_CHECKPOINT, help="SAM3 weights")


def add_prompt_embed_arg(p: argparse._ActionsContainer) -> None:
    """``--prompt_embed`` — the learned subject prompt, from the same catalog.

    Here for the reason ``--checkpoint`` is: it names a file a ⚙ Settings →
    Models row writes, and it is a
    :data:`anime_tools.gui.stages.SETTING_FIELDS` dest filled once from
    Settings — which only works while every stage that takes one spells it
    identically. Takes a group as readily as a parser, because the detection
    stages declare it inside their ``detection`` group.
    """
    p.add_argument(
        "--prompt_embed",
        default=DEFAULT_SUBJECT_PROMPT_EMBED,
        help="learned soft prompt (.safetensors) used in place of the "
        f"{SUBJECT_PROMPT!r} text prompt for the subject pass; every other "
        f"prompt stays textual. Default = the shipped "
        f"{DEFAULT_SUBJECT_PROMPT_EMBED}; pass `none` for the plain text prompt",
    )


_NO_PROMPTS = {"none", "off"}


def prompt_list(spec: str) -> tuple[str, ...]:
    """A comma-separated prompt flag as the tuple of prompts it names.

    ``none`` / ``off`` mean *no prompts*, the word ``--prompt_embed`` already
    takes for the same job. Emptying the field is not enough to say it: the GUI
    omits a flag whose value is blank, so a cleared prompt field would come back
    as its default rather than as "prompt for nothing".
    """
    if spec.strip().lower() in _NO_PROMPTS:
        return ()
    return tuple(t.strip() for t in spec.split(",") if t.strip())


def autocast(device: str):
    """The half-precision context a SAM3 pass runs under, or nothing on CPU.

    ``torch.autocast(device_type="cuda")`` on a machine without one only warns
    and disables itself, so this is not a correctness fix — but every SAM3 stage
    wants the same answer, and a CPU run should not have to read a warning to
    learn that it got it.
    """
    import torch

    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_sam3(
    checkpoint: str | Path | None = None,
    device: str = "cuda",
    *,
    confidence_threshold: float | None = None,
    disable_act_ckpt: bool = False,
):
    """Build a frozen SAM3 image model and its processor.

    ``checkpoint`` names a local ``.pt``; ``None`` lets sam3 fetch its own
    weights from HF. ``confidence_threshold`` is the processor's *own* floor,
    applied before the caller ever sees a box — pass the lowest score any
    consumer might ask for and re-gate on top (see ``build_detect_fn``'s
    GOTCHA 1); ``None`` keeps the processor default.

    ``disable_act_ckpt`` turns off the two activation-checkpoint paths SAM3
    leaves on even in eval mode; they cost a recompute for nothing when the trunk
    is frozen, so only the soft-prompt trainer asks for it.

    Returns ``(model, processor)``.
    """
    from sam3.model_builder import build_sam3_image_model

    build_kwargs: dict = {"device": device, "eval_mode": True}
    if checkpoint is not None:
        build_kwargs["checkpoint_path"] = str(checkpoint)
        build_kwargs["load_from_HF"] = False
    model = build_sam3_image_model(**build_kwargs)

    if disable_act_ckpt:
        n = 0
        for m in model.modules():
            for attr in ("act_ckpt", "use_act_checkpoint"):
                if getattr(m, attr, False) is True:
                    setattr(m, attr, False)
                    n += 1
        if n:
            print(f"sam3: disabled activation checkpointing on {n} modules", flush=True)

    return model, make_processor(model, confidence_threshold)


def ground_with_soft_prompt(processor, model, state: dict, soft_prompt: dict) -> dict:
    """One grounding pass whose prompt is already encoded, not text.

    The drop-in for ``processor.set_text_prompt(state=…, prompt=…)`` when the
    prompt is a learned tensor: a soft prompt *is* what SAM3's text encoder
    would have produced, so the encode is skipped and ``load_soft_prompt``'s
    triple goes straight into the state :meth:`Sam3Processor.set_image` built.
    Saying that means reaching past ``set_text_prompt`` into the grounding call
    underneath it, so it is written once here rather than in each stage that
    takes a ``--prompt_embed``. The caller still owns *which* of its prompts the
    embed stands in for — see :data:`SUBJECT_PROMPT`.
    """
    state["backbone_out"].update(soft_prompt)
    state.setdefault("geometric_prompt", model._get_dummy_prompt())
    return processor._forward_grounding(state)


def detect_union(
    processor,
    model,
    state: dict,
    prompts,
    shape: tuple[int, int],
    threshold: float,
    *,
    soft_prompt: dict | None = None,
) -> np.ndarray:
    """OR-combine SAM3's detections for every prompt into one binary mask.

    The whole of what a mask stage does with SAM3 between ``set_image`` and the
    mask it writes: prompt, drop anything under ``threshold``, union. Here
    rather than in a stage because the soft-prompt substitution rule lives here
    too — a ``soft_prompt`` stands in for :data:`SUBJECT_PROMPT` and for no
    other prompt, and a caller whose prompts are all textual passes ``None``.
    """
    import torch

    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    for prompt in prompts:
        if soft_prompt is not None and prompt == SUBJECT_PROMPT:
            output = ground_with_soft_prompt(processor, model, state, soft_prompt)
        else:
            output = processor.set_text_prompt(state=state, prompt=prompt)
        for mask, score in zip(output["masks"], output["scores"]):
            if score < threshold:
                continue
            mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            out = np.maximum(out, (mask_np > 0.5).astype(np.uint8))
    return out


def make_processor(model, confidence_threshold: float | None = None):
    """A ``Sam3Processor`` over an already-loaded model.

    Split out so a caller holding a model can rebuild the processor at a lower
    threshold without paying for the weights again.
    """
    from sam3.model.sam3_image_processor import Sam3Processor

    if confidence_threshold is None:
        return Sam3Processor(model)
    return Sam3Processor(model, confidence_threshold=confidence_threshold)
