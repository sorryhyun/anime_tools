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
from pathlib import Path

import numpy as np

from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT

# Deliberately a module-level side effect — see the docstring.
if not hasattr(np, "bool"):
    np.bool = np.bool_


def add_checkpoint_arg(p: argparse.ArgumentParser) -> None:
    """``--checkpoint`` — SAM3 weights, defaulted from the download catalog.

    Beside :func:`load_sam3` because the flag and the load are the same fact: a
    stage must name the file the ⚙ Settings → Models row *writes*, or the
    download button and the loader disagree. It is also a
    :data:`anime_tools.gui.stages.SETTING_FIELDS` dest, filled once from
    Settings — which only works while all three SAM3 CLIs spell it identically.
    """
    p.add_argument("--checkpoint", default=DEFAULT_SAM3_CHECKPOINT, help="SAM3 weights")


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


def make_processor(model, confidence_threshold: float | None = None):
    """A ``Sam3Processor`` over an already-loaded model.

    Split out so a caller holding a model can rebuild the processor at a lower
    threshold without paying for the weights again.
    """
    from sam3.model.sam3_image_processor import Sam3Processor

    if confidence_threshold is None:
        return Sam3Processor(model)
    return Sam3Processor(model, confidence_threshold=confidence_threshold)
