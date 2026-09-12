"""The one SAM3 entry point: model + processor construction, and the two shims
``import sam3`` needs. What a *prompt* is — the subject's text prompt, the learned
soft prompt, the two flags' help — is :mod:`_prompts`, which importing this module
must not be the only way to reach: this one aliases ``np.bool`` on import.

**The numpy shim is an import side effect of this module**: ``sam3`` still spells
``np.bool``, which numpy 2 removed, so the alias has to exist *before* ``import sam3``.
The sam3 imports are deferred into the functions, which also keeps importing this module
torch-free.

**The triton shim runs at the sam3 import**: ``sam3.model.edt`` does ``import triton``
at module scope, and triton has no macOS build (Windows takes the ``triton-windows``
wheel, see ``pyproject.toml``). The kernel belongs to the video tracker, which already
falls back to skimage on CPU, so the image model never calls it; :func:`stub_edt_kernel`
pre-seeds that one module with a stand-in that refuses to run. It stubs the *sam3*
module, never ``triton`` itself: torch guards its own ``import triton`` and would take a
fake one for the real thing.

**The off-CUDA shim runs at the model build**: the image model's builder honours
``device`` only when it is ``"cuda"``, and two of its constant caches are built on a
literal ``"cuda"`` besides. :func:`shim_sam3_off_cuda` points those at the device
actually in use and is inert on CUDA, so the same model runs on an Apple MPS device
or on a CPU. :func:`load_sam3` does the ``.to(device)`` the builder skips.
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np

from anime_tools._progress import phase
from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT
from anime_tools.masking._prompts import CHECKPOINT_HELP

# A module-level side effect on purpose — see the docstring.
if not hasattr(np, "bool"):
    np.bool = np.bool_

EDT_MODULE = "sam3.model.edt"
"""The one sam3 module that imports triton at module scope."""

TRITON_MISSING = (
    "sam3's triton kernel (edt_triton) was called, but triton is not installed on "
    "this platform — this is the video-tracker path, which the image model does not take"
)


def stub_edt_kernel() -> bool:
    """Pre-seed :data:`EDT_MODULE` on ``sys.modules`` with a stand-in when triton
    cannot be imported, so ``import sam3`` no longer trips over it. Returns whether the
    stand-in was installed (``False`` when triton exists or sam3's own module is
    already loaded)."""
    import importlib.util
    import sys
    import types

    if EDT_MODULE in sys.modules or importlib.util.find_spec("triton") is not None:
        return False

    def edt_triton(data):
        raise RuntimeError(TRITON_MISSING)

    module = types.ModuleType(EDT_MODULE)
    module.edt_triton = edt_triton  # type: ignore[attr-defined]
    module.__anime_tools_stub__ = True  # type: ignore[attr-defined]
    sys.modules[EDT_MODULE] = module
    return True


def shim_sam3_off_cuda(device: str = "cpu") -> bool:
    """Make ``build_sam3_image_model`` true to a non-CUDA ``device``.

    The position-encoding precompute and the decoder's coordinate cache are built on
    a literal ``"cuda"`` (``position_encoding.py`` / ``decoder.py``), which a torch
    without CUDA cannot allocate. The precompute exists to keep ``torch.compile``
    from tracing symbolic shapes, so off CUDA the cache fills lazily instead; the
    coordinate cache is built on ``device`` — **on the model's device, not on CPU**,
    since it is indexed against tensors that live there and a CPU index into an MPS
    tensor is refused outright. The ViT's fused ``addmm_act`` (``perflib/fused.py``)
    casts its operands to bf16 unconditionally, which only a CUDA autocast graph is
    already in, so off CUDA it is the plain ``activation(linear(x))``.
    ``Tensor.pin_memory`` becomes a no-op: the geometry encoder pins a vector bound
    for the model's device, and pinning serves host→CUDA copies only (on a Mac it
    pins to MPS and the copy to CPU is refused). Every wrapper checks at call time,
    so a process with CUDA is untouched. Returns whether the shim was installed.
    """
    import torch

    if torch.cuda.is_available():
        return False
    from sam3.model.decoder import TransformerDecoder
    from sam3.model.position_encoding import PositionEmbeddingSine

    if getattr(PositionEmbeddingSine, "__anime_tools_shim__", False):
        _COORD_DEVICE["device"] = str(device)
        return False
    _COORD_DEVICE["device"] = str(device)

    sine_init = PositionEmbeddingSine.__init__

    def __init__(self, *args, **kwargs):
        if not torch.cuda.is_available():
            args = args[:4]  # (num_pos_feats, temperature, normalize, scale)
            kwargs.pop("precompute_resolution", None)
        sine_init(self, *args, **kwargs)

    get_coords = TransformerDecoder._get_coords

    def _get_coords(H, W, device):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = _COORD_DEVICE["device"]
        return get_coords(H, W, device)

    from sam3.model import vitdet

    fused = vitdet.addmm_act

    def addmm_act(activation, linear, mat1):
        # The fused op casts to bf16 unconditionally — written for a CUDA autocast
        # graph that is bf16 anyway. Off CUDA the plain fp32 path is the same math.
        if mat1.device.type == "cuda":
            return fused(activation, linear, mat1)
        return (
            activation()(linear(mat1))
            if isinstance(activation, type)
            else activation(linear(mat1))
        )

    pin_memory = torch.Tensor.pin_memory

    def _pin_memory(self, *args, **kwargs):
        # Pinned memory serves host→CUDA copies. The geometry encoder pins a scale
        # vector on its way to the model's device; on a Mac that pins to MPS, and the
        # copy back to CPU is then refused.
        if not torch.cuda.is_available():
            return self
        return pin_memory(self, *args, **kwargs)

    vitdet.addmm_act = addmm_act
    torch.Tensor.pin_memory = _pin_memory  # type: ignore[method-assign]
    PositionEmbeddingSine.__init__ = __init__  # type: ignore[method-assign]
    PositionEmbeddingSine.__anime_tools_shim__ = True  # type: ignore[attr-defined]
    TransformerDecoder._get_coords = staticmethod(_get_coords)  # type: ignore[method-assign]
    return True


_COORD_DEVICE: dict[str, str] = {"device": "cpu"}
"""Where :func:`shim_sam3_off_cuda` sends the decoder's ``"cuda"`` coordinate cache.
A dict rather than a closure variable because the shim is installed once per process
and the device is per :func:`load_sam3` call — two models on two devices in one
interpreter is not a thing any stage does, but the last build winning is at least a
rule, and it is the device the next forward runs on."""


def add_checkpoint_arg(p: argparse._ActionsContainer) -> None:
    """``--checkpoint`` — SAM3 weights, defaulted from the download catalog."""
    p.add_argument(
        "--checkpoint", default=DEFAULT_SAM3_CHECKPOINT, help=CHECKPOINT_HELP
    )


def autocast(device: str):
    """The half-precision context a SAM3 pass runs under, or nothing off CUDA.

    ``torch.autocast(device_type="cuda")`` on a machine without one only warns and
    disables itself, so this is a warning suppressed, not a correctness fix.

    MPS is deliberately not autocast. It has the context, but SAM3 off CUDA already
    runs through :func:`shim_sam3_off_cuda`'s unfused ``addmm_act``, and half
    precision on top of that is a second dtype seam through the same graph for a
    saving nobody has measured. fp32 there is the conservative default.
    """
    import torch

    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


_LOADED: dict[tuple, tuple] = {}
"""``load_sam3``'s per-process cache, keyed on its arguments."""


def load_sam3(
    checkpoint: str | Path | None = None,
    device: str | None = None,
    *,
    confidence_threshold: float | None = None,
    disable_act_ckpt: bool = False,
):
    """Build a frozen SAM3 image model and its processor on ``device`` (auto when
    ``None``).

    ``checkpoint`` names a local ``.pt``; ``None`` lets sam3 fetch its own weights from
    HF. ``confidence_threshold`` is the processor's *own* floor, applied before the caller
    ever sees a box — pass the lowest score any consumer might ask for and re-gate on top
    (see ``build_detect_fn``'s GOTCHA 1); ``None`` keeps the processor default.

    ``disable_act_ckpt`` turns off the two activation-checkpoint paths SAM3 leaves on even
    in eval mode; they cost a recompute for nothing when the trunk is frozen.

    Returns ``(model, processor)``. Cached per process on every argument, so a
    text-mask pass that follows a subject-mask pass (or a position pass) in one
    interpreter reuses the model instead of reading the weights again.
    """
    from anime_tools._device import resolve_device

    device = resolve_device(device)
    key = (
        None if checkpoint is None else str(checkpoint),
        device,
        confidence_threshold,
        disable_act_ckpt,
    )
    loaded = _LOADED.get(key)
    if loaded is not None:
        return loaded

    stub_edt_kernel()
    from sam3.model_builder import build_sam3_image_model

    shim_sam3_off_cuda(device)

    build_kwargs: dict = {"device": device, "eval_mode": True}
    if checkpoint is not None:
        build_kwargs["checkpoint_path"] = str(checkpoint)
        build_kwargs["load_from_HF"] = False
    # The one quiet stretch of every SAM3 stage; under the daemon a heartbeat
    # keeps it from reading as a stall (``_progress``).
    with phase("load sam3"):
        model = build_sam3_image_model(**build_kwargs)
    # The builder moves the weights itself only for "cuda" (`_setup_device_and_mode`),
    # so anything else — MPS above all — is still on CPU here.
    if not str(device).startswith("cuda"):
        model = model.to(device)

    if disable_act_ckpt:
        n = 0
        for m in model.modules():
            for attr in ("act_ckpt", "use_act_checkpoint"):
                if getattr(m, attr, False) is True:
                    setattr(m, attr, False)
                    n += 1
        if n:
            print(f"sam3: disabled activation checkpointing on {n} modules", flush=True)

    loaded = _LOADED[key] = (model, make_processor(model, confidence_threshold))
    return loaded


def ground_with_soft_prompt(processor, model, state: dict, soft_prompt: dict) -> dict:
    """One grounding pass whose prompt is already encoded, not text.

    The drop-in for ``processor.set_text_prompt(state=…, prompt=…)`` when the prompt is a
    learned tensor: a soft prompt *is* what SAM3's text encoder would have produced, so
    the encode is skipped and ``load_soft_prompt``'s triple goes straight into the state
    :meth:`Sam3Processor.set_image` built — which means reaching past ``set_text_prompt``
    into the grounding call underneath. The caller still owns *which* of its prompts the
    embed stands in for (:data:`_prompts.SUBJECT_PROMPT`).
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
) -> np.ndarray:
    """OR-combine SAM3's detections for every prompt into one binary mask.

    The whole of what a mask stage does with SAM3 between ``set_image`` and the mask it
    writes: prompt, drop anything under ``threshold``, union. Each prompt is either text
    (a ``str``, through SAM3's text encoder) or a soft prompt already loaded by
    ``load_soft_prompt`` (a ``dict``, grounded directly), so one list can mix the two.
    """
    import torch

    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    for prompt in prompts:
        if isinstance(prompt, dict):
            output = ground_with_soft_prompt(processor, model, state, prompt)
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

    Split out so a caller holding a model can rebuild the processor at a lower threshold
    without paying for the weights again.
    """
    stub_edt_kernel()
    from sam3.model.sam3_image_processor import Sam3Processor

    # The processor's own default is a literal "cuda"; follow the weights instead.
    kwargs = {"device": next(model.parameters()).device}
    if confidence_threshold is not None:
        kwargs["confidence_threshold"] = confidence_threshold
    return Sam3Processor(model, **kwargs)
