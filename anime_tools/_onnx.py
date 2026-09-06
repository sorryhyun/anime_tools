"""ONNX Runtime sessions — the one provider choice, shared by every stage on ORT.

Two model families here run on onnxruntime rather than torch: PP-OCRv6 and the
CTD gate (:mod:`anime_tools.ocr._onnx`), and the dbv4 tagger backbone once
``<ckpt>/dbv4.onnx`` has been exported (:mod:`anime_tools.tagger.dbv4_onnx`).
Both ask the same two questions — *is there a GPU provider* and *which providers
does this session get* — so both are answered here, once.

Stdlib only at import time: ``onnxruntime`` is imported inside the functions, so
a torch-free process can import this module to ask what it would get.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def preload_cuda_libs(ort: Any) -> None:
    """Put the ``nvidia-*`` wheels' CUDA/cuDNN libraries on the loader path.

    Idempotent. A failed preload leaves the provider list exactly as it was, which the
    caller already warns about. Absent on the CPU wheel and on onnxruntime < 1.21, hence
    the ``getattr``.
    """
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:  # pragma: no cover - depends on the install
        return
    try:
        preload()
    except Exception:  # noqa: BLE001, S110 - a failed preload is just "no GPU"
        pass


def resolve_onnx_device(name: str | None = None) -> str:
    """``name`` when the caller asked for one, else ``cuda`` if **ORT** sees a GPU.

    :func:`anime_tools._device.resolve_device` answers the same question with
    ``torch.cuda.is_available()``, and for a stage that runs on onnxruntime that probe
    is not free: it imports torch and initialises CUDA, and torch's context then
    time-shares the device with ORT's for the life of the process. Measured on
    PP-OCRv6 over 160 images, 23 ms each against 40 — the probe cost 1.8x the run.

    So the runtime that will do the work is the one asked. A disagreement with
    :func:`make_session` is impossible, since both read the same provider list after
    the same preload; a missing provider answers ``cpu`` and the caller never asks for
    a GPU it cannot have.
    """
    if name:
        return str(name)
    try:
        import onnxruntime as ort

        preload_cuda_libs(ort)
        providers = ort.get_available_providers()
    except Exception:  # noqa: BLE001 - a failed probe is just "no GPU"
        return "cpu"
    return "cuda" if "CUDAExecutionProvider" in providers else "cpu"


CUDA_GPU_MEM_LIMIT_GB = 4.0
"""Ceiling on one session's CUDA arena, in GiB (``ANIME_TOOLS_ORT_GPU_MEM_GB``
overrides). See :func:`cuda_provider_options`."""


def cuda_provider_options() -> dict[str, Any]:
    """The CUDA provider's memory policy, the same for every session.

    ORT's defaults are a BFC arena that grows by power-of-two chunks and never
    shrinks, plus an *exhaustive* cuDNN algorithm search that allocates its own
    workspace per new input shape. Measured on the AnimeText detector fed each
    page at its native size (2026-09-06): the arena reached 15 GB over 351 pages
    and the next process on the card died in ``cublasCreate``. So every session
    extends its arena by exactly what a request needs, picks its convolution
    algorithms heuristically, and is capped — the cap is generous for the
    fixed-shape sessions this package runs (PP-OCRv6 at 1440² sits under 2 GB)
    and is a hard wall rather than a leak for anything shape-varying.
    """
    import os

    gb = float(os.environ.get("ANIME_TOOLS_ORT_GPU_MEM_GB", CUDA_GPU_MEM_LIMIT_GB))
    return {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
        "gpu_mem_limit": int(gb * 1024**3),
    }


def make_session(onnx_path: str | Path, device: str, *, what: str):
    """An ``InferenceSession`` on the GPU when one was asked for and is there.

    The CUDA provider is a separate wheel (``onnxruntime-gpu``), so asking for it where
    only the CPU build is installed warns — naming ``what`` fell back — and continues.
    On the GPU it runs under :func:`cuda_provider_options` — a bounded arena, for
    every session alike.

    ``preload_dlls()`` first, and it is not optional: the CUDA and cuDNN libraries the
    provider links against ship as their own ``nvidia-*`` wheels and nothing else puts
    them on the loader path. Without it the provider ``.so`` fails to open and
    ``get_available_providers()`` reports the *CPU* build's list.

    Only CPU and CUDA are ever asked for. CoreML is available on macOS and is *not*
    used: measured on the dbv4 backbone it took 165 partitions out of a 1208-node
    graph, ran at 1.13 s/img against the CPU provider's 0.49, and left the process
    crashing on teardown.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            f"onnxruntime is required for {what} but is not installed — `uv sync`. "
            "It is a declared dependency, split by platform marker "
            "(onnxruntime on macOS, onnxruntime-gpu elsewhere), so an "
            "environment missing it was not synced against the lockfile."
        ) from exc

    providers = ["CPUExecutionProvider"]
    if device.startswith("cuda"):
        preload_cuda_libs(ort)
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = [
                ("CUDAExecutionProvider", cuda_provider_options()),
                "CPUExecutionProvider",
            ]
        else:
            print(
                f"WARNING: onnxruntime CUDAExecutionProvider unavailable — {what} "
                "falls back to CPU (install onnxruntime-gpu)",
                flush=True,
            )
    return ort.InferenceSession(str(onnx_path), providers=providers)
