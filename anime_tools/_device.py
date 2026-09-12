"""Torch device selection — the one probe, for every stage.

Every stage CLI's ``--device`` defaults to ``None`` = *auto*: the GUI is a
torch-free process and does not expose the flag, so the child decides for
itself. ``torch`` is imported inside the function to keep this module
importable without it.

Since 2026-09-09 every model here runs on torch, so this is the only device
question the package asks.
"""

from __future__ import annotations

import argparse

DEVICE_HELP = "cuda|mps|cpu (default: auto)"
"""The help every ``device`` field and flag carries. Its dest is in
:data:`anime_tools.gui.stages.AUTO_FIELDS`, neither shown on the form nor sent
on the argv, so every stage must spell it identically and default it to
``None``."""


def add_device_arg(p: argparse._ActionsContainer) -> None:
    """``--device`` — the flag :func:`resolve_device` answers, for the CLIs that
    are not request objects (the tagger's, the probes). Takes a group as
    readily as a parser."""
    p.add_argument("--device", default=None, help=DEVICE_HELP)


def resolve_device(name: str | None = None) -> str:
    """``name`` when the caller asked for one, else the best device torch sees:
    ``cuda``, then ``mps``, then ``cpu``.

    An unimportable torch, or a probe that raises on a broken driver, resolves
    to ``cpu``; the caller imports torch itself and fails with a better message.

    MPS comes second and not first because a machine with both is a machine with
    a real GPU. It is worth reaching for over a CPU by a wide margin — measured
    2026-09-09: the dbv4 tagger backbone 1799 → 87 ms an image, PE-Spatial
    2606 → 117, the AnimeText detector 392 → 75.
    """
    if name:
        return str(name)
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 - a failed probe is just "no GPU"
        return "cpu"
    return "cpu"
