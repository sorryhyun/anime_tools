"""Torch device selection.

Every stage CLI's ``--device`` defaults to ``None`` = *auto*: the GUI is a
torch-free process and does not expose the flag, so the child decides for
itself. ``torch`` is imported inside the function to keep this module
importable without it.
"""

from __future__ import annotations

import argparse


def add_device_arg(p: argparse._ActionsContainer) -> None:
    """``--device`` — the flag :func:`resolve_device` answers.

    Its dest is in :data:`anime_tools.gui.stages.AUTO_FIELDS`, neither shown on
    the form nor sent on the argv, so every CLI must spell it identically and
    default it to ``None``. Takes a group as readily as a parser.
    """
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")


def resolve_device(name: str | None = None) -> str:
    """``name`` when the caller asked for one, else ``cuda`` if torch sees a
    GPU and ``cpu`` otherwise.

    An unimportable torch, or a probe that raises on a broken driver, resolves
    to ``cpu``; the caller imports torch itself and fails with a better message.
    """
    if name:
        return str(name)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - a failed probe is just "no GPU"
        return "cpu"
