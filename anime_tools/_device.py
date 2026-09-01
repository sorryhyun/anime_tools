"""Torch device selection, in one place.

Every stage CLI's ``--device`` defaults to ``None`` = *auto*: the GUI is a
torch-free process and does not expose the flag, so the child decides for
itself. ``torch`` is imported inside the function, so torch-free callers can
still import this module.
"""

from __future__ import annotations

import argparse


def add_device_arg(p: argparse._ActionsContainer) -> None:
    """``--device`` — the flag :func:`resolve_device` answers.

    Beside the resolver because the flag and the resolution are one fact: the
    dest is :data:`anime_tools.gui.stages.AUTO_FIELDS`, which is *neither shown
    on the form nor sent on the argv*, so every CLI must spell it identically
    and default it to ``None`` or the GUI silently stops being able to leave it
    out. ``tests/test_stage_cli_args.py`` pins that across the stages; declaring
    it once is what makes the pin structural rather than a coincidence eleven
    parsers were maintaining by hand.

    Takes a group as readily as a parser, like
    :func:`anime_tools.masking._sam3.add_checkpoint_arg`.
    """
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")


def resolve_device(name: str | None = None) -> str:
    """``name`` when the caller asked for one, else ``cuda`` if torch sees a
    GPU and ``cpu`` otherwise.

    A torch that cannot even be imported (or whose CUDA probe raises, which a
    broken driver install does) resolves to ``cpu``: the caller is about to
    import torch anyway and will fail with its own, better message.
    """
    if name:
        return str(name)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - a failed probe is just "no GPU"
        return "cpu"
