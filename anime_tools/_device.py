"""Torch device selection, in one place.

Every stage CLI's ``--device`` defaults to ``None`` = *auto*: the GUI is a
torch-free process and does not expose the flag, so the child decides for
itself. ``torch`` is imported inside the function, so torch-free callers can
still import this module.
"""

from __future__ import annotations


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
