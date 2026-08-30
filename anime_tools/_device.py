"""Torch device selection, in one place.

Every stage CLI's ``--device`` defaults to ``None`` = *auto*: the GUI does not
expose the flag at all (it is a torch-free process and has no business guessing
what hardware the child will find), so the child has to decide for itself.

Kept import-light on purpose — ``torch`` is imported inside the function, so
``anime_tools.gui`` and the other torch-free callers can import this module.
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
