"""The GUI's settings file: one JSON blob beside the curation home.

Kept out of :mod:`anime_tools.gui.server` so a plain CLI (``workspace.migrate``)
can read it without importing FastAPI. Stdlib plus :mod:`anime_tools._json`
only; the *shape* of the blob stays with the code that gives it meaning
(``gui.dataset.SETTINGS_KEY``, ``gui.stages.SETTINGS_KEY``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anime_tools._env import curation_home
from anime_tools._json import read_json, write_json

SETTINGS_NAME = ".anime_tools_gui.json"


def settings_path() -> Path:
    return curation_home() / SETTINGS_NAME


def load_settings() -> dict[str, Any]:
    """The saved settings, or ``{}``.

    Unreadable is the same as absent on purpose: a corrupted settings file must
    not stop the server from starting, and every value in it has a default.
    """
    p = settings_path()
    if p.exists():
        try:
            return read_json(p)
        except (OSError, ValueError):
            return {}
    return {}


def save_settings(data: dict[str, Any]) -> None:
    write_json(settings_path(), data)
