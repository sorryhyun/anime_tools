"""The GUI's settings file: one JSON blob beside the curation home.

Split out of :mod:`anime_tools.gui.server` so a plain CLI can read it without
importing FastAPI — ``anime_tools.workspace.migrate`` has to know whether the
saved dataset roots still pin the pre-workspace paths, and the alternative was
retyping the filename in a second place.

Stdlib plus :mod:`anime_tools._json` only. The *shape* of the blob stays with
the code that gives it meaning: ``gui.dataset.SETTINGS_KEY`` owns the roots,
``gui.stages.SETTINGS_KEY`` the stage defaults, and ``server``'s settings-derived
helpers turn one read of this mapping into everything a request needs.
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
