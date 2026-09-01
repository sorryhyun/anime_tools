"""The GUI's settings file: one JSON blob beside the curation home.

Stdlib only, so a plain CLI (``workspace.migrate``) can read it without FastAPI.
The blob's shape lives with the code that reads it (``gui.dataset.SETTINGS_KEY``,
``gui.stages.SETTINGS_KEY``).
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
    """The saved settings, or ``{}``. Unreadable counts as absent."""
    p = settings_path()
    if p.exists():
        try:
            return read_json(p)
        except (OSError, ValueError):
            return {}
    return {}


def save_settings(data: dict[str, Any]) -> None:
    write_json(settings_path(), data)
