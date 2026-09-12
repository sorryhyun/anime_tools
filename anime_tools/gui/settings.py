"""The GUI's settings file: one JSON blob beside the curation home.

Stdlib only, so a plain CLI (``workspace.migrate``) can read it without FastAPI.
The blob's shape lives with the code that reads it (``gui.dataset.SETTINGS_KEY``,
``gui.stages.SETTINGS_KEY``).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from anime_tools._env import curation_home
from anime_tools._json import read_json, write_json

SETTINGS_NAME = ".anime_tools_gui.json"

_LOCK = threading.RLock()
"""Serialises :func:`edit_settings` — the blob is read-modify-written from the
event loop (a settings PUT), from the threadpool (a job start recording its
form values) and from an update check's worker thread, so a plain load/save pair
loses whichever write lands second. Re-entrant: an edit nested inside another
sees the same file and does not deadlock."""


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
    with _LOCK:
        write_json(settings_path(), data)


@contextmanager
def edit_settings() -> Iterator[dict[str, Any]]:
    """Read-modify-write the blob under the lock: mutate what is yielded.

    The one way to change settings. A bare ``load_settings()`` → mutate →
    ``save_settings()`` re-reads only to narrow the window, and still drops a
    concurrent writer's keys; this holds the lock across both halves, so the two
    writes serialise instead. An exception inside the block writes nothing.
    """
    with _LOCK:
        data = load_settings()
        yield data
        write_json(settings_path(), data)
