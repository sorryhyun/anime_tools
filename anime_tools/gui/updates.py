"""The Update pane's server half: the cached release check and the job that
installs one.

:mod:`anime_tools.update` is the whole of what an update *is* — this module only
decides how often the GUI is allowed to ask GitHub, and which job runs the
answer. Two settings keys, split so the browser's toggle can never overwrite the
server's cache on the way through ``PUT /api/settings``:

``auto_update``
    the checkbox: may the panel check on its own? Written by the browser,
    default on.
``update_check``
    the last answer GitHub gave and when — written here only.

Unauthenticated GitHub allows 60 requests an hour per IP, and the panel is
reloaded far more often than a release is cut, so a check older than
:data:`CACHE_TTL` is the only one that goes out; "Check now" (``force``) is the
way past it.
"""

from __future__ import annotations

import time
import urllib.error
from typing import Any

from anime_tools import update as U
from anime_tools.gui.jobs import Step
from anime_tools.gui.settings import load_settings, save_settings

AUTO_KEY = "auto_update"
CACHE_KEY = "update_check"
CACHE_TTL = 6 * 3600
"""Six hours: fresh enough to notice the release of the morning, quiet enough
that a page reload is never a request."""

JOB_PREFIX = "update:"
"""``update:<tag>`` — what tells an adopted job apart from a stage or a
download, the same way ``download:`` does."""


def auto_check(settings: dict[str, Any] | None = None) -> bool:
    """Is the panel allowed to check by itself? Anything but ``False`` is on."""
    saved = load_settings() if settings is None else settings
    return saved.get(AUTO_KEY, True) is not False


def cached(
    settings: dict[str, Any] | None = None, *, ttl: float | None = CACHE_TTL
) -> dict[str, Any] | None:
    """The saved answer, or ``None``. ``ttl=None`` accepts it at any age — a
    failed check would rather show yesterday's tag than nothing."""
    saved = load_settings() if settings is None else settings
    entry = saved.get(CACHE_KEY)
    if not isinstance(entry, dict) or not entry.get("latest"):
        return None
    at = entry.get("checked_at")
    if not isinstance(at, (int, float)):
        return None
    if ttl is not None and time.time() - float(at) > ttl:
        return None
    return entry


def _save(entry: dict[str, Any]) -> None:
    # Re-read rather than reusing the caller's copy: this runs on a worker
    # thread and a settings PUT may have landed while GitHub was answering.
    data = load_settings()
    data[CACHE_KEY] = entry
    save_settings(data)


def check(*, force: bool = False, ttl: float = CACHE_TTL) -> dict[str, Any]:
    """What the Update pane shows: the local facts always, the remote ones
    cached.

    The network is touched only when asked to (``force``) or when the checkbox
    is on and the cache has aged out — so a panel that loads with checking
    turned off, or offline, still renders the version row it already knows.
    """
    settings = load_settings()
    fresh = cached(settings, ttl=ttl)
    current = U.current_version()
    kind = U.install_kind()
    out: dict[str, Any] = {
        "current": current,
        "latest": "",
        "status": U.UNKNOWN,
        "notes": "",
        "url": U.RELEASES_URL,
        "install": kind,
        "can_update": kind == "uv-tool",
        "hint": U.INSTALL_HINTS.get(kind, ""),
        "auto_check": auto_check(settings),
        "checked_at": None,
        "checked": False,
        "error": "",
    }

    entry = fresh
    if force or (fresh is None and out["auto_check"]):
        try:
            release = U.latest_release()
            entry = {
                "latest": release.tag,
                "notes": release.notes,
                "url": release.url,
                "checked_at": int(time.time()),
            }
            _save(entry)
            out["checked"] = True
        except (urllib.error.URLError, OSError, ValueError) as e:
            # A stale answer beats a blank row; the error rides along beside it.
            out["error"] = str(getattr(e, "reason", None) or e)
            entry = cached(settings, ttl=None)

    if entry:
        out.update(
            latest=entry.get("latest", ""),
            notes=entry.get("notes", ""),
            url=entry.get("url") or U.RELEASES_URL,
            checked_at=entry.get("checked_at"),
        )
        out["status"] = U.compare_versions(current, str(out["latest"]))
    return out


def refusal() -> str | None:
    """Why the running install may not be rewritten from the panel, or ``None``.

    The pane greys its button on ``can_update`` and this is the 409 the route
    answers with — the same question asked once, so the two cannot disagree.
    """
    kind = U.install_kind()
    return None if kind == "uv-tool" else (U.INSTALL_HINTS.get(kind) or kind)


def steps(tag: str | None = None) -> list[Step]:
    """The job an Update button starts: one ``python -m anime_tools.update``.

    Passing no tag lets the child resolve the latest release itself, so the
    thing installed is what GitHub says now rather than what the cache said.
    """
    argv = ["--version", tag] if tag else []
    return [Step(U.__name__, argv, label="update")]
