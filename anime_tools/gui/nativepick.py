"""The host's own folder/file chooser, behind the ``…`` on a path field.

Headless hosts and remote browsers have no native dialog, so :func:`pick` reports
``available=False`` and the caller falls back to ``/api/ls``. The dialog runs in a
subprocess (each toolkit wants the main thread, and an abandoned dialog must not
hold a server thread past :data:`TIMEOUT_S`); no shell is involved, and the two
scripted backends get their strings through :func:`_quote_as`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_S = 300
"""How long a chooser may stay open before the request gives up on it."""

MAX_TITLE = 80
"""The window title is client text, trimmed before it reaches a script."""

OWNER_VAR = "$owner"
"""The PowerShell variable holding the dialog's owner window.

A server-spawned process cannot take the Windows foreground, so an *unowned*
``ShowDialog()`` opens behind the browser with the request blocked on it. A
topmost owner fixes that without a foreground right: ``WS_EX_TOPMOST`` is a
Z-order band and an owned window is drawn above its owner.
"""

WIN_TOPMOST_OWNER = (
    f"{OWNER_VAR} = New-Object System.Windows.Forms.Form;"
    f"{OWNER_VAR}.TopMost = $true;"
    f"{OWNER_VAR}.ShowInTaskbar = $false;"
    f"{OWNER_VAR}.FormBorderStyle = 'None';"
    f"{OWNER_VAR}.Opacity = 0;"
    f"{OWNER_VAR}.Width = 1; {OWNER_VAR}.Height = 1;"
    f"{OWNER_VAR}.StartPosition = 'CenterScreen';"
    f"{OWNER_VAR}.Show(); {OWNER_VAR}.Activate();"
)
"""A one-pixel, borderless, transparent, taskbar-less window carrying the topmost
band for the real dialog. ``Activate`` only nudges the focus and may do nothing.
Nothing pumps messages for this form; ``ShowDialog`` runs its own modal loop."""

WIN_OWNER_CLOSE = f";{OWNER_VAR}.Close()"
"""Closes the owner after the dialog, cancel included, so no transparent topmost
window is leaked."""


@dataclass(frozen=True)
class Pick:
    """What came back from the desktop.

    ``available`` is about the *host*: ``False`` means there was no chooser to
    open; ``True`` with a ``None`` path is an ordinary cancel.
    """

    path: str | None
    available: bool


def _quote_as(kind: str, s: str) -> str:
    """One argument, escaped for the one scripted backend that takes it."""
    if kind == "applescript":
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return "'" + s.replace("'", "''") + "'"  # powershell single-quoted


def _clean_title(title: str) -> str:
    return "".join(c for c in title if c.isprintable())[:MAX_TITLE] or "anime_tools"


def _argv(kind: str, start: Path | None, title: str) -> list[str] | None:
    """The chooser this desktop has, or ``None`` if it has none."""
    if sys.platform == "darwin":
        verb = "choose folder" if kind == "dir" else "choose file"
        loc = (
            f" default location POSIX file {_quote_as('applescript', str(start))}"
            if start
            else ""
        )
        script = (
            f"POSIX path of ({verb} with prompt {_quote_as('applescript', title)}{loc})"
        )
        return ["osascript", "-e", script]

    if sys.platform == "win32":
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            return None
        q = _quote_as("powershell", str(start)) if start else "''"
        if kind == "dir":
            body = (
                "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
                f"$d.Description = {_quote_as('powershell', title)};"
                f"$d.SelectedPath = {q};"
                f"if ($d.ShowDialog({OWNER_VAR}) -eq 'OK')"
                " { [Console]::Out.Write($d.SelectedPath) }"
            )
        else:
            body = (
                "$d = New-Object System.Windows.Forms.OpenFileDialog;"
                f"$d.Title = {_quote_as('powershell', title)};"
                f"$d.InitialDirectory = {q};"
                f"if ($d.ShowDialog({OWNER_VAR}) -eq 'OK')"
                " { [Console]::Out.Write($d.FileName) }"
            )
        return [
            powershell,
            "-NoProfile",
            "-STA",
            "-Command",
            "Add-Type -AssemblyName System.Windows.Forms;"
            + WIN_TOPMOST_OWNER
            + body
            + WIN_OWNER_CLOSE,
        ]

    # X11/Wayland: a chooser without a display never returns, so the display is
    # part of "is there one".
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    zenity = shutil.which("zenity") or shutil.which("qarma")
    if zenity:
        argv = [zenity, "--file-selection", f"--title={title}"]
        if kind == "dir":
            argv.append("--directory")
        if start:
            # The trailing separator makes GTK open *in* the directory rather
            # than with it typed into the name box.
            argv.append(f"--filename={start}{os.sep}")
        return argv
    kdialog = shutil.which("kdialog")
    if kdialog:
        sub = "--getexistingdirectory" if kind == "dir" else "--getopenfilename"
        return [kdialog, sub, str(start or Path.home()), "--title", title]
    return None


def available() -> bool:
    """Is there a chooser on this host at all?"""
    return _argv("dir", None, "anime_tools") is not None


def pick(kind: str = "dir", start: Path | None = None, *, title: str = "") -> Pick:
    """Open the host's chooser and wait for it.

    Blocks for as long as the person in front of the dialog takes, so the route
    awaits it on a thread. Cancel, timeout and an empty answer all mean "nothing
    chosen".
    """
    argv = _argv(kind if kind == "dir" else "file", start, _clean_title(title))
    if argv is None:
        return Pick(None, available=False)
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        return Pick(None, available=True)
    except OSError:  # the binary went away between which() and exec
        return Pick(None, available=False)
    if proc.returncode != 0:  # cancelled
        return Pick(None, available=True)
    chosen = proc.stdout.splitlines()[0].strip() if proc.stdout.strip() else ""
    return Pick(chosen or None, available=True)
