"""The host's own folder/file chooser, behind the ``…`` on a path field.

The panel is a local tool: the browser and the dataset are on one machine, so
the desktop's own chooser is the honest answer to "which directory?". The
``/api/ls`` browser is the fallback for the two hosts no native dialog serves —
headless, or a browser elsewhere — which is why :func:`pick` reports *no chooser
here* (``available=False``) rather than raising.

The dialog runs in a **subprocess**, never in this process: each toolkit here
wants the main thread, and a chooser the user leaves open must not hold a
server thread past :data:`TIMEOUT_S`. Nothing here imports torch, and nothing
here runs a shell — the two scripted backends (AppleScript, PowerShell) get
their strings through :func:`_quote_as` because a path is data.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_S = 300
"""How long a chooser may stay open before the request gives up on it: an
abandoned dialog would otherwise hold a thread and a socket forever."""

MAX_TITLE = 80
"""The window title is client text (the panel's own translated string), so it
is trimmed to something a title bar can hold before it reaches a script."""


@dataclass(frozen=True)
class Pick:
    """What came back from the desktop.

    ``available`` is about the *host*, not the answer: ``False`` means there was
    no chooser to open and the caller should fall back, while ``True`` with a
    ``None`` path is an ordinary cancel and means the field keeps its value.
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
                "if ($d.ShowDialog() -eq 'OK') { [Console]::Out.Write($d.SelectedPath) }"
            )
        else:
            body = (
                "$d = New-Object System.Windows.Forms.OpenFileDialog;"
                f"$d.Title = {_quote_as('powershell', title)};"
                f"$d.InitialDirectory = {q};"
                "if ($d.ShowDialog() -eq 'OK') { [Console]::Out.Write($d.FileName) }"
            )
        return [
            powershell,
            "-NoProfile",
            "-STA",
            "-Command",
            "Add-Type -AssemblyName System.Windows.Forms;" + body,
        ]

    # X11/Wayland: a chooser without a display is a process that never returns,
    # so the display is part of "is there one".
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    zenity = shutil.which("zenity") or shutil.which("qarma")
    if zenity:
        argv = [zenity, "--file-selection", f"--title={title}"]
        if kind == "dir":
            argv.append("--directory")
        if start:
            # The trailing separator is what makes GTK open *in* the directory
            # rather than with it typed into the name box.
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

    Blocking by design, and slow by nature -- it is as slow as the person in
    front of the dialog -- so the route awaits it on a thread. A cancel, a
    timeout and an empty answer are all the same ordinary "nothing chosen".
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
