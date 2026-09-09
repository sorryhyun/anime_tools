"""The host's own desktop: its file chooser, and its file manager.

Two gestures reach out of the browser and onto the machine running the server —
the ``…`` on a path field (:func:`pick`) and the ↗ beside a name in the panel
(:func:`reveal`). Both are offered only to a browser on that same machine, which
is the route's rule (``_is_loopback``) and not this module's; what is decided
here is whether the *desktop* has anything to open, which headless hosts do not:
:func:`available` and :func:`can_reveal` answer that, and the panel hides the
gesture rather than opening a window nobody can see.

Everything runs as a subprocess (each toolkit wants the main thread, and an
abandoned dialog must not hold a server thread past :data:`TIMEOUT_S`); no shell
is involved, and the two scripted backends get their strings through
:func:`_quote_as`. The difference between the two halves is what they wait for:
a chooser blocks until someone answers it, a reveal only until the desktop takes
the request.
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
            # The console encoding is the codepage otherwise (cp949/cp932 on
            # a CJK Windows), and a chosen path outside it comes back
            # mojibake or undecodable. `pick` reads UTF-8; say so here.
            "[Console]::OutputEncoding = "
            "[System.Text.Encoding]::UTF8;"
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
            argv,
            capture_output=True,
            text=True,
            # Every chooser above is told to answer in UTF-8; decoding in
            # the platform codepage instead would raise on the first CJK
            # path. `replace` keeps an odd byte from losing the whole pick.
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Pick(None, available=True)
    except OSError:  # the binary went away between which() and exec
        return Pick(None, available=False)
    if proc.returncode != 0:  # cancelled
        return Pick(None, available=True)
    chosen = proc.stdout.splitlines()[0].strip() if proc.stdout.strip() else ""
    return Pick(chosen or None, available=True)


REVEAL_TIMEOUT_S = 15
"""How long a file manager may take to launch before the request gives up.

Every backend below hands the path to an already-running desktop process and
returns at once; the timeout is for the cold start of one that is not running
yet, and never waits on the window itself.
"""

FM_BUS = (
    "org.freedesktop.FileManager1",
    "/org/freedesktop/FileManager1",
    "org.freedesktop.FileManager1.ShowItems",
)
"""The freedesktop interface that selects a file inside its folder.

``xdg-open`` on a *file* opens it in whatever app claims the type — an image
viewer for a PNG, which is not what a reveal means. Only this D-Bus call says
"show me the folder with this item picked", so it is tried first and the parent
directory is the fallback.
"""


def _reveal_argv(p: Path) -> list[str] | None:
    """How this desktop shows ``p`` in its file manager, or ``None`` if it can't.

    A directory is opened; a file is *revealed* — its folder, with the file
    selected — so the two are different commands almost everywhere.
    """
    is_dir = p.is_dir()
    if sys.platform == "darwin":
        # -R reveals rather than opens, which for a file is the difference
        # between Finder and Preview. A directory wants the plain form.
        return ["open", str(p)] if is_dir else ["open", "-R", str(p)]

    if sys.platform == "win32":
        explorer = shutil.which("explorer") or "explorer"
        # No space after the comma: explorer parses `/select,<path>` as one
        # token and silently opens Documents if it is split.
        return [explorer, str(p)] if is_dir else [explorer, f"/select,{p}"]

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    dbus = shutil.which("dbus-send")
    if dbus and not is_dir:
        dest, path, method = FM_BUS
        return [
            dbus,
            "--session",
            f"--dest={dest}",
            "--type=method_call",
            path,
            method,
            f"array:string:{p.as_uri()}",
            "string:",
        ]
    xdg = shutil.which("xdg-open")
    if xdg:
        return [xdg, str(p if is_dir else p.parent)]
    return None


def can_reveal() -> bool:
    """Is there a file manager on this host to reveal into at all?

    Asked of a directory, which is the form every backend supports: a desktop
    that can open a folder can always show a file in one, even if only by
    falling back to the folder.
    """
    return _reveal_argv(Path(os.getcwd())) is not None


def reveal(p: Path) -> bool:
    """Show ``p`` in the host's file manager. True if the command launched.

    Never waits for the window. ``explorer`` exits ``1`` on success, so the
    return code is not read on any backend — what a caller can honestly report
    is whether the desktop took the request, not what it did with it.
    """
    argv = _reveal_argv(p)
    if argv is None:
        return False
    try:
        subprocess.run(argv, capture_output=True, timeout=REVEAL_TIMEOUT_S, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return True
