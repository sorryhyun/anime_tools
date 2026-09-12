"""Starting the server and opening the window: ``anime-tools-gui``'s ``main``.

Everything here is about the process rather than the API — which port it lands
on, which browser draws it, and whether closing that window takes the server with
it. The app itself is :func:`anime_tools.gui.server.create_app`'s.
"""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from pathlib import Path

from anime_tools._env import curation_home
from anime_tools.gui.server import ClientWatch, create_app

__all__ = ["main", "pick_port"]


def pick_port(host: str, preferred: int, *, tries: int = 50) -> int:
    """``preferred`` if bindable, else the first free port above it (``0`` = OS-chosen)."""
    import socket

    candidates = [0] if preferred == 0 else range(preferred, preferred + tries)
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise SystemExit(f"no free port in {preferred}..{preferred + tries - 1} on {host}")


_CHROMIUM_BINARIES = {
    "darwin": (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Vivaldi.app/Contents/MacOS/Vivaldi",
    ),
    "win32": (
        "chrome.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ),
    "linux": (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "brave-browser",
        "microsoft-edge",
    ),
}


def _chromium_binary() -> str | None:
    """First Chromium-family browser on this machine, or ``None`` — only
    Chromium understands ``--app=URL``."""
    import shutil
    import sys

    for cand in _CHROMIUM_BINARIES.get(sys.platform, _CHROMIUM_BINARIES["linux"]):
        if os.path.isabs(cand):
            if os.path.exists(cand):
                return cand
        elif (found := shutil.which(cand)) is not None:
            return found
    return None


def _open_app_window(url: str) -> None:
    """Open ``url`` as a chromeless app window, falling back to a browser tab.

    ``--app=`` reuses the running browser's default profile.
    """
    import subprocess

    binary = _chromium_binary()
    if binary is not None:
        try:
            subprocess.Popen(
                [binary, f"--app={url}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except OSError:
            pass
    webbrowser.open(url)


def _open_when_ready(host: str, port: int, url: str, *, timeout: float = 60.0) -> None:
    """Open ``url`` once the server accepts connections, rather than after a fixed
    delay that would race startup."""
    import socket
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    else:
        return
    _open_app_window(url)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="anime_tools web GUI")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose on the LAN")
    p.add_argument(
        "--port",
        type=int,
        default=8790,
        help="Preferred port; if busy, the next free one above it is used (0 = let the OS pick)",
    )
    p.add_argument(
        "--home",
        default=None,
        help="Curation home (the source tree, workspace/, models/ live "
        "here). Default: $ANIME_TOOLS_HOME, $ANIMA_HOME, or the CWD",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the GUI on start, in a chromeless Chromium app window if there is one",
    )
    p.add_argument(
        "--exit-with-window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop the server a few seconds after the last GUI window closes "
        "(default: on with --open, off without it)",
    )
    args = p.parse_args(argv)
    if args.home:
        os.environ["ANIME_TOOLS_HOME"] = str(Path(args.home).expanduser().resolve())

    import uvicorn

    port = pick_port(args.host, args.port)
    if port != args.port and args.port != 0:
        print(f"port {args.port} is in use; using {port}", flush=True)
    connect_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{connect_host}:{port}"
    print(f"anime_tools GUI → {url}   (home: {curation_home()})", flush=True)
    # The server is built before it exists, so the watch stops it through a
    # closure rather than holding it; nothing can fire before serve() is running.
    server: uvicorn.Server | None = None

    def stop() -> None:
        if server is not None:
            server.should_exit = True

    exit_with_window = (
        args.open if args.exit_with_window is None else args.exit_with_window
    )
    app = create_app(watch=ClientWatch(stop) if exit_with_window else None)
    if args.open:
        threading.Thread(
            target=_open_when_ready, args=(connect_host, port, url), daemon=True
        ).start()
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=port, log_level="warning")
    )
    server.run()


if __name__ == "__main__":
    main()
