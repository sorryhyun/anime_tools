"""Self-update: what is installed, what the latest release is, and the one
command that moves between them.

``install.sh`` / ``install.ps1`` put this package in its *own* uv tool
environment (``uv tool install "anime-tools @ git+…@<tag>"``), so an update here
is not the file merge over a working tree that the trainer's ``scripts/update.py``
performs — there is no tree to merge and no baseline manifest to keep. It is the
same ``uv tool install --force`` at a newer tag, and the installed version is
whatever ``importlib.metadata`` says. That leaves three install shapes, and only
one of them may be rewritten from here:

``uv-tool``
    the installer's shape — an isolated tool environment this module owns.
``checkout``
    a git clone of this repo (``uv sync``); ``git pull`` owns that tree.
``other``
    a venv install, or the trainer's git dependency — its own resolver owns it.

Stdlib only (``urllib``, no ``requests``) and torch-free: the GUI server imports
this for the version row and the release check, and runs
``python -m anime_tools.update`` as a job for the upgrade itself.

An upgrade replaces the environment the running GUI was launched from. The
process keeps going (its imports are already resolved), but anything imported
lazily afterwards is reading files that moved — so a finished update always ends
with "restart ``anime-tools-gui``", and on Windows the running environment
cannot be replaced at all while it is held open.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from anime_tools import __version__

REPO = "sorryhyun/anime_tools"
REPO_URL = f"https://github.com/{REPO}"
RELEASES_URL = f"{REPO_URL}/releases"
_RELEASE_API = f"https://api.github.com/repos/{REPO}/releases"

PACKAGE = "anime-tools"
"""The distribution name — what ``uv tool`` knows this by."""

TOOL_PYTHON = "3.13"
"""Kept in step with ``install.sh``; ``requires-python`` is >= 3.13."""

NUMPY_OVERRIDE = "numpy>=2.0"
"""sam3's ``numpy<2`` pin is stale and ``[tool.uv] override-dependencies`` says
so — but uv reads ``tool.uv`` only from a workspace root, and installed this way
anime-tools is a dependency rather than the root. ``install.sh`` hands uv the
same override on the argv; so does :func:`update_argv`."""

# What `status()` answers with. "unknown" is the honest one: a checkout reports
# 0+unknown, and a non-numeric tag (v0.6.0rc1) is not something to order.
CURRENT = "current"
AVAILABLE = "available"
AHEAD = "ahead"
UNKNOWN = "unknown"

INSTALL_HINTS: dict[str, str] = {
    "checkout": "a git checkout of the repo — `git pull && uv sync` owns this tree",
    "other": (
        "installed by something other than `uv tool` (a venv, or the trainer's "
        "git dependency) — update it with the resolver that put it here"
    ),
}
"""Why an install shape is not ours to rewrite, shown where the button would be."""


@dataclass(frozen=True)
class Release:
    """One GitHub release: the tag, its notes as written, and its page."""

    tag: str
    notes: str = ""
    url: str = RELEASES_URL


def current_version() -> str:
    """The installed version, or ``0+unknown`` for a checkout on ``sys.path``."""
    return __version__


def parse_version(text: str) -> tuple[int, ...] | None:
    """``v0.6.0`` → ``(0, 6, 0)``; anything not purely numeric → ``None``.

    A local segment (``0+unknown``, the uninstalled checkout) and a pre-release
    tag are both ``None`` rather than a guess: an unorderable pair is reported
    as ``unknown``, never as an update the user does not have.
    """
    head = text.strip().lstrip("vV")
    if not head or "+" in head:
        return None
    parts = head.split(".")
    if not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def compare_versions(current: str, latest: str) -> str:
    """One of :data:`CURRENT` / :data:`AVAILABLE` / :data:`AHEAD` / :data:`UNKNOWN`."""
    cur, new = parse_version(current), parse_version(latest)
    if cur is None or new is None:
        return UNKNOWN
    if new > cur:
        return AVAILABLE
    return CURRENT if new == cur else AHEAD


def latest_release(tag: str | None = None, *, timeout: float = 15.0) -> Release:
    """The latest release, or the named tag's. Raises on any network failure.

    Unauthenticated GitHub is 60 requests an hour per IP, which is why the GUI
    caches this rather than asking per page load.
    """
    url = f"{_RELEASE_API}/latest" if tag is None else f"{_RELEASE_API}/tags/{tag}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "anime-tools-update",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return Release(
        tag=str(data.get("tag_name") or ""),
        notes=str(data.get("body") or "").strip(),
        url=str(data.get("html_url") or RELEASES_URL),
    )


def install_kind() -> str:
    """Which of the three shapes this process is running out of.

    ``uv tool`` writes a ``uv-receipt.toml`` at the root of every tool
    environment — the one marker that says "uv owns this venv and can replace
    it", which is exactly the question being asked.
    """
    if (Path(sys.prefix) / "uv-receipt.toml").is_file():
        return "uv-tool"
    root = Path(__file__).resolve().parent.parent
    if (root / ".git").exists() and (root / "pyproject.toml").is_file():
        return "checkout"
    return "other"


def can_update() -> bool:
    """Is the running install one this module may rewrite?"""
    return install_kind() == "uv-tool"


def update_argv(
    tag: str, *, overrides: str | os.PathLike[str], index: str | None = None
) -> list[str]:
    """The ``uv tool install`` that lands ``tag`` — ``install.sh``'s command.

    ``--force`` is what makes it an upgrade rather than "already installed";
    ``overrides`` is the file holding :data:`NUMPY_OVERRIDE`, and ``index`` is
    ``TORCH_INDEX`` for a CPU-only or Windows host.
    """
    argv = [
        "uv",
        "tool",
        "install",
        "--force",
        "--python",
        TOOL_PYTHON,
        "--overrides",
        str(overrides),
    ]
    if index:
        argv += ["--index", index]
    return [*argv, f"{PACKAGE} @ git+{REPO_URL}@{tag}"]


def status(tag: str | None = None, *, timeout: float = 15.0) -> dict[str, object]:
    """The whole version answer as the GUI shows it, network included.

    Raises whatever :func:`latest_release` raises; the caller decides what a
    failed check looks like.
    """
    release = latest_release(tag, timeout=timeout)
    current = current_version()
    kind = install_kind()
    return {
        "current": current,
        "latest": release.tag,
        "status": compare_versions(current, release.tag),
        "notes": release.notes,
        "url": release.url,
        "install": kind,
        "can_update": kind == "uv-tool",
        "hint": INSTALL_HINTS.get(kind, ""),
    }


def run_update(
    tag: str | None = None,
    *,
    index: str | None = None,
    dry_run: bool = False,
) -> int:
    """Install ``tag`` (default: the latest release) over this environment.

    Prints the command it runs and, whatever happens, the command to run by
    hand — a refusal, a missing uv and a Windows file lock all leave the user
    holding the same one line.
    """
    kind = install_kind()
    fallback = f'uv tool install --force "{PACKAGE} @ git+{REPO_URL}@<tag>"'

    if tag is None:
        try:
            tag = latest_release().tag
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"could not reach GitHub: {e}", file=sys.stderr)
            return 2
    print(f"anime_tools update — {current_version()} → {tag} ({kind})")

    if kind != "uv-tool":
        print(f"  refusing: {INSTALL_HINTS.get(kind, kind)}", file=sys.stderr)
        return 2
    uv = shutil.which("uv")
    if uv is None:
        print(f"  uv is not on PATH; install it, then: {fallback}", file=sys.stderr)
        return 2
    if os.name == "nt":
        # uv replaces the environment directory, and Windows will not let it
        # delete the python.exe this process is running from.
        print(
            "  note: on Windows the running environment cannot be replaced while "
            "the GUI holds it open. If uv reports a permission error, close the "
            f"GUI and run: {fallback.replace('<tag>', tag)}"
        )

    # --overrides wants a file, and the tool environment has no pyproject for
    # uv to read `[tool.uv] override-dependencies` out of.
    with tempfile.TemporaryDirectory(prefix="anime-tools-update-") as td:
        overrides = Path(td) / "overrides.txt"
        overrides.write_text(f"{NUMPY_OVERRIDE}\n", encoding="utf-8")
        argv = update_argv(tag, overrides=overrides, index=index)
        print(f"  $ {' '.join(argv)}")
        if dry_run:
            print("(dry run — nothing installed)")
            return 0
        code = subprocess.run([uv, *argv[1:]], check=False).returncode

    if code != 0:
        print(
            f"\nupdate failed (exit {code}); from a terminal with the GUI closed: "
            f"{fallback.replace('<tag>', tag)}",
            file=sys.stderr,
        )
        return code
    print(
        f"\ninstalled {tag}. Restart `anime-tools-gui` — this server is still "
        "running the old environment, and a stage started before the restart "
        "may not find its files."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m anime_tools.update",
        description=(
            "Update anime-tools to a GitHub release. Only the `uv tool` install "
            "the bootstrap installer makes is updated from here; a checkout is "
            "`git pull`'s and a venv is its own resolver's."
        ),
    )
    p.add_argument(
        "--version",
        metavar="TAG",
        help="Release tag to install (e.g. v0.6.0). Default: the latest release.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Report installed vs latest as JSON and exit, installing nothing",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the uv command that would run and exit",
    )
    p.add_argument(
        "--index",
        default=os.environ.get("TORCH_INDEX"),
        help="Extra package index for torch (CPU-only or Windows hosts); "
        "defaults to $TORCH_INDEX",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        try:
            print(json.dumps(status(args.version), indent=2, ensure_ascii=False))
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"could not reach GitHub: {e}", file=sys.stderr)
            return 2
        return 0
    return run_update(args.version, index=args.index, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
