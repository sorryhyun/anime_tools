"""The three routes that reach the host's desktop: the folder chooser, the
file-manager reveal, and the directory listing that stands in for the chooser
when there is none.

All three are localhost-only, because the window opens where the *server* is —
offered to a browser anywhere else they would hold the request on a dialog nobody
can see. ``/api/ls`` is the exception that still answers a remote client, inside
``D.dataset_bases()`` only, since it is the fallback path browser.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from anime_tools._env import curation_home, resolve_path
from anime_tools.gui import dataset as D
from anime_tools.gui import nativepick as NP
from anime_tools.gui._context import is_loopback

router = APIRouter()


def _within(p: Path) -> bool:
    """Is this path one the panel may show a stranger's browser?"""
    try:
        D.reachable(p)
    except D.DatasetError:
        return False
    return True


def _start_dir(path: str) -> Path | None:
    """Where the host's chooser should open, given whatever the field holds.

    The field may be relative, may name a file, and may not exist yet, so walk
    up to the first directory that *is* there; ``None`` lets the desktop decide.
    """
    try:
        p = resolve_path(path) if path.strip() else curation_home()
    except (OSError, ValueError):
        return None
    for cand in (p, *p.parents):
        if cand.is_dir():
            return cand
    return None


@router.post("/api/pick")
async def pick_path(request: Request) -> dict[str, Any]:
    """Open the *host's* folder/file chooser and answer with what it got.

    The dialog opens on the machine running this server, so it is offered only
    to a browser on that same machine; from anywhere else it would hold the
    request on a window nobody can see. That refusal and a host with no chooser
    both come back as ``available: false``, the cue to fall back to ``/api/ls``.

    The answer is home-relative under the home, absolute outside it.
    """
    if not is_loopback(request):
        return {"available": False, "path": None}
    body = await request.json()
    kind = "dir" if str(body.get("kind") or "dir") != "file" else "file"
    # The dialog blocks for as long as the person in front of it takes, so it
    # waits on a thread while the loop keeps serving the panel.
    res = await asyncio.to_thread(
        NP.pick,
        kind,
        _start_dir(str(body.get("path") or "")),
        title=str(body.get("title") or ""),
    )
    return {
        "available": res.available,
        "path": D.rel_to_home(Path(res.path)) if res.path else None,
    }


@router.post("/api/reveal")
async def reveal_path(request: Request) -> dict[str, Any]:
    """Show a path in the *host's* file manager: a folder opened, a file
    selected inside its own.

    The same machine rule as ``/api/pick`` — the window opens where the
    server is — and the same reachability rule as every other read: only the
    home and the roots pinned outside it (:func:`D.reachable`), so the button
    cannot be talked into revealing ``/etc``. Nothing is opened *with* an
    app: a file goes to the file manager selected, never to whatever claims
    its type, so a click here can only ever show a folder.
    """
    if not is_loopback(request):
        raise HTTPException(403, "not this machine")
    body = await request.json()
    try:
        p = D.reachable(str(body.get("path") or ""))
    except D.DatasetError as e:
        raise HTTPException(404, "not found") from e
    if not p.exists():
        raise HTTPException(404, "not found")
    # The desktop returns at once, but a cold file manager may not; the
    # thread keeps a slow launch off the loop.
    return {"revealed": await asyncio.to_thread(NP.reveal, p)}


@router.get("/api/ls")
def ls(request: Request, path: str = "") -> dict[str, Any]:
    """Directory listing for the fallback path browser.

    For a browser on *this* machine it walks anywhere, so a host with no native
    chooser can still point a root at a sibling tree; ``parent`` is how it goes
    up (the client joins names but never takes a path apart). From anywhere else
    it stays inside ``D.dataset_bases``.
    """
    local = is_loopback(request)
    try:
        p = (
            (D.lexical(path) if local else D.reachable(path))
            if path
            else curation_home()
        )
    except D.DatasetError as e:
        raise HTTPException(404, "not found") from e
    if not p.is_dir():
        raise HTTPException(404, "not found")
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    parent = p.parent
    return {
        # Home-relative inside the home, absolute outside it, "" at the home
        # itself; the picker joins names onto this and hands it back.
        "path": "" if p == curation_home() else D.rel_to_home(p),
        # None at the filesystem root and at the edge of what this client may
        # see; the ".." the picker draws is exactly this field.
        "parent": (
            D.rel_to_home(parent)
            if parent != p and (local or _within(parent))
            else None
        ),
        "entries": [
            {"name": e.name, "dir": e.is_dir()}
            for e in entries
            if not e.name.startswith(".")
        ][:500],
    }
