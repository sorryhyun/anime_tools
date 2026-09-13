"""Browsing and editing the dataset: the sidebar's rows, one item's detail, the
caption write, the exclusion toggle, the tag panel, thumbnails and file reads.

The roots every route here reads come from one settings snapshot
(:class:`~anime_tools.gui._context.RunContext`), with the per-request
``src`` / ``dst`` / ``masks`` overrides folded in. ``D.DatasetError`` is left to
the app-wide 400 handler except where a route owes a 404 instead, which it says
at the call that produces it.
"""

from __future__ import annotations

import mimetypes
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse, Response

from anime_tools.gui import dataset as D
from anime_tools.gui import tags as T
from anime_tools.gui._context import (
    NO_CACHE,
    RunContext,
    mask_root,
    report_root,
    roots_for,
)
from anime_tools.gui.settings import edit_settings, load_settings

router = APIRouter()


def _overrides(body: dict[str, Any]) -> dict[str, str]:
    """The root spellings a POST/PUT body may carry, blanks included (they fall
    back to the saved roots)."""
    return {k: str(body.get(k) or "") for k in D.DEFAULT_ROOTS}


@router.get("/api/dataset/roots")
def dataset_roots() -> dict[str, Any]:
    settings = load_settings()
    roots = roots_for(settings)
    return {
        "roots": roots.as_dict(),
        "defaults": D.DEFAULT_ROOTS,
        # What a blank `report_root` / `mask_root` resolves to, as
        # Settings' placeholders.
        "report_root": report_root({}, roots),
        "mask_root": mask_root({}, roots),
    }


@router.put("/api/dataset/roots")
async def put_dataset_roots(request: Request) -> dict[str, Any]:
    body = await request.json()
    picked = {k: str(body.get(k) or "").strip() for k in D.DEFAULT_ROOTS}
    roots = D.resolve_roots(picked, trusted=True)
    # The only ``trusted`` resolve: saving Settings is what defines the trees
    # the panel may read (``dataset_bases``).
    try:
        created = D.ensure_roots(roots)
    except OSError as e:
        raise HTTPException(500, f"cannot create root: {e}") from e
    with edit_settings() as data:
        data[D.SETTINGS_KEY] = picked
    return {
        "roots": roots.as_dict(),
        "defaults": D.DEFAULT_ROOTS,
        "report_root": report_root({}, roots),
        "mask_root": mask_root({}, roots),
        "created": created,
    }


@router.get("/api/dataset")
def dataset_list(
    src: str = "",
    dst: str = "",
    masks: str = "",
    pattern: str = "",
    q: str = "",
    limit: int = D.MAX_ITEMS,
) -> dict[str, Any]:
    return D.list_items(
        RunContext.load(src=src, dst=dst, masks=masks).roots,
        pattern=pattern or None,
        query=q,
        limit=limit,
    )


@router.get("/api/dataset/groups")
def dataset_groups() -> dict[str, Any]:
    """The near-twin components the **Groups** stage wrote, rels only.

    The client joins them onto the ``/api/dataset`` rows it already has, so one
    filter and one truncation serve both sidebar orderings. The path is derived
    like a stage's report, so it follows Settings.
    """
    return D.load_groups(RunContext.load().report_root)


@router.get("/api/dataset/analysis")
def dataset_analysis(rel: str) -> dict[str, Any]:
    """The caption panel's analysis badge: what the position stage (and its
    multiview audit phase) last recorded about this image, masks included.
    Derived like the groups manifest, so it follows Settings."""
    return D.load_analysis(RunContext.load().report_root, rel)


@router.post("/api/dataset/items")
async def dataset_items(request: Request) -> dict[str, Any]:
    """Refresh named sidebar rows: re-stat what a job wrote rather than
    re-walking the source root."""
    body = await request.json()
    rels = [str(r) for r in (body.get("rels") or [])][: D.MAX_ITEMS]
    ctx = RunContext.load(**_overrides(body))
    return {"items": D.item_rows(ctx.roots, rels)}


@router.get("/api/dataset/item")
def dataset_item(
    rel: str, src: str = "", dst: str = "", masks: str = ""
) -> dict[str, Any]:
    ctx = RunContext.load(src=src, dst=dst, masks=masks)
    try:
        return D.item_detail(ctx.roots, rel, min_pixels=ctx.min_pixels)
    except D.DatasetError as e:
        # 404, not the app-wide 400: the roots resolved, this image is
        # simply not in the dataset.
        raise HTTPException(404, str(e)) from e


@router.put("/api/dataset/item")
async def put_dataset_item(request: Request) -> dict[str, Any]:
    body = await request.json()
    ctx = RunContext.load(**_overrides(body))
    try:
        return D.write_caption(
            ctx.roots,
            str(body.get("rel") or ""),
            str(body.get("kind") or ""),
            body.get("text") or "",
        )
    except OSError as e:
        raise HTTPException(500, f"write failed: {e}") from e


@router.post("/api/dataset/exclude")
async def dataset_exclude(request: Request) -> dict[str, Any]:
    """Take one image out of the pipeline, or put it back.

    An instant action, not a job: it moves the image's files between
    ``workspace/`` and ``workspace/_excluded/`` and answers with the row as
    it now is. There is nothing to run and nothing to undo from a report —
    the inverse gesture is the same route with ``excluded: false``.
    """
    body = await request.json()
    ctx = RunContext.load(**_overrides(body))
    return D.set_excluded(
        ctx.roots,
        str(body.get("rel") or ""),
        excluded=bool(body.get("excluded")),
        note=str(body.get("note") or ""),
    )


@router.post("/api/dataset/parse")
async def dataset_parse(request: Request) -> dict[str, Any]:
    """Parse an *unsaved* caption for the editor's live clause preview.

    The browser must never hand-split a caption on commas, so it asks the one
    grammar implementation (``position_clauses``) instead.
    """
    body = await request.json()
    return D.parsed_caption(str(body.get("text") or ""))


@router.get("/api/tags/describe")
def describe_tag(tag: str) -> dict[str, Any]:
    """What one Danbooru tag means — the caption panel's click-a-tag panel.

    Answers even when the KB is not downloaded (``installed: false``), so
    the panel can point at Settings > Models instead of erroring.
    """
    tag = tag.strip()
    if not tag:
        raise HTTPException(400, "tag is required")
    return T.describe(tag)


@router.post("/api/tags/groups")
def drop_groups_of(body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
    """Which drop group each of a caption's tags falls under — the caption
    editor's group view, asked only while that view is on.

    A plain ``def``, like the describe route: the first call loads the KB, and
    FastAPI runs it in the threadpool rather than on the loop a job streams on.
    """
    tags = body.get("tags")
    if not isinstance(tags, list):
        raise HTTPException(400, "tags must be a list")
    return T.groups(s for t in tags if (s := str(t).strip()))


@router.get("/api/thumb")
def thumb(path: str, size: int = 192) -> Response:
    try:
        data = D.thumbnail(path, size)
    except D.DatasetError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:  # unreadable / corrupt image
        raise HTTPException(415, f"cannot thumbnail: {e}") from e
    return Response(
        data,
        media_type="image/webp",
        headers={"cache-control": "max-age=3600"},
    )


@router.get("/api/files")
def files(path: str) -> FileResponse:
    # reachable collapses ".." before the containment test — resolve_path +
    # is_relative_to alone is purely textual and lets traversal through.
    try:
        p = D.reachable(path)
    except D.DatasetError as e:
        raise HTTPException(404, "not found") from e
    if not p.is_file():
        raise HTTPException(404, "not found")
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return FileResponse(p, media_type=mime, headers=NO_CACHE)
