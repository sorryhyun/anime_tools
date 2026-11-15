"""Dataset browsing for the web GUI: the image/caption tree behind the sidebar.

The trees it joins are keyed by the *same relative path* in each
(:mod:`anime_tools.workspace` owns the layout):

``src``     ``<source>/<rel>``                 source image + hand-written master caption
``master``  ``workspace/master/<rel>``         the revised master overlay: where a master edit lands, read before ``src``
``dst``     ``workspace/resized/<rel>``        resized image + revised caption + ``.variants.txt`` + ``.history.txt``
``masks``   ``workspace/masks/<rel>``          ``{stem}_mask.png`` (nested; flat is the legacy fallback; ``<out>/masks`` when neither)
``out``     ``<export>/``                      the export destination, written by Export

Beside them sits ``workspace/_excluded/``, which is not a root and does not join
by ``rel``: it is where :mod:`anime_tools.exclude` *moves* an image the curator
took out of the pipeline. A row still lists — you have to be able to see one to
put it back — but wearing ``excluded``, and every one of its files has left the
three trees above.

An image's captions are a **ladder** (:data:`CAPTION_LADDER`): the hand-written
master, the revised caption, the variants generated beside it, and last the
versions that caption used to be. That order feeds both the dots on a sidebar row
(:func:`_row`) and the badges over the caption editor (:func:`caption_versions`).
The last rung is last because it is not a text among texts: the panel draws it as
the diff of what changed. Only rungs marked ``editable`` can be written; editing
the caption beside ``.variants.txt`` makes that sidecar stale, which
:func:`write_caption` reports.

This module is the façade and the listing: every name the routes and tests reach
as ``D.<name>`` is here, re-exported from the private siblings that own a slice
— :mod:`~anime_tools.gui._dataset_roots` (roots, reachability, the ``rel`` gate,
the exclusion trees), :mod:`~anime_tools.gui._dataset_captions` (the ladder),
:mod:`~anime_tools.gui._dataset_reports` (groups manifest, analysis) and
:mod:`~anime_tools.gui._dataset_temp` (the drop area) — while the sidebar rows,
the item panel and thumbnails live below.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from anime_tools import workspace as WS
from anime_tools._walk import (
    filter_paths_by_glob,
    glob_images_pathlib,
    sibling_image,
)
from anime_tools.captions.ocr_sidecar import (
    ocr_sidecar_path,
    read_ocr,
    usable_lines,
)
from anime_tools.exclude import (
    ExclusionError,
    exclude_one,
    restore_one,
)
from anime_tools.gui._dataset_captions import (
    CAPTION_KINDS,
    CAPTION_LADDER,
    HISTORY_OF,
    Rung,
    caption_paths,
    caption_versions,
    caption_write_path,
    ladder_schema,
    parsed_caption,
    write_caption,
)
from anime_tools.gui._dataset_reports import (
    ANALYSIS_KINDS,
    GROUPS_SUBPATH,
    POSITION_SUBPATH,
    load_analysis,
    load_groups,
)
from anime_tools.gui._dataset_roots import (
    DEFAULT_ROOTS,
    EXPORT_ROOTS,
    OUTPUT_ROOTS,
    SETTINGS_KEY,
    DatasetError,
    Roots,
    _rel_key,
    dataset_bases,
    ensure_output_dir,
    ensure_roots,
    excluded_entries,
    excluded_trees,
    item_pattern,
    lexical,
    owned,
    reachable,
    rel_for_image,
    rel_to_home,
    resolve_roots,
)
from anime_tools.gui._dataset_temp import (
    DATASET_SPACE,
    SPACES,
    TEMP_SPACE,
    clear_temp,
    delete_temp_image,
    save_temp_file,
    temp_home,
    temp_roots,
    temp_summary,
)
from anime_tools.masking._mask_paths import find_mask

__all__ = [
    "ANALYSIS_KINDS",
    "CAPTION_KINDS",
    "CAPTION_LADDER",
    "DATASET_SPACE",
    "DEFAULT_ROOTS",
    "EXPORT_ROOTS",
    "GROUPS_SUBPATH",
    "HISTORY_OF",
    "MAX_ITEMS",
    "OUTPUT_ROOTS",
    "POSITION_SUBPATH",
    "SETTINGS_KEY",
    "SPACES",
    "TEMP_SPACE",
    "WS",
    "DatasetError",
    "Roots",
    "Rung",
    "caption_paths",
    "caption_versions",
    "caption_write_path",
    "clear_temp",
    "dataset_bases",
    "delete_temp_image",
    "drawn_on",
    "ensure_output_dir",
    "ensure_roots",
    "excluded_entries",
    "excluded_trees",
    "item_detail",
    "item_pattern",
    "item_rows",
    "ladder_schema",
    "lexical",
    "list_items",
    "load_analysis",
    "load_groups",
    "mask_path",
    "ocr_lines",
    "ocr_path",
    "owned",
    "parsed_caption",
    "reachable",
    "rel_for_image",
    "rel_to_home",
    "resolve_roots",
    "save_temp_file",
    "set_excluded",
    "temp_home",
    "temp_roots",
    "temp_summary",
    "thumbnail",
    "write_caption",
]


MAX_ITEMS = 20000
"""Hard cap on one listing, and the default. Past this the answer is
``path_pattern``, not a bigger payload."""


def mask_path(roots: Roots, rel: Path) -> Path | None:
    """``masks/<subdir>/{stem}_mask.png``, or the legacy flat one; failing both,
    the same two under the published ``<out>/masks``.

    Each tree is ``masking._mask_paths.find_mask``, the lookup every mask reader
    shares. The published tree comes last, so a mask the trainer already holds
    shows before any workspace run, and a workspace mask still wins once one
    exists.
    """
    for root in (roots.masks, roots.out / "masks"):
        if (p := find_mask(root, rel)) is not None:
            return p
    return None


def set_excluded(
    roots: Roots, rel_str: str, *, excluded: bool, note: str = ""
) -> dict[str, Any]:
    """Take one image out of the pipeline, or put it back.

    An instant action like a caption write, not a run: it moves files and answers
    with the row as it now is, so the sidebar folds the result in rather than
    re-walking the tree. Excluding checks the source tree first — a rel with no
    image behind it would be a ledger row nothing ever matches — while restoring
    does not, since the point may be that the source went away.
    """
    rel = _rel_key(rel_str)
    trees = excluded_trees(roots)
    if excluded and not (roots.src / rel).is_file():
        raise DatasetError(f"not in the dataset: {rel.as_posix()}")
    try:
        result = (
            exclude_one(trees, rel.as_posix(), note=note)
            if excluded
            else restore_one(trees, rel.as_posix())
        )
    except ExclusionError as e:
        raise DatasetError(str(e)) from e
    except OSError as e:
        raise DatasetError(f"could not move {rel.as_posix()}: {e}") from e
    entries = excluded_entries(roots)
    return {
        **result.to_dict(),
        "row": _row(roots, rel, rel.name, excluded=rel.as_posix() in entries),
    }


def list_items(
    roots: Roots,
    *,
    pattern: str | None = None,
    query: str = "",
    limit: int = MAX_ITEMS,
) -> dict[str, Any]:
    """Flat, sorted image list for the sidebar; the client nests it by folder.

    Enumerates the *source* tree — the master is the dataset. Unlike
    ``_walk.walk_images`` this tolerates same-stem collisions: a browser must
    show a tree the stages would refuse to run on.
    """
    if not roots.src.is_dir():
        return {
            "root": rel_to_home(roots.src),
            "missing": True,
            "total": 0,
            "items": [],
            "ladder": ladder_schema(),
        }

    paths = glob_images_pathlib(roots.src, recursive=True)
    if pattern and pattern != "*":
        keep = filter_paths_by_glob([str(p) for p in paths], str(roots.src), pattern)
        paths = [p for p, k in zip(paths, keep) if k]
    if query:
        needle = query.strip().lower()
        paths = [
            p for p in paths if needle in p.relative_to(roots.src).as_posix().lower()
        ]

    total = len(paths)
    limit = max(1, min(int(limit), MAX_ITEMS))
    entries = excluded_entries(roots)
    items = []
    for p in sorted(paths, key=lambda p: (p.parent.as_posix().lower(), p.name.lower()))[
        :limit
    ]:
        rel = p.relative_to(roots.src)
        items.append(_row(roots, rel, p.name, excluded=rel.as_posix() in entries))
    return {
        "root": rel_to_home(roots.src),
        "missing": False,
        "total": total,
        "truncated": total > len(items),
        "items": items,
        "ladder": ladder_schema(),
    }


def _row(
    roots: Roots, rel: Path, name: str, *, excluded: bool = False
) -> dict[str, Any]:
    """One sidebar row: the image plus which of its siblings exist.

    ``captions`` is one flag per dotted :data:`CAPTION_LADDER` rung, one stat
    apiece so this stays cheap for a whole-dataset listing. ``resized`` is matched on *stem*
    (:func:`sibling_image`) and is a row flag rather than a caption dot: it says
    whether the stages downstream of resize can see this image.

    ``excluded`` is told rather than looked up, since it is one set membership
    for the whole listing (:func:`excluded_entries`). It explains the rest of the
    row: an excluded image has no resized copy and no mask, because they moved.
    """
    caps = caption_paths(roots, rel)
    parent = rel.parent.as_posix()
    return {
        "rel": rel.as_posix(),
        "dir": "" if parent == "." else parent,
        "name": name,
        "stem": rel.stem,
        "captions": {r.kind: caps[r.kind].is_file() for r in CAPTION_LADDER if r.dot},
        "resized": sibling_image(roots.dst / rel.parent, rel.stem) is not None,
        "mask": mask_path(roots, rel) is not None,
        "excluded": excluded,
    }


def item_rows(roots: Roots, rels: list[str]) -> list[dict[str, Any]]:
    """:func:`list_items` rows for named images only, so a run that touched 40
    captions costs 40 stats rather than a walk of the source root. An unreadable
    or vanished rel is dropped."""
    entries = excluded_entries(roots)
    out = []
    for raw in rels:
        try:
            rel = _rel_key(raw)
        except DatasetError:
            continue
        if not (roots.src / rel).is_file():
            continue
        out.append(_row(roots, rel, rel.name, excluded=rel.as_posix() in entries))
    return out


def _image_info(p: Path | None) -> dict[str, Any] | None:
    """One image file as the panel gets it: where it is, how big, how many pixels."""
    if p is None or not p.is_file():
        return None
    info: dict[str, Any] = {"path": rel_to_home(p), "bytes": p.stat().st_size}
    try:
        from PIL import Image

        with Image.open(p) as im:  # lazy: reads the header, not the pixels
            info["width"], info["height"] = im.size
    except (OSError, ValueError):
        # A corrupt file still belongs in the tree; the keys stay so the shape
        # never varies.
        info["width"] = info["height"] = None
    size = (info["width"], info["height"])
    info["pixels"] = size[0] * size[1] if None not in size else None
    return info


def ocr_lines(roots: Roots, rel: Path) -> list[dict[str, Any]]:
    """The lines of the image's ``{stem}.ocr.txt`` (:func:`ocr_path`), or ``[]``.

    Not a :data:`CAPTION_LADDER` rung: it holds the words *in the picture*, not a
    text that could be written back into the caption. Joined by the same relative
    path as every other root. A missing sidecar means no text was found.

    Each row carries one field the sidecar does not spell, ``usable``: whether
    :func:`~anime_tools.captions.ocr_sidecar.usable_lines` would let that line
    reach a published caption. The two floors behind it (detector confidence,
    glyph size) are the caption's own rule, so the answer is computed here and
    never re-derived in the browser — the panel only draws the line it got back
    differently.
    """
    path = ocr_path(roots, rel)
    lines = read_ocr(path) if path else []
    keep = {line.seq for line in usable_lines(lines)}
    return [{**line.to_dict(), "usable": line.seq in keep} for line in lines]


def ocr_path(roots: Roots, rel: Path) -> Path | None:
    """The workspace's ``{stem}.ocr.txt``, else the one under ``<out>/ocr`` —
    the trainer's own OCR tree, which Export never writes."""
    txt = rel.with_suffix(".txt")
    for root in (roots.workspace / WS.OCR_SUBDIR, roots.out / "ocr"):
        p = ocr_sidecar_path(root / txt)
        if p.is_file():
            return p
    return None


def drawn_on(roots: Roots, rel: Path, p: Path | None) -> Path | None:
    """The resized image a mask or OCR file at ``p`` was made on, whose pixels
    its geometry is in: the workspace's for a workspace file, the trainer's
    ``<out>/resized`` for a published one."""
    if p is None:
        return None
    ours = p.is_relative_to(roots.masks) or p.is_relative_to(roots.workspace)
    tree = roots.dst if ours else roots.out / "resized"
    return sibling_image(tree / rel.parent, rel.stem)


def item_detail(roots: Roots, rel_str: str) -> dict[str, Any]:
    """One image and everything hanging off it, for the item panel."""
    rel = _rel_key(rel_str)
    src_image = roots.src / rel
    if not src_image.is_file():
        raise DatasetError(f"not in the dataset: {rel.as_posix()}")
    parent = rel.parent.as_posix()
    entry = excluded_entries(roots).get(rel.as_posix())
    mask = mask_path(roots, rel)
    return {
        "rel": rel.as_posix(),
        "dir": "" if parent == "." else parent,
        "name": src_image.name,
        "stem": rel.stem,
        # The whole ledger row, not a flag: the panel says when it was excluded
        # and why, which is the part a curator coming back to it needs.
        "excluded": entry.to_dict() if entry else None,
        "image": _image_info(src_image),
        "resized": _image_info(sibling_image(roots.dst / rel.parent, rel.stem)),
        "mask": _image_info(mask),
        # What each is drawn over: a mask or a box is in the pixels of the
        # resized copy it was made on, not the source's.
        "mask_base": _image_info(drawn_on(roots, rel, mask)),
        "versions": caption_versions(roots, rel),
        "ocr": ocr_lines(roots, rel),
        "ocr_base": _image_info(drawn_on(roots, rel, ocr_path(roots, rel))),
    }


@lru_cache(maxsize=512)
def _thumb_bytes(path: str, mtime: float, size: int) -> bytes:
    # `mtime` is unused on purpose: it is part of the cache key, so an
    # overwritten file misses the cache instead of serving the stale thumb.
    from PIL import Image

    with Image.open(path) as im:
        im.draft("RGB", (size, size))  # JPEG fast path; no-op elsewhere
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = BytesIO()
        im.save(buf, "WEBP", quality=80, method=4)
    return buf.getvalue()


def thumbnail(path: str, size: int = 192) -> bytes:
    """Cached WEBP thumbnail. Keyed on mtime, so an overwritten file re-renders."""
    p = reachable(path)
    if not p.is_file():
        raise DatasetError(f"not found: {path}")
    return _thumb_bytes(str(p), p.stat().st_mtime, max(16, min(int(size), 1024)))
