"""The drop area: images dropped onto the window, a curation home of its own
under :data:`~anime_tools.workspace.TEMP`. Re-exported by
:mod:`anime_tools.gui.dataset`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from anime_tools import workspace as WS
from anime_tools._env import curation_home
from anime_tools._walk import IMAGE_EXTENSIONS, glob_images_pathlib
from anime_tools.captions.history import history_sidecar_path
from anime_tools.captions.variants import variants_sidecar_path
from anime_tools.contract import CAPTION_SIDECAR_SUFFIXES, is_caption_sidecar
from anime_tools.exclude import artifacts, write_entries
from anime_tools.gui._dataset_reports import ANALYSIS_KINDS, POSITION_SUBPATH
from anime_tools.gui._dataset_roots import (
    DEFAULT_ROOTS,
    DatasetError,
    Roots,
    _rel_key,
    excluded_entries,
    excluded_trees,
    reachable,
    rel_to_home,
)
from anime_tools.stages._analysis import analysis_paths

DATASET_SPACE = "dataset"
TEMP_SPACE = "temp"
SPACES = (DATASET_SPACE, TEMP_SPACE)
"""What a request is answered against: the dataset the roots name, or the drop
area under :data:`~anime_tools.workspace.TEMP`."""

_TEMP_SUFFIXES = frozenset(e.lower() for e in IMAGE_EXTENSIONS) | {".txt"}
"""What a drop may bring: images, and the captions beside them (which land as
the drop area's hand-written masters)."""


def temp_home() -> Path:
    """The drop area — a curation home of its own, home-relative like every
    default root rather than under ``ANIME_TOOLS_WORKSPACE``, so it is inside
    what :func:`reachable` allows wherever the workspace was pointed."""
    return curation_home() / WS.TEMP


def temp_roots() -> Roots:
    """:data:`DEFAULT_ROOTS` laid out again under :func:`temp_home`. Saved
    Settings roots do not move it: they name the dataset, which this is not."""
    home = temp_home()
    return Roots(
        **{name: home / default for name, default in DEFAULT_ROOTS.items()},
        workspace=home / WS.WORKSPACE,
    )


def temp_summary() -> dict[str, Any]:
    """Whether the drop area holds anything — what decides that the sidebar
    draws its third mode at all — and what a drop may bring: the suffixes
    :func:`save_temp_file` takes and the sidecar suffixes it refuses, so the
    browser can leave the rest of a dropped folder behind without spelling
    either list itself (``.avif`` / ``.jxl`` are in only where their plugin is)."""
    src = temp_roots().src
    n = len(glob_images_pathlib(src, recursive=True)) if src.is_dir() else 0
    return {
        "root": rel_to_home(src),
        "count": n,
        "accept": sorted(_TEMP_SUFFIXES),
        "sidecars": list(CAPTION_SIDECAR_SUFFIXES),
    }


def _free_stem(directory: Path, stem: str) -> str:
    """``stem``, or the first of ``stem_1``, ``stem_2``, … that no image or
    caption in ``directory`` is named after.

    Taken by stem and not by filename, since ``a.jpg`` dropped beside ``a.png`` is
    a same-stem pair the stages refuse; and caselessly, since on a case-insensitive
    filesystem ``A.png`` is ``a.png``.
    """
    taken = (
        {
            p.stem.lower()
            for p in directory.iterdir()
            if p.suffix.lower() in _TEMP_SUFFIXES and not is_caption_sidecar(p.name)
        }
        if directory.is_dir()
        else set()
    )
    name, n = stem, 0
    while name.lower() in taken:
        n += 1
        name = f"{stem}_{n}"
    return name


def save_temp_file(
    rel_str: str, data: bytes, *, unique: bool = False
) -> dict[str, Any]:
    """Write one dropped file into the drop area's source tree, at the path it
    had under whatever was dropped (a folder keeps its subfolders).

    An image or a caption only; a ``{stem}.*.txt`` sidecar is refused, since in
    the source tree it would be read as nothing at all. With ``unique`` a name
    the folder already holds becomes ``{stem}_1``, ``{stem}_2``, … (a second drop
    of ``a.png`` is a copy, not a replacement), and the answer's ``rel`` says
    where it landed — which is how the browser sends the caption that came with
    it to the same stem. Without it the file is written at ``rel`` as given.

    The pick and the write have no ``await`` between them, so two uploads in
    flight cannot both land on one free name.
    """
    rel = _rel_key(rel_str)
    if rel.suffix.lower() not in _TEMP_SUFFIXES or is_caption_sidecar(rel.name):
        raise DatasetError(f"not an image or a caption: {rel.as_posix()}")
    src = temp_roots().src
    if unique:
        rel = rel.with_name(_free_stem(src / rel.parent, rel.stem) + rel.suffix)
    dest = src / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return {"rel": rel.as_posix(), "image": rel.suffix.lower() != ".txt"}


def delete_temp_image(rel_str: str, report_root: str) -> dict[str, Any]:
    """Delete one dropped image and everything the drop area holds of it.

    The source image, its caption in the source and master trees with their
    history and variants sidecars, its resized copy, mask and OCR sidecar
    (:func:`~anime_tools.exclude.artifacts`, the same list an exclusion moves),
    its analysis records, and — for an image that was excluded — what went to
    ``_excluded`` along with its ledger row. All of it, because the name is free
    again afterwards: the next drop of ``a.png`` lands as ``a.png``, and must not
    open on the deleted one's caption or mask.

    Only the drop area: the dataset's images are a curation's input, and
    taking one out of it is an exclusion, which can be put back.
    """
    rel = _rel_key(rel_str)
    roots = temp_roots()
    if not (roots.src / rel).is_file():
        raise DatasetError(f"not in the temp area: {rel.as_posix()}")
    trees = excluded_trees(roots)
    key = rel.as_posix()
    paths: list[Path] = [roots.src / rel]
    for root in (roots.src, roots.master):
        caption = root / rel.with_suffix(".txt")
        paths += [
            caption,
            history_sidecar_path(caption),
            variants_sidecar_path(caption),
        ]
    paths += [p for _, p in artifacts(trees, key)]
    for sub in ANALYSIS_KINDS.values():
        paths += analysis_paths(
            reachable(f"{report_root}/{POSITION_SUBPATH}/{sub}"), key
        )
    entries = excluded_entries(roots)
    entry = entries.pop(key, None)
    if entry is not None:
        paths += [trees.excluded / slot for slot in entry.moved]
    removed = 0
    for p in paths:
        if p.is_file():
            p.unlink()
            removed += 1
    if entry is not None:
        write_entries(trees.excluded, entries)
    return {"rel": key, "removed": removed, "count": temp_summary()["count"]}


def clear_temp() -> dict[str, Any]:
    """Delete the drop area — the source tree and everything a run made from it.
    The dataset is never under it, so this is the whole of what clearing means."""
    removed = temp_summary()["count"]
    home = temp_home()
    if home.is_dir():
        shutil.rmtree(home)
    return {"removed": removed}
