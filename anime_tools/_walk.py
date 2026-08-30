"""Curation-side dataset walking (pure ``pathlib`` / ``os``; torch-free).

Deliberate ≤50-line duplicates of the trainer's ``library.datasets.image_utils``
(``IMAGE_EXTENSIONS`` / ``glob_images_pathlib`` / ``_assert_unique_stems``),
``library.preprocess._dataset.walk_images``, ``library.io.walk.safe_walk`` and
``library.io.cache.caption_key`` — the curation package must not import those
trainer modules, and both sides walk the same caption master.
``tests/test_curation_walk_parity.py`` pins the two copies to identical
behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from anime_tools.path_filter import filter_paths_by_glob

IMAGE_EXTENSIONS: list[str] = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
IMAGE_EXTENSIONS.extend([ext.upper() for ext in IMAGE_EXTENSIONS])

# Optional-plugin formats, mirrored from the trainer's image_utils so both
# walkers see the same files (the parity test pins this).
try:
    import pillow_avif  # noqa: F401

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except Exception:
    pass
for _jxl_plugin in ("jxlpy", "pillow_jxl"):
    try:
        __import__(_jxl_plugin)
    except Exception:
        continue
    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])


def glob_images_pathlib(dir_path: Path, recursive: bool) -> list[Path]:
    out: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        out += list(
            dir_path.rglob("*" + ext) if recursive else dir_path.glob("*" + ext)
        )
    return sorted(set(out))


def assert_unique_stems(img_paths, source_label: str = "directory") -> None:
    """Raise if two image paths share a stem *within the same subfolder*
    (sidecars are stem-keyed per folder)."""
    seen: dict[tuple[str, str], str] = {}
    collisions: dict[tuple[str, str], list[str]] = {}
    for p in img_paths:
        key = (os.path.dirname(p), os.path.splitext(os.path.basename(p))[0])
        if key in seen:
            collisions.setdefault(key, [seen[key]]).append(p)
        else:
            seen[key] = p
    if collisions:
        lines = [
            f"  stem '{stem}' in {parent}: " + ", ".join(paths)
            for (parent, stem), paths in sorted(collisions.items())
        ]
        raise ValueError(
            f"Duplicate image stems within a single folder of {source_label}. "
            "Cache filenames are stem-keyed; rename one of the colliding "
            "files (or move it to a different subfolder).\n" + "\n".join(lines)
        )


def walk_images(
    data_dir: Path, recursive: bool = False, pattern: str | None = None
) -> list[Path]:
    """Enumerate dataset images, sorted; ``pattern`` is the ``path_pattern``
    glob (``|``-OR) against paths relative to ``data_dir``; ``None``/``"*"``
    keeps all. Same-folder stem collisions raise."""
    paths = glob_images_pathlib(Path(data_dir), recursive)
    if pattern and pattern != "*":
        keep = filter_paths_by_glob([str(p) for p in paths], str(data_dir), pattern)
        paths = [p for p, k in zip(paths, keep) if k]
    assert_unique_stems([str(p) for p in paths], source_label=str(data_dir))
    return paths


def safe_walk(
    top: str | os.PathLike, *, followlinks: bool = True
) -> Iterator[tuple[str, list[str], list[str]]]:
    """``os.walk`` that follows symlinks but never revisits a directory."""
    seen: set[str] = {os.path.realpath(top)}
    for dirpath, dirnames, filenames in os.walk(top, followlinks=followlinks):
        kept: list[str] = []
        for d in dirnames:
            real = os.path.realpath(os.path.join(dirpath, d))
            if real in seen:
                continue
            seen.add(real)
            kept.append(d)
        dirnames[:] = kept
        yield dirpath, dirnames, filenames


def caption_key(
    image_path: str | os.PathLike, image_dir: str | os.PathLike | None = None
) -> str:
    """Stable subdir-aware caption-index key: path relative to ``image_dir``,
    extension stripped, ``/``-separated; bare stem when unrelated."""
    stem_no_ext = os.path.splitext(os.fspath(image_path))[0]
    if image_dir is not None:
        try:
            rel = os.path.relpath(stem_no_ext, os.fspath(image_dir))
        except ValueError:
            rel = ""
        if rel and rel != "." and not rel.startswith(".."):
            return rel.replace(os.sep, "/")
    return os.path.basename(stem_no_ext)
