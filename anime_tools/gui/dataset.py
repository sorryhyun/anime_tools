"""Dataset browsing for the web GUI: the image/caption tree behind the sidebar.

Torch-free and Qt-free like the rest of ``anime_tools.gui`` — it only walks
paths and calls into the (torch-free) caption grammar, so the server process
stays light enough for ``tests/test_boundary.py``.

The three trees it joins are the ones ``docs/contract.md`` pins, keyed by the
*same relative path* in each:

``src``    ``image_dataset/<rel>``                    source image + hand-written master caption
``dst``    ``post_image_dataset/resized/<rel>``       resized image + derived caption + ``.variants.txt``
``masks``  ``post_image_dataset/masks/<rel>``         ``{stem}_mask.png`` (nested; flat is the legacy fallback)

Only ``master`` and ``derived`` are writable. ``.variants.txt`` is generated
(``# … do not hand-edit``) and is served read-only; editing a derived caption
makes its sidecar stale, which :func:`write_caption` reports so the UI can say
so out loud.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from anime_tools._env import curation_home, resolve_path
from anime_tools._walk import IMAGE_EXTENSIONS, glob_images_pathlib
from anime_tools.captions.position_clauses import parse_caption
from anime_tools.captions.variants import read_variants_sidecar, variants_sidecar_path
from anime_tools.masking._masks import mask_name
from anime_tools.path_filter import filter_paths_by_glob

SETTINGS_KEY = "dataset"
DEFAULT_ROOTS: dict[str, str] = {
    "src": "image_dataset",
    "dst": "post_image_dataset/resized",
    "masks": "post_image_dataset/masks",
}
CAPTION_KINDS = ("master", "derived")
MAX_ITEMS = 20000
"""Hard cap on one listing. The sidebar renders lazily per folder, but the JSON
still has to cross the wire — past this the answer is ``path_pattern``, not a
bigger payload."""


class DatasetError(ValueError):
    """Bad root / rel path — the server turns this into a 400 or 404."""


@dataclass(frozen=True)
class Roots:
    src: Path
    dst: Path
    masks: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            name: {"path": rel_to_home(p), "exists": p.is_dir()}
            for name, p in (("src", self.src), ("dst", self.dst), ("masks", self.masks))
        }


def under_home(path: str | Path) -> Path:
    """``resolve_path`` + the same containment rule ``/api/files`` enforces."""
    p = resolve_path(path)
    # Collapse ".." *before* the test: ``is_relative_to`` is purely textual, so
    # ``<home>/../elsewhere`` would otherwise sail through it — harmless while
    # every caller only read, load-bearing now that saving roots mkdirs them.
    # Lexical (``normpath``), not ``resolve()``: a symlinked dataset root has to
    # keep working.
    p = Path(os.path.normpath(p))
    home = curation_home()
    if not p.is_relative_to(home):
        raise DatasetError(f"outside the curation home: {p}")
    return p


def rel_to_home(p: Path) -> str:
    home = curation_home()
    return p.relative_to(home).as_posix() if p.is_relative_to(home) else p.as_posix()


def resolve_roots(values: dict[str, Any] | None = None) -> Roots:
    """Roots from ``values`` (a ``settings["dataset"]`` blob or query params),
    falling back to :data:`DEFAULT_ROOTS`. Blank strings fall back too, so an
    emptied form field means "default", not "the home directory"."""
    got = values or {}
    paths = {}
    for name, default in DEFAULT_ROOTS.items():
        raw = got.get(name)
        paths[name] = under_home(str(raw).strip() if raw else default)
    return Roots(**paths)


def ensure_roots(roots: Roots) -> list[str]:
    """Create the three root directories, returning the names actually made.

    Only ever called from an *explicit write* — saving the Settings dialog —
    never from :func:`resolve_roots`, which every read request goes through: a
    listing must keep reporting a missing root as missing (``list_items`` says
    ``missing: True``) instead of quietly conjuring an empty tree behind a
    typo. The paths are already :func:`under_home` by construction, so this
    cannot mkdir outside the curation home.
    """
    made = []
    for name, p in (("src", roots.src), ("dst", roots.dst), ("masks", roots.masks)):
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
            made.append(name)
    return made


def ensure_output_dir(path: str | Path) -> Path | None:
    """Best-effort mkdir for a directory a job is about to *write* to.

    Returns the path when it exists afterwards, ``None`` when it is outside the
    curation home or could not be created — a stage that mkdirs its own output
    (most do) or fails loudly is a better error than a 500 from here.
    """
    try:
        p = under_home(path)
        p.mkdir(parents=True, exist_ok=True)
    except (DatasetError, OSError):
        return None
    return p


def _rel_key(rel: str) -> Path:
    """Validate a client-supplied relative image path.

    Rejects absolute paths and any ``..`` segment before it is ever joined to a
    root — ``under_home`` catches escapes too, but only after the join, and a
    root can itself sit deep enough that ``..`` stays inside the home.
    """
    p = Path(str(rel).replace("\\", "/"))
    if p.is_absolute() or any(part == ".." for part in p.parts) or not p.parts:
        raise DatasetError(f"bad relative path: {rel!r}")
    return p


def item_pattern(rel: str) -> str:
    """A ``path_pattern`` matching exactly the one dataset image ``rel``.

    This is how the GUI runs a stage on the selected image: the stages have no
    "just this file" flag, and they should not grow one — narrowing the glob
    they already take means a one-image run and a batch run take the identical
    code path (a replay included, see ``stages.replay._keep_by_pattern``).

    ``<dir>/<stem>.*``, not the full filename: a stage matches the pattern
    against the *resized* tree and the resize step may re-encode (``.jpg``
    master → ``.png`` resized), so the extension has to be a wildcard. That
    cannot widen the match — :func:`~anime_tools._walk.assert_unique_stems`
    already refuses two images sharing a stem in one folder.

    fnmatch metacharacters in the path are escaped, so a literal ``[`` in a
    filename stays a ``[``; a literal ``|`` cannot be, because ``|`` is the
    pattern's own alternative separator, so such a name is refused outright
    rather than silently selecting nothing.
    """
    p = _rel_key(rel)
    if "|" in p.as_posix():
        raise DatasetError(
            f"cannot scope a run to {rel!r}: '|' separates path_pattern "
            "alternatives, so no pattern can name this file"
        )
    return glob.escape(p.with_suffix("").as_posix()) + ".*"


def _sibling_image(directory: Path, stem: str) -> Path | None:
    """The image named ``stem`` in ``directory``, whatever its extension.

    The resize step may re-encode (``.jpg`` master → ``.png`` resized), so the
    derived tree is matched on stem, not on the full relative path.
    """
    for ext in IMAGE_EXTENSIONS:
        p = directory / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def rel_for_image(roots: Roots, image: str) -> str | None:
    """The dataset rel a stage report's ``image`` names, or ``None``.

    Reports name images relative to the **resized** tree, and the resize step
    may have re-encoded on the way there (``.jpg`` master → ``.png`` resized),
    so the join back to the source tree is on directory + stem — the same rule
    :func:`_sibling_image` uses in the other direction. An image the source tree
    no longer has is dropped rather than raising: the caller is decorating a
    listing with what a run proposed, and a row it cannot place is one it should
    leave out.
    """
    try:
        rel = _rel_key(image)
    except DatasetError:
        return None
    if (roots.src / rel).is_file():
        return rel.as_posix()
    p = _sibling_image(roots.src / rel.parent, rel.stem)
    return p.relative_to(roots.src).as_posix() if p else None


def mask_path(roots: Roots, rel: Path) -> Path | None:
    """``masks/<subdir>/{stem}_mask.png``, or the legacy flat one.

    The name comes from :func:`anime_tools.masking._masks.mask_name`, the same
    one the generators write, so the two sides cannot drift. The flat fallback
    is this reader's alone: the generators have mirrored the source subdir
    since they grew ``--recursive``, but a mask tree made before that is still
    a valid one to browse.
    """
    name = mask_name(rel.stem)
    nested = roots.masks / rel.parent / name
    if nested.is_file():
        return nested
    flat = roots.masks / name
    return flat if flat.is_file() else None


def caption_paths(roots: Roots, rel: Path) -> dict[str, Path]:
    txt = rel.with_suffix(".txt")
    derived = roots.dst / txt
    return {
        "master": roots.src / txt,
        "derived": derived,
        "variants": variants_sidecar_path(derived),
    }


def list_items(
    roots: Roots, *, pattern: str | None = None, query: str = "", limit: int = 2000
) -> dict[str, Any]:
    """Flat, sorted image list for the sidebar; the client nests it by folder.

    Enumerates the *source* tree — the master is the dataset. Unlike
    ``_walk.walk_images`` this tolerates same-stem collisions: a browser must
    still show a tree the stages would refuse to run on.
    """
    if not roots.src.is_dir():
        return {
            "root": rel_to_home(roots.src),
            "missing": True,
            "total": 0,
            "items": [],
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
    items = []
    for p in sorted(paths, key=lambda p: (p.parent.as_posix().lower(), p.name.lower()))[
        :limit
    ]:
        items.append(_row(roots, p.relative_to(roots.src), p.name))
    return {
        "root": rel_to_home(roots.src),
        "missing": False,
        "total": total,
        "truncated": total > len(items),
        "items": items,
    }


def _row(roots: Roots, rel: Path, name: str) -> dict[str, Any]:
    """One sidebar row: the image plus which of its siblings exist."""
    caps = caption_paths(roots, rel)
    parent = rel.parent.as_posix()
    return {
        "rel": rel.as_posix(),
        "dir": "" if parent == "." else parent,
        "name": name,
        "stem": rel.stem,
        "master": caps["master"].is_file(),
        "derived": caps["derived"].is_file(),
        "variants": caps["variants"].is_file(),
        "mask": mask_path(roots, rel) is not None,
    }


def item_rows(roots: Roots, rels: list[str]) -> list[dict[str, Any]]:
    """:func:`list_items` rows for named images only.

    A stage that just wrote 40 captions changed 40 of the sidebar's rows, not
    the tree; re-walking the whole source root to learn that is the wrong shape.
    An unreadable or vanished rel is dropped rather than raising — the caller is
    patching a listing, and a row it cannot refresh is one it should leave be.
    """
    out = []
    for raw in rels:
        try:
            rel = _rel_key(raw)
        except DatasetError:
            continue
        if not (roots.src / rel).is_file():
            continue
        out.append(_row(roots, rel, rel.name))
    return out


def _image_info(p: Path | None) -> dict[str, Any] | None:
    if p is None or not p.is_file():
        return None
    info: dict[str, Any] = {"path": rel_to_home(p), "bytes": p.stat().st_size}
    try:
        from PIL import Image

        with Image.open(p) as im:  # lazy: reads the header, not the pixels
            info["width"], info["height"] = im.size
    except (OSError, ValueError):
        # A corrupt or unsupported file still belongs in the tree; it just has
        # no dimensions to report. The keys stay so the shape never varies.
        info["width"] = info["height"] = None
    return info


def parsed_caption(text: str) -> dict[str, Any]:
    """The caption grammar as JSON: flat bag + position clauses, never a
    ``split(",")`` on the client."""
    parsed = parse_caption(text)
    return {
        "flat_tags": list(parsed.flat_tags),
        "clauses": [
            {
                "header": c.header,
                "prefix": c.prefix,
                "position": c.position,
                "tags": list(c.tags),
            }
            for c in parsed.clauses
        ],
    }


def _caption_entry(kind: str, p: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "kind": kind,
        "path": rel_to_home(p),
        "exists": p.is_file(),
    }
    if entry["exists"]:
        text = p.read_text(encoding="utf-8").strip()
        entry["text"] = text
        entry["mtime"] = p.stat().st_mtime
        entry["parsed"] = parsed_caption(text)
    else:
        entry["text"] = ""
        entry["parsed"] = None
    return entry


def item_detail(roots: Roots, rel_str: str) -> dict[str, Any]:
    rel = _rel_key(rel_str)
    src_image = roots.src / rel
    if not src_image.is_file():
        raise DatasetError(f"not in the dataset: {rel.as_posix()}")
    caps = caption_paths(roots, rel)
    variants = caps["variants"]
    parent = rel.parent.as_posix()
    return {
        "rel": rel.as_posix(),
        "dir": "" if parent == "." else parent,
        "name": src_image.name,
        "stem": rel.stem,
        "image": _image_info(src_image),
        "resized": _image_info(_sibling_image(roots.dst / rel.parent, rel.stem)),
        "mask": _image_info(mask_path(roots, rel)),
        "captions": [_caption_entry(k, caps[k]) for k in CAPTION_KINDS],
        "variants": {
            "path": rel_to_home(variants),
            "exists": variants.is_file(),
            "rows": (
                [
                    {"label": lab, "text": txt}
                    for lab, txt in read_variants_sidecar(variants)
                ]
                if variants.is_file()
                else []
            ),
        },
    }


def write_caption(roots: Roots, rel_str: str, kind: str, text: str) -> dict[str, Any]:
    """Write one caption file. ``master`` and ``derived`` only.

    A caption is a single line by contract, so any newline the textarea picked
    up is folded to a space. An empty body is refused rather than treated as a
    delete — losing a caption should take more than a stray select-all.
    """
    if kind not in CAPTION_KINDS:
        raise DatasetError(f"not an editable caption: {kind!r}")
    rel = _rel_key(rel_str)
    if not (roots.src / rel).is_file():
        raise DatasetError(f"not in the dataset: {rel.as_posix()}")
    body = " ".join(str(text).split())
    if not body:
        raise DatasetError("refusing to write an empty caption")

    p = caption_paths(roots, rel)[kind]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")

    entry = _caption_entry(kind, p)
    # The sidecar was generated from the previous derived text, so v0 no longer
    # matches what the TE step would encode.
    sidecar = caption_paths(roots, rel)["variants"]
    entry["variants_stale"] = kind == "derived" and sidecar.is_file()
    return entry


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
    p = under_home(path)
    if not p.is_file():
        raise DatasetError(f"not found: {path}")
    return _thumb_bytes(str(p), p.stat().st_mtime, max(16, min(int(size), 1024)))
