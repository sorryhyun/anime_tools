"""Dataset browsing for the web GUI: the image/caption tree behind the sidebar.

Torch-free and Qt-free like the rest of ``anime_tools.gui`` — it only walks
paths and calls into the (torch-free) caption grammar, so the server process
stays light enough for ``tests/test_boundary.py``.

The trees it joins are keyed by the *same relative path* in each — three of
them today, and the two the workspace phase added beside them
(:mod:`anime_tools.workspace` is where the layout is written down):

``src``     ``image_dataset/<rel>``            source image + hand-written master caption
``master``  ``workspace/master/<rel>``         the revised master overlay (empty until Phase 2 fills it)
``dst``     ``workspace/resized/<rel>``        resized image + derived caption + ``.variants.txt``
``masks``   ``workspace/masks/<rel>``          ``{stem}_mask.png`` (nested; flat is the legacy fallback)
``out``     ``post_image_dataset/``            the export destination -- browsed by nothing, written by Export

Only ``master`` and ``derived`` are writable. ``.variants.txt`` is generated
(``# … do not hand-edit``) and is served read-only; editing a derived caption
makes its sidecar stale, which :func:`write_caption` reports so the UI can say
so out loud.

The sidebar has a second way of ordering the same rows — :func:`load_groups`
reads the grouping stage's ``groups.json`` and hands the client the near-twin
components, keyed by the same rel. It stays a *listing* of rels, not a second
listing of rows: group view joins it against the one ``list_items`` payload the
tree view already has, so a filter, a truncation and the run's pending dots mean
the same thing in both modes.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from anime_tools import workspace as WS
from anime_tools._env import curation_home, resolve_path
from anime_tools._json import read_json
from anime_tools._walk import IMAGE_EXTENSIONS, glob_images_pathlib
from anime_tools.captions.position_clauses import parse_caption
from anime_tools.captions.variants import read_variants_sidecar, variants_sidecar_path
from anime_tools.grouping.groups import MANIFEST_VERSION
from anime_tools.gui.settings import load_settings
from anime_tools.masking._masks import mask_name
from anime_tools.path_filter import filter_paths_by_glob

SETTINGS_KEY = "dataset"
DEFAULT_ROOTS = WS.DEFAULT_ROOTS
OUTPUT_ROOTS = WS.OUTPUT_ROOTS
EXPORT_ROOTS = WS.EXPORT_ROOTS
"""The layout, imported rather than restated: :mod:`anime_tools.workspace` is
the one place the default paths are written, so the GUI and the migrate CLI
cannot drift about where the workspace is."""
CAPTION_KINDS = ("master", "derived")
GROUPS_SUBPATH = "groups/groups.json"
"""The grouping manifest's tail under the Settings ``report_root`` — the same
split ``stages.report_subpath`` makes of ``build_groups``' own ``--out``
default, so the view reads exactly the file the **Groups** stage writes.
``tests/test_gui_groups.py`` pins the two together."""

MAX_ITEMS = 20000
"""Hard cap on one listing, and the default: a listing shows the whole dataset.

The sidebar renders lazily per folder, but the JSON still has to cross the wire
— past this the answer is ``path_pattern``, not a bigger payload. It used to sit
behind a 2000 default that nothing overrode, which made the cap unreachable and
truncated any real dataset silently. It is one number because the tree and the
group orderings draw the *same* listing (see the module docstring): a cap that
differed between them would make "truncated" mean two things.
"""


class DatasetError(ValueError):
    """Bad root / rel path — the server turns this into a 400 or 404."""


@dataclass(frozen=True)
class Roots:
    """The five dataset roots of one request, resolved and containment-checked.

    Field order is :data:`DEFAULT_ROOTS`' order (input, workspace, output),
    which is also the order ⚙ Settings lists them in. :meth:`items` walks that
    mapping rather than a hand-written tuple, so a sixth root would only have
    to be declared once.
    """

    src: Path
    master: Path
    dst: Path
    masks: Path
    out: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        return tuple((name, getattr(self, name)) for name in DEFAULT_ROOTS)

    def as_dict(self) -> dict[str, Any]:
        return {
            name: {"path": rel_to_home(p), "exists": p.is_dir()}
            for name, p in self.items()
        }


def lexical(path: str | Path) -> Path:
    """``resolve_path`` with ``..`` collapsed, and nothing else.

    Lexical (``normpath``), not ``resolve()``: a dataset root is routinely a
    symlink to where the images actually live, and following it would report
    the tree under a name the user never typed — and would defeat the
    containment test below, which compares the path the user gave against the
    trees they said they use.
    """
    return Path(os.path.normpath(resolve_path(path)))


def dataset_bases() -> tuple[Path, ...]:
    """Every tree this panel may reach: the curation home, plus any dataset root
    the **saved** settings pin outside it.

    The home alone was the rule until roots were allowed out of it, and it was
    only ever a proxy for the real one: what the panel may list, thumbnail and
    serve is *the dataset it is showing*. A curation home beside the trainer's
    checkout (``anime_tools/`` next to ``anima_lora/``) makes
    ``../anima_lora/image_dataset`` the ordinary answer for ``src``, and there
    is no home that contains both without swallowing everything else beside
    them.

    **Saved**, never a request's own root overrides: a root outside the home
    widens what may be read, so only the explicit Settings save that means it
    gets to do the widening (:func:`resolve_roots` with ``trusted``). A query
    param can still name any root it likes, but it is checked against these
    bases like every other path.
    """
    home = curation_home()
    bases = [home]
    saved = load_settings().get(SETTINGS_KEY) or {}
    for name in DEFAULT_ROOTS:
        raw = str(saved.get(name) or "").strip()
        if not raw:
            continue
        p = lexical(raw)
        if not p.is_relative_to(home):
            bases.append(p)
    return tuple(bases)


def reachable(path: str | Path) -> Path:
    """``lexical`` + the containment rule every read enforces.

    ``..`` is collapsed *before* the test: ``is_relative_to`` is purely textual,
    so ``<home>/../elsewhere`` would otherwise sail through it.
    """
    p = lexical(path)
    bases = dataset_bases()
    if not any(p.is_relative_to(b) for b in bases):
        raise DatasetError(f"outside the curation home and the dataset roots: {p}")
    return p


def rel_to_home(p: Path) -> str:
    home = curation_home()
    return p.relative_to(home).as_posix() if p.is_relative_to(home) else p.as_posix()


def resolve_roots(
    values: dict[str, Any] | None = None, *, trusted: bool = False
) -> Roots:
    """Roots from ``values`` (a ``settings["dataset"]`` blob or query params),
    falling back to :data:`DEFAULT_ROOTS`. Blank strings fall back too, so an
    emptied form field means "default", not "the home directory".

    ``trusted`` is the Settings **save** and nothing else: that request is what
    *defines* :func:`dataset_bases`, so it cannot be checked against them —
    a root outside the home could never be set a first time. Every other
    caller (every read request, through ``server.roots_for``) is checked, which
    is what keeps a query param from pointing the listing at a stranger's tree.
    """
    got = values or {}
    check = lexical if trusted else reachable
    paths = {}
    for name, default in DEFAULT_ROOTS.items():
        raw = got.get(name)
        paths[name] = check(str(raw).strip() if raw else default)
    return Roots(**paths)


def owned(p: Path) -> bool:
    """Is this a directory the panel may *create*?

    The panel reads the trees you point it at and creates only what is under
    its own home. A root outside the home is one that already exists — a
    sibling checkout's ``image_dataset`` — and a typo in one is a missing root
    the Settings row says is missing, not a new empty directory somewhere out
    in the filesystem.
    """
    return p.is_relative_to(curation_home())


def ensure_roots(roots: Roots) -> list[str]:
    """Create the root directories, returning the names actually made.

    Every root but :data:`EXPORT_ROOTS`' — an ``out`` tree that exists should
    mean an export happened, and Export makes its own destination — and only
    the ones this panel :func:`owned`.

    Only ever called from an *explicit write* — saving the Settings dialog —
    never from :func:`resolve_roots`, which every read request goes through: a
    listing must keep reporting a missing root as missing (``list_items`` says
    ``missing: True``) instead of quietly conjuring an empty tree behind a
    typo.
    """
    made = []
    for name, p in roots.items():
        if name in EXPORT_ROOTS or not owned(p):
            continue
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
            made.append(name)
    return made


def ensure_output_dir(path: str | Path) -> Path | None:
    """Best-effort mkdir for a directory a job is about to *write* to.

    Returns the path when it exists afterwards, ``None`` when it is not one the
    panel :func:`owned` or could not be created — a stage that mkdirs its own
    output (most do) or fails loudly is a better error than a 500 from here.
    """
    try:
        p = reachable(path)
        if not owned(p):
            return None
        p.mkdir(parents=True, exist_ok=True)
    except (DatasetError, OSError):
        return None
    return p


def _rel_key(rel: str) -> Path:
    """Validate a client-supplied relative image path.

    Rejects absolute paths and any ``..`` segment before it is ever joined to a
    root — ``reachable`` catches escapes too, but only after the join, and a
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
    roots: Roots,
    *,
    pattern: str | None = None,
    query: str = "",
    limit: int = MAX_ITEMS,
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
    """One sidebar row: the image plus which of its siblings exist.

    ``resized`` is matched on *stem* through :func:`_sibling_image`, like every
    other read of the derived tree — resize may re-encode a ``.jpg`` master to a
    ``.png``, so the rel that names the row is not the name of its own output.
    It is a row flag rather than a caption dot because it is an image, not a
    caption: nothing selects it, it only says whether the stages downstream of
    resize can see this image at all.
    """
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
        "resized": _sibling_image(roots.dst / rel.parent, rel.stem) is not None,
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


def load_groups(report_root: str) -> dict[str, Any]:
    """The grouping manifest under ``report_root``, as the sidebar's group view.

    Rels only — the group view draws the *same* rows the tree view does, joined
    against the one ``/api/dataset`` listing on the client, so a filter or a
    truncation cannot mean two different things depending on which mode is up.

    A missing manifest is not an error: the **Groups** stage has simply not run
    yet, and the panel says so and points at it. A manifest built against some
    other source tree is the one failure worth naming out loud (its rels join
    onto nothing), so ``source_dir`` rides along for the client to show.
    """
    path = reachable(f"{report_root}/{GROUPS_SUBPATH}")
    out: dict[str, Any] = {
        "path": rel_to_home(path),
        "missing": True,
        "stale": False,
        "source_dir": "",
        "groups": [],
    }
    if not path.is_file():
        return out
    try:
        data = read_json(path)
    except (OSError, ValueError) as e:
        raise DatasetError(f"unreadable grouping manifest {out['path']}: {e}") from e
    if not isinstance(data, dict):
        raise DatasetError(f"not a grouping manifest: {out['path']}")
    src = str(data.get("source_dir") or "")
    out.update(
        missing=False,
        # A v1 manifest still lists usable components; it is the knobs that
        # moved, so this is a "rebuild me" note, not a reason to show nothing.
        stale=data.get("version") != MANIFEST_VERSION,
        source_dir=rel_to_home(Path(src)) if src else "",
        groups=[
            {
                "id": int(g.get("id", i)),
                "artist": str(g.get("artist") or ""),
                "mean_cosine": g.get("mean_cosine"),
                "members": [str(m) for m in (g.get("members") or [])],
            }
            for i, g in enumerate(data.get("groups") or [])
            if isinstance(g, dict)
        ],
    )
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
    p = reachable(path)
    if not p.is_file():
        raise DatasetError(f"not found: {path}")
    return _thumb_bytes(str(p), p.stat().st_mtime, max(16, min(int(size), 1024)))
