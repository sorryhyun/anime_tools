"""Dataset browsing for the web GUI: the image/caption tree behind the sidebar.

The trees it joins are keyed by the *same relative path* in each
(:mod:`anime_tools.workspace` owns the layout):

``src``     ``image_dataset/<rel>``            source image + hand-written master caption
``master``  ``workspace/master/<rel>``         the revised master overlay (empty until Phase 2 fills it)
``dst``     ``workspace/resized/<rel>``        resized image + revised caption + ``.variants.txt`` + ``.history.txt``
``masks``   ``workspace/masks/<rel>``          ``{stem}_mask.png`` (nested; flat is the legacy fallback)
``out``     ``post_image_dataset/``            the export destination, written by Export

An image's captions are a **ladder** (:data:`CAPTION_LADDER`): the hand-written
master, the versions the revised caption used to be, that caption itself, then the
generated variants. That order feeds both the dots on a sidebar row (:func:`_row`)
and the badges over the caption editor (:func:`caption_versions`). Only rungs marked
``editable`` can be written; editing the caption above ``.variants.txt`` makes that
sidecar stale, which :func:`write_caption` reports.
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
from anime_tools._env import curation_home, resolve_path, workspace_dir
from anime_tools._json import read_json
from anime_tools._walk import IMAGE_EXTENSIONS, glob_images_pathlib
from anime_tools.captions.history import (
    history_sidecar_path,
    push_history,
    read_history,
)
from anime_tools.captions.ocr_sidecar import ocr_sidecar_path, read_ocr
from anime_tools.captions.position_clauses import parse_caption, tag_spans
from anime_tools.captions.variants import read_variants_sidecar, variants_sidecar_path
from anime_tools.grouping.groups import MANIFEST_VERSION
from anime_tools.gui.settings import load_settings
from anime_tools.masking._masks import mask_name
from anime_tools.path_filter import filter_paths_by_glob
from anime_tools.stages.resize import DEFAULT_MIN_PIXELS, below_min_pixels

SETTINGS_KEY = "dataset"
DEFAULT_ROOTS = WS.DEFAULT_ROOTS
OUTPUT_ROOTS = WS.OUTPUT_ROOTS
EXPORT_ROOTS = WS.EXPORT_ROOTS


@dataclass(frozen=True)
class Rung:
    """One rung of the caption ladder: a caption file and what may be done to it.

    ``root`` names the :class:`Roots` field the file lives in. ``expand`` marks a
    rung that is a *sidecar* of many captions rather than one: ``"variants"`` holds
    the generated ``v0``/``v1``/``r1`` samples, ``"history"`` the versions a write
    superseded. Each is one file and one sidebar dot, expanded into a badge apiece
    by :func:`caption_versions`.
    """

    kind: str
    root: str
    editable: bool
    expand: str | None = None
    """``"variants"`` / ``"history"`` — the sidecar beside the caption, not the
    caption."""
    of: str = ""
    """Which rung a history sidecar records the past of; its badges wear that
    rung's name (``revised@2``)."""


CAPTION_LADDER: tuple[Rung, ...] = (
    Rung("master", "src", editable=True),
    Rung("history", "dst", editable=False, expand="history", of="revised"),
    Rung("revised", "dst", editable=True),
    Rung("variants", "dst", editable=False, expand="variants"),
)
"""The captions of one image, oldest first.

``history`` sits above ``revised`` because it holds the versions that caption used
to be. The run bar has no Apply gate, so the text a run replaces survives as a
badge here. The sidebar strip, the panel's badges and :func:`write_caption`'s guard
all read this tuple.
"""

CAPTION_KINDS = tuple(r.kind for r in CAPTION_LADDER if r.editable)
"""The writable rungs — what :func:`write_caption` accepts."""

_SIDECAR_PATH = {
    "variants": variants_sidecar_path,
    "history": history_sidecar_path,
}
"""``Rung.expand`` → the sidecar path beside a caption."""

HISTORY_OF = {r.of: r.kind for r in CAPTION_LADDER if r.expand == "history"}
"""Editable rung → the history rung recording it, for writers that push a version
before overwriting one. Empty for a rung the ladder gives no history."""


def ladder_schema() -> list[dict[str, Any]]:
    """:data:`CAPTION_LADDER` as the listing hands it to the browser.

    The sidebar's dot strip is drawn from this, so it cannot come apart from
    :func:`_row`'s ``captions`` map. The ``root`` stays server-side.
    """
    return [{"kind": r.kind, "editable": r.editable} for r in CAPTION_LADDER]


GROUPS_SUBPATH = "groups/groups.json"
"""The grouping manifest's tail under the Settings ``report_root``; the same split
``stages.report_subpath`` makes of ``build_groups``' own ``--out`` default."""

MAX_ITEMS = 20000
"""Hard cap on one listing, and the default. Past this the answer is
``path_pattern``, not a bigger payload."""


class DatasetError(ValueError):
    """Bad root / rel path — the server turns this into a 400 or 404."""


@dataclass(frozen=True)
class Roots:
    """The five dataset roots of one request, resolved and containment-checked.

    Field order is :data:`DEFAULT_ROOTS`' order, which :meth:`items` walks.
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

    ``normpath``, not ``resolve()``: a dataset root is routinely a symlink, and
    following it would report the tree under a name the user never typed and
    defeat the containment test below.
    """
    return Path(os.path.normpath(resolve_path(path)))


def dataset_bases() -> tuple[Path, ...]:
    """Every tree this panel may reach: the curation home, plus any dataset root
    the **saved** settings pin outside it.

    Saved only — a request's own root overrides never widen this, since a root
    outside the home widens what may be read. Only the Settings save does that
    (:func:`resolve_roots` with ``trusted``); every other path is checked here.
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
    """A path as the panel shows it: home-relative under the home, ``../``-style
    for the sibling tree beside it, absolute anywhere else.

    One level up only; deeper than that the absolute path says more. ``lexical``
    collapses the ``..`` again on the way back in, so a path that goes out this
    way comes back.
    """
    home = curation_home()
    if p.is_relative_to(home):
        return p.relative_to(home).as_posix()
    if p != home.parent and p.is_relative_to(home.parent):
        return "../" + p.relative_to(home.parent).as_posix()
    return p.as_posix()


def resolve_roots(
    values: dict[str, Any] | None = None, *, trusted: bool = False
) -> Roots:
    """Roots from ``values``, falling back to :data:`DEFAULT_ROOTS`. Blank
    strings fall back too, so an emptied field means "default".

    ``trusted`` is the Settings **save** and nothing else: that request is what
    *defines* :func:`dataset_bases`, so it cannot be checked against them. Every
    other caller is.
    """
    got = values or {}
    check = lexical if trusted else reachable
    paths = {}
    for name, default in DEFAULT_ROOTS.items():
        raw = got.get(name)
        paths[name] = check(str(raw).strip() if raw else default)
    return Roots(**paths)


def owned(p: Path) -> bool:
    """Is this a directory the panel may *create*? Only under its own home, so a
    typo in an external root is a missing root rather than a new empty tree."""
    return p.is_relative_to(curation_home())


def ensure_roots(roots: Roots) -> list[str]:
    """Create the root directories, returning the names actually made.

    Every root but :data:`EXPORT_ROOTS`' (an ``out`` tree that exists should mean
    an export happened), and only the ones this panel :func:`owned`. Called from
    an explicit write (the Settings save) only — never from :func:`resolve_roots`,
    so a read keeps reporting a missing root as missing.
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

    ``None`` when the path is not one the panel :func:`owned` or could not be
    created; the stage's own mkdir or failure covers that case.
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
    root — ``reachable`` only catches escapes after the join, and a root can sit
    deep enough that ``..`` stays inside the home.
    """
    p = Path(str(rel).replace("\\", "/"))
    if p.is_absolute() or any(part == ".." for part in p.parts) or not p.parts:
        raise DatasetError(f"bad relative path: {rel!r}")
    return p


def item_pattern(rel: str) -> str:
    """A ``path_pattern`` matching exactly the one dataset image ``rel``.

    ``<dir>/<stem>.*``, not the full filename, because the stage matches against
    the *resized* tree where the extension may differ (:func:`_sibling_image`);
    that cannot widen the match, since ``_walk.assert_unique_stems`` refuses two
    images sharing a stem in one folder.

    fnmatch metacharacters are escaped; ``|`` cannot be, since it separates the
    pattern's own alternatives, so such a name is refused outright.
    """
    p = _rel_key(rel)
    if "|" in p.as_posix():
        raise DatasetError(
            f"cannot scope a run to {rel!r}: '|' separates path_pattern "
            "alternatives, so no pattern can name this file"
        )
    return glob.escape(p.with_suffix("").as_posix()) + ".*"


def _sibling_image(directory: Path, stem: str) -> Path | None:
    """The image named ``stem`` in ``directory``, whatever its extension: the
    resize step may re-encode (``.jpg`` master → ``.png`` resized), so the
    revised tree is matched on stem, not on the full relative path."""
    for ext in IMAGE_EXTENSIONS:
        p = directory / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def rel_for_image(roots: Roots, image: str) -> str | None:
    """The dataset rel a stage report's ``image`` names, or ``None``.

    Reports name images relative to the **resized** tree, so the join back is on
    directory + stem (:func:`_sibling_image`). An image the source tree no longer
    has is dropped rather than raising.
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

    The name comes from ``masking._masks.mask_name``. The flat fallback is this
    reader's alone, so an older mask tree stays browsable.
    """
    name = mask_name(rel.stem)
    nested = roots.masks / rel.parent / name
    if nested.is_file():
        return nested
    flat = roots.masks / name
    return flat if flat.is_file() else None


def caption_paths(roots: Roots, rel: Path) -> dict[str, Path]:
    """Every rung's file, keyed by rung kind; :data:`CAPTION_LADDER` names the
    root each lives in."""
    txt = rel.with_suffix(".txt")
    out: dict[str, Path] = {}
    for r in CAPTION_LADDER:
        p = getattr(roots, r.root) / txt
        out[r.kind] = _SIDECAR_PATH[r.expand](p) if r.expand else p
    return out


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
        "ladder": ladder_schema(),
    }


def _row(roots: Roots, rel: Path, name: str) -> dict[str, Any]:
    """One sidebar row: the image plus which of its siblings exist.

    ``captions`` is one flag per :data:`CAPTION_LADDER` rung, one stat apiece so
    this stays cheap for a whole-dataset listing. ``resized`` is matched on *stem*
    (:func:`_sibling_image`) and is a row flag rather than a caption dot: it says
    whether the stages downstream of resize can see this image.
    """
    caps = caption_paths(roots, rel)
    parent = rel.parent.as_posix()
    return {
        "rel": rel.as_posix(),
        "dir": "" if parent == "." else parent,
        "name": name,
        "stem": rel.stem,
        "captions": {r.kind: caps[r.kind].is_file() for r in CAPTION_LADDER},
        "resized": _sibling_image(roots.dst / rel.parent, rel.stem) is not None,
        "mask": mask_path(roots, rel) is not None,
    }


def item_rows(roots: Roots, rels: list[str]) -> list[dict[str, Any]]:
    """:func:`list_items` rows for named images only, so a run that touched 40
    captions costs 40 stats rather than a walk of the source root. An unreadable
    or vanished rel is dropped."""
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

    Rels only: the group view joins them against the one ``/api/dataset`` listing
    the tree view already has. A missing manifest is not an error, but one built
    against another source tree joins onto nothing, so ``source_dir`` rides along.
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
        # A v1 manifest still lists usable components, so this is a "rebuild me"
        # note rather than a reason to show nothing.
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


def _image_info(p: Path | None, *, min_pixels: int = 0) -> dict[str, Any] | None:
    """One image file as the panel gets it: where it is, how big, how many pixels.

    ``min_pixels`` is the resize floor, applied to the *source* image only. An
    image under it never reaches ``workspace/resized/``, which every stage walks,
    so ``too_small`` is what keeps a run that sees zero images legible.
    """
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
    # ``None`` means *unmeasured* (no floor applied, or the header would not
    # read), not "fine"; the panel shows no chip for it.
    info["too_small"] = (
        below_min_pixels(size, min_pixels)
        if min_pixels > 0 and None not in size
        else None
    )
    return info


def parsed_caption(text: str) -> dict[str, Any]:
    """The caption grammar as JSON: flat bag + position clauses + where each tag
    sits in the string, never a ``split(",")`` on the client.

    ``spans`` lets the caption editor draw a box around every tag inside the
    textarea by slicing the text it holds at offsets the parse handed it.
    """
    parsed = parse_caption(text)
    return {
        "flat_tags": list(parsed.flat_tags),
        "spans": [
            {"start": s.start, "end": s.end, "kind": s.kind, "clause": s.clause}
            for s in tag_spans(text)
        ],
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


def _version_entry(
    kind: str,
    *,
    path: str,
    exists: bool,
    editable: bool,
    mtime: float | None,
    text: str,
    rung: str = "",
    note: str = "",
) -> dict[str, Any]:
    """One caption version as the panel gets it — the badge's wire shape.

    ``rung`` is the :data:`CAPTION_LADDER` row this entry came out of, which is not
    always its ``kind``: an expanded sidecar entry is called ``v1`` or ``revised@2``.
    It defaults to ``kind``. ``note`` is the one extra line a badge can carry (a
    history entry's who and when).

    ``exists`` is told, not read off ``text``: an empty caption file is a caption
    that says nothing, a different answer from a rung nobody ever wrote, and only
    the second draws hollow.
    """
    return {
        "kind": kind,
        "rung": rung or kind,
        "note": note,
        "path": path,
        "exists": exists,
        "editable": editable,
        "mtime": mtime,
        "text": text,
        "parsed": parsed_caption(text) if exists else None,
    }


def _caption_entry(
    kind: str, p: Path, *, editable: bool, rung: str = "", note: str = ""
) -> dict[str, Any]:
    """One caption *file* as a version: read it, or say it is not there.

    An absent file is still a (hollow) entry, so the rung keeps its place on the
    badge row. ``editable`` is the rung's property, not the file's.
    """
    exists = p.is_file()
    return _version_entry(
        kind,
        path=rel_to_home(p),
        exists=exists,
        editable=editable,
        mtime=p.stat().st_mtime if exists else None,
        text=p.read_text(encoding="utf-8").strip() if exists else "",
        rung=rung,
        note=note,
    )


def _expanded(rung: Rung, sidecar: Path) -> list[tuple[str, str, str]]:
    """A sidecar rung's captions as ``(label, note, text)``, oldest first.

    A variant's label *is* its name (``v0``); a history entry wears the rung's
    name plus its sequence, with a note saying who replaced it and when.
    """
    if rung.expand == "history":
        return [(e.label(rung.of), e.note(), e.text) for e in read_history(sidecar)]
    return [(label, "", text) for label, text in read_variants_sidecar(sidecar)]


def caption_versions(roots: Roots, rel: Path) -> list[dict[str, Any]]:
    """Every caption this image has, oldest first — the panel's badge row.

    :data:`CAPTION_LADDER` in order, with a sidecar rung *expanded* into one entry
    per caption it holds (``v0``, ``v1``… for variants, ``revised@1``, ``revised@2``…
    for superseded versions). An absent sidecar still contributes one hollow rung.
    Every entry arrives parsed, since the browser may not split a caption.
    """
    caps = caption_paths(roots, rel)
    out: list[dict[str, Any]] = []
    for r in CAPTION_LADDER:
        p = caps[r.kind]
        rows = _expanded(r, p) if r.expand and p.is_file() else []
        if not rows:
            out.append(_caption_entry(r.kind, p, editable=r.editable))
            continue
        # The sidecar is one file however many captions it holds: stat'd and
        # named once, with each row's text already in hand.
        mtime = p.stat().st_mtime
        home = rel_to_home(p)
        out.extend(
            _version_entry(
                label,
                path=home,
                exists=True,
                editable=False,
                mtime=mtime,
                text=text,
                rung=r.kind,
                note=note,
            )
            for label, note, text in rows
        )
    return out


def ocr_lines(roots: Roots, rel: Path) -> list[dict[str, Any]]:
    """The image's ``{stem}.ocr.txt`` from the OCR tree, or ``[]``.

    Not a :data:`CAPTION_LADDER` rung: it holds the words *in the picture*, not a
    text that could be written back into the caption. Joined by the same relative
    path as every other root. A missing sidecar means no text was found.
    """
    txt = rel.with_suffix(".txt")
    sidecar = ocr_sidecar_path(workspace_dir() / WS.OCR_SUBDIR / txt)
    return [line.to_dict() for line in read_ocr(sidecar)]


def item_detail(
    roots: Roots, rel_str: str, *, min_pixels: int = DEFAULT_MIN_PIXELS
) -> dict[str, Any]:
    """One image and everything hanging off it, for the item panel.

    ``min_pixels`` is the resize preflight's floor (the Settings *Preprocess*
    block), threaded in by the caller. It rides along in the answer as well as
    being applied, so the panel can say what the floor was.
    """
    rel = _rel_key(rel_str)
    src_image = roots.src / rel
    if not src_image.is_file():
        raise DatasetError(f"not in the dataset: {rel.as_posix()}")
    parent = rel.parent.as_posix()
    return {
        "rel": rel.as_posix(),
        "dir": "" if parent == "." else parent,
        "name": src_image.name,
        "stem": rel.stem,
        "min_pixels": int(min_pixels),
        "image": _image_info(src_image, min_pixels=min_pixels),
        "resized": _image_info(_sibling_image(roots.dst / rel.parent, rel.stem)),
        "mask": _image_info(mask_path(roots, rel)),
        "versions": caption_versions(roots, rel),
        "ocr": ocr_lines(roots, rel),
    }


def write_caption(roots: Roots, rel_str: str, kind: str, text: str) -> dict[str, Any]:
    """Write one caption file. The ladder's editable rungs only.

    A caption is a single line by contract, so newlines fold to spaces. An empty
    body is refused rather than treated as a delete. A hand edit pushes history
    like a stage run does, for the rungs :data:`HISTORY_OF` gives one.
    """
    if kind not in CAPTION_KINDS:
        raise DatasetError(f"not an editable caption: {kind!r}")
    rel = _rel_key(rel_str)
    if not (roots.src / rel).is_file():
        raise DatasetError(f"not in the dataset: {rel.as_posix()}")
    body = " ".join(str(text).split())
    if not body:
        raise DatasetError("refusing to write an empty caption")

    caps = caption_paths(roots, rel)
    p = caps[kind]
    if kind in HISTORY_OF and p.is_file():
        push_history(p, p.read_text(encoding="utf-8").strip(), by="edit")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")

    entry = _caption_entry(kind, p, editable=True)
    # The sidecar was generated from the previous revised text, so its v0 no
    # longer matches what the TE step would encode.
    entry["variants_stale"] = kind == "revised" and caps["variants"].is_file()
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
