"""The dataset roots of one request, and the containment rule every read obeys.

:class:`Roots` is the five trees :mod:`anime_tools.gui.dataset` joins by relative
path; :func:`reachable` / :func:`owned` are what the panel may read and create;
:func:`_rel_key` is the one gate a client-supplied ``rel`` passes before it is
joined to a root. The exclusion trees hang off the same roots. Re-exported by
:mod:`anime_tools.gui.dataset`, which is where callers reach them.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anime_tools import workspace as WS
from anime_tools._env import curation_home, resolve_path, workspace_dir
from anime_tools._walk import sibling_image
from anime_tools.exclude import Entry, ExclusionError, Trees, read_entries
from anime_tools.gui.settings import load_settings

SETTINGS_KEY = "dataset"
DEFAULT_ROOTS = WS.DEFAULT_ROOTS
OUTPUT_ROOTS = WS.OUTPUT_ROOTS
EXPORT_ROOTS = WS.EXPORT_ROOTS


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
    workspace: Path = field(default_factory=workspace_dir)
    """The workspace the trees above belong to — where the trees no root names
    live (the OCR sidecars, ``_excluded``). Not a root: nothing binds a flag to
    it, and the temp area (:func:`temp_roots`) is the one caller that moves it."""

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


def dataset_bases(settings: Mapping[str, Any] | None = None) -> tuple[Path, ...]:
    """Every tree this panel may reach: the curation home, plus any dataset root
    the **saved** settings pin outside it.

    Saved only — a request's own root overrides never widen this, since a root
    outside the home widens what may be read. Only the Settings save does that
    (:func:`resolve_roots` with ``trusted``); every other path is checked here.

    ``settings`` is the already-loaded blob, for a caller that has one: this is
    on the path of every root a request resolves, so reading the file here is
    one read per root on top of the caller's own.
    """
    home = curation_home()
    bases = [home]
    saved = (load_settings() if settings is None else settings).get(SETTINGS_KEY) or {}
    for name in DEFAULT_ROOTS:
        raw = str(saved.get(name) or "").strip()
        if not raw:
            continue
        p = lexical(raw)
        if not p.is_relative_to(home):
            bases.append(p)
    return tuple(bases)


def reachable(path: str | Path, bases: Sequence[Path] | None = None) -> Path:
    """``lexical`` + the containment rule every read enforces.

    ``..`` is collapsed *before* the test: ``is_relative_to`` is purely textual,
    so ``<home>/../elsewhere`` would otherwise sail through it.

    ``bases`` is :func:`dataset_bases` already computed, for a caller checking
    several paths against the same settings.
    """
    p = lexical(path)
    bases = dataset_bases() if bases is None else bases
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
    values: dict[str, Any] | None = None,
    *,
    trusted: bool = False,
    bases: Sequence[Path] | None = None,
) -> Roots:
    """Roots from ``values``, falling back to :data:`DEFAULT_ROOTS`. Blank
    strings fall back too, so an emptied field means "default".

    ``trusted`` is the Settings **save** and nothing else: that request is what
    *defines* :func:`dataset_bases`, so it cannot be checked against them. Every
    other caller is.

    ``bases`` is those bases already computed. Without it each of the four roots
    recomputes them, and each of those is another read of the settings file.
    """
    got = values or {}
    if not trusted and bases is None:
        bases = dataset_bases()
    check = lexical if trusted else (lambda raw: reachable(raw, bases))
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
    the *resized* tree where the extension may differ (:func:`sibling_image`);
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


def rel_for_image(roots: Roots, image: str) -> str | None:
    """The dataset rel a stage report's ``image`` names, or ``None``.

    Reports name images relative to the **resized** tree, so the join back is on
    directory + stem (:func:`sibling_image`). An image the source tree no longer
    has is dropped rather than raising.
    """
    try:
        rel = _rel_key(image)
    except DatasetError:
        return None
    if (roots.src / rel).is_file():
        return rel.as_posix()
    p = sibling_image(roots.src / rel.parent, rel.stem)
    return p.relative_to(roots.src).as_posix() if p else None


def excluded_trees(roots: Roots) -> Trees:
    """The trees an exclusion moves between, for this request's roots.

    ``resized`` and ``masks`` follow the dataset roots, so a relocated root is
    honoured; the OCR sidecars and ``_excluded`` itself are workspace trees no
    root names, resolved the way :func:`ocr_path` resolves the first of them.
    """
    ws = roots.workspace
    return Trees(
        resized=roots.dst,
        masks=roots.masks,
        ocr=ws / WS.OCR_SUBDIR,
        excluded=ws / WS.EXCLUDED_SUBDIR,
    )


def excluded_entries(roots: Roots) -> dict[str, Entry]:
    """The exclusion ledger, as a 400 rather than a 500 when it will not parse.

    Read once per listing: it is one small JSON file, and a stat per row would be
    the wrong shape for a flag that is a set membership.
    """
    try:
        return read_entries(excluded_trees(roots).excluded)
    except ExclusionError as e:
        raise DatasetError(str(e)) from e
