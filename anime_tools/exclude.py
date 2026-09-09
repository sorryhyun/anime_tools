"""Taking an image out of the pipeline, and putting it back.

An exclusion is a curation decision, not a stage: no model, no report, no diff to
agree to. It moves every file the image has out of the live workspace trees into
``workspace/_excluded/`` and writes down where each came from
(:data:`~anime_tools.workspace.EXCLUDED`); un-excluding moves them back.

**The ledger is the state.** Every stage but ``resize`` walks
``workspace/resized/``, so an image whose resized copy has left that tree is
already invisible to all of them — but ``resize`` would put it straight back on
the next preflight, which is why it reads
:func:`excluded_rels` and adds them to its own ``--skip``. That one chokepoint is
the whole enforcement; nothing else has to know.

The ledger is keyed by the image's path relative to the **source** tree
(``char_aki/a.jpg``), which is what ``resize``'s ``--skip`` names and what a
sidebar row is. The files it moves are keyed by directory + stem instead, because
resize re-encodes (``a.jpg`` → ``a.png``) and each tree spells the tail its own
way (``a.txt``, ``a_mask.png``, ``a.ocr.txt``).

Torch-free.
"""

from __future__ import annotations

import argparse
import shutil
import stat
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from anime_tools import workspace as WS
from anime_tools._env import resolve_path
from anime_tools._json import read_json, write_json
from anime_tools._walk import IMAGE_EXTENSIONS
from anime_tools.captions.history import history_sidecar_path
from anime_tools.captions.ocr_sidecar import ocr_sidecar_path
from anime_tools.captions.variants import variants_sidecar_path
from anime_tools.masking._masks import mask_name
from anime_tools.stages.resize import CAPTION_EXTENSIONS

__all__ = [
    "Entry",
    "ExclusionError",
    "Result",
    "Trees",
    "artifacts",
    "exclude_one",
    "excluded_rels",
    "is_excluded",
    "manifest_path",
    "read_entries",
    "restore_one",
    "write_entries",
]

MANIFEST_VERSION = 1
"""Bumped when the ledger's shape changes. An older file is read as it is —
there is no v0, and a reader that could not parse one refuses rather than
guessing (:class:`ExclusionError`)."""


class ExclusionError(ValueError):
    """A bad rel, or a ledger that will not parse.

    Unparseable is an error and never an empty ledger: "nothing is excluded" is
    the answer that would put every excluded image back through the pipeline on
    the next preflight.
    """


@dataclass(frozen=True)
class Entry:
    """One excluded image: what it was, when, why, and what moved with it."""

    rel: str
    """The image's path relative to the source tree — the ``resize --skip``
    spelling, forward slashes, extension kept."""
    at: float
    """Unix time of the *first* exclusion. Re-excluding an image already in the
    ledger keeps this, since that is when it left the dataset."""
    note: str = ""
    """Free text from the curator; the GUI shows it on the item panel."""
    moved: tuple[str, ...] = ()
    """Every file that moved, as a slot: ``<tree>/<path under that tree>``, with
    ``<tree>`` one of :data:`~anime_tools.workspace.EXCLUDED_TREES`. The slot is
    both where the file sits under ``_excluded`` and how to put it back."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel": self.rel,
            "at": self.at,
            "note": self.note,
            "moved": list(self.moved),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Entry | None:
        """One ledger row, or ``None`` for a row with no usable ``rel``."""
        if not isinstance(raw, dict):
            return None
        rel = str(raw.get("rel") or "").strip()
        if not rel:
            return None
        try:
            at = float(raw.get("at") or 0.0)
        except (TypeError, ValueError):
            at = 0.0
        moved = raw.get("moved")
        return cls(
            rel=rel,
            at=at,
            note=str(raw.get("note") or ""),
            moved=tuple(str(m) for m in moved) if isinstance(moved, list) else (),
        )


@dataclass
class Result:
    """What one exclude or restore actually did — the CLI's line and the GUI's
    answer, so neither has to re-derive it from the ledger."""

    rel: str
    action: str
    """``excluded`` / ``restored`` / ``already-excluded`` / ``not-excluded``."""
    moved: tuple[str, ...] = ()
    """Slots that changed tree."""
    skipped: tuple[str, ...] = ()
    """Slots left where they were, because the live tree already holds that path
    (a resize that ran without the ledger, say). Never overwritten — the copy in
    ``_excluded`` is the older one, and clobbering is not a curation decision."""
    entry: Entry | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel": self.rel,
            "action": self.action,
            "moved": list(self.moved),
            "skipped": list(self.skipped),
            "entry": self.entry.to_dict() if self.entry else None,
        }


@dataclass(frozen=True)
class Trees:
    """The three live trees an exclusion empties, and the tree it fills.

    Field names are :data:`~anime_tools.workspace.EXCLUDED_TREES` plus
    ``excluded``, so a slot's first component names the attribute to put the file
    back into and the mapping is never written down twice.
    """

    resized: Path
    masks: Path
    ocr: Path
    excluded: Path

    @classmethod
    def build(
        cls,
        *,
        dst: str | Path = WS.RESIZED,
        masks: str | Path = WS.MASKS,
        ocr: str | Path = WS.OCR,
        excluded: str | Path = WS.EXCLUDED,
    ) -> Trees:
        """The workspace defaults, each anchored under the curation home."""
        return cls(
            resized=resolve_path(dst),
            masks=resolve_path(masks),
            ocr=resolve_path(ocr),
            excluded=resolve_path(excluded),
        )

    def live(self, tree: str) -> Path:
        """The live tree a slot's first component names."""
        if tree not in WS.EXCLUDED_TREES:
            raise ExclusionError(f"not an excluded tree: {tree!r}")
        return getattr(self, tree)


# ---- the ledger ---------------------------------------------------------


def manifest_path(excluded: str | Path) -> Path:
    return Path(excluded) / WS.EXCLUDED_MANIFEST


def read_entries(excluded: str | Path) -> dict[str, Entry]:
    """The ledger, keyed by rel. A ledger that is not there is an empty one; a
    ledger that will not parse is an error."""
    path = manifest_path(excluded)
    if not path.is_file():
        return {}
    try:
        data = read_json(path)
    except (OSError, ValueError) as e:
        raise ExclusionError(f"unreadable exclusion ledger {path}: {e}") from e
    if not isinstance(data, dict):
        raise ExclusionError(f"not an exclusion ledger: {path}")
    rows = data.get("excluded")
    entries = [Entry.from_dict(r) for r in rows] if isinstance(rows, list) else []
    return {e.rel: e for e in entries if e is not None}


def write_entries(excluded: str | Path, entries: dict[str, Entry]) -> Path:
    """Write the ledger, rels sorted so a diff of it reads as a diff.

    The file stays even when it is empty: an ``_excluded`` directory with no
    ledger in it would read as a tree nobody wrote.
    """
    return write_json(
        manifest_path(excluded),
        {
            "version": MANIFEST_VERSION,
            "excluded": [entries[rel].to_dict() for rel in sorted(entries)],
        },
    )


def excluded_rels(excluded: str | Path) -> tuple[str, ...]:
    """Every excluded rel, sorted — the ``resize --skip`` list."""
    return tuple(sorted(read_entries(excluded)))


def is_excluded(excluded: str | Path, rel: str) -> bool:
    return rel_key(rel) in read_entries(excluded)


def rel_key(rel: str | Path) -> str:
    """A caller's rel as the ledger spells it: forward slashes, relative, no
    ``..``. Refused rather than normalised, since a rel is a dataset path and a
    surprising one would move the wrong file."""
    s = str(rel).replace("\\", "/")
    win, posix = PureWindowsPath(s), PurePosixPath(s)
    # The anchor test is Windows' on every host, because it is the strict one:
    # `Path("/abs").is_absolute()` is False on Windows (no drive), so a rooted
    # rel slips past `is_absolute` there and `tree / rel` then lands on the
    # drive root -- outside the dataset. Windows rules refuse a drive ("C:/x"),
    # a UNC share ("//srv/x") and a bare leading slash alike.
    if (
        win.drive
        or win.root
        or not posix.parts
        or any(part in ("..", ".") for part in posix.parts)
    ):
        raise ExclusionError(f"bad relative path: {rel!r}")
    return posix.as_posix()


# ---- what an image is made of -------------------------------------------


def _resized_files(root: Path, rel: Path) -> Iterator[Path]:
    """The resized image and every sidecar keyed to its stem.

    Matched on directory + stem, not on ``rel`` itself: resize re-encodes to PNG,
    so the master's ``a.jpg`` is ``a.png`` here. Spelled out rather than globbed
    ``a.*``, which would also sweep up the neighbouring image ``a.b.png``.
    """
    directory = root / rel.parent
    caption = directory / f"{rel.stem}.txt"
    for ext in (*IMAGE_EXTENSIONS, *CAPTION_EXTENSIONS):
        yield directory / f"{rel.stem}{ext}"
    yield variants_sidecar_path(caption)
    yield history_sidecar_path(caption)


def _mask_files(root: Path, rel: Path) -> Iterator[Path]:
    """The mirrored mask, and the legacy flat one — the same two-step lookup
    ``gui.dataset.mask_path`` and Export's ``_mask_source`` do."""
    yield root / rel.parent / mask_name(rel.stem)
    yield root / mask_name(rel.stem)


def _ocr_files(root: Path, rel: Path) -> Iterator[Path]:
    yield ocr_sidecar_path(root / rel.parent / f"{rel.stem}.txt")


_FILES = {"resized": _resized_files, "masks": _mask_files, "ocr": _ocr_files}


def _slot(trees: Trees, tree: str, path: Path) -> str:
    return f"{tree}/{path.relative_to(trees.live(tree)).as_posix()}"


def artifacts(trees: Trees, rel: str) -> list[tuple[str, Path]]:
    """``[(slot, live path)]`` for every file this image has *right now*.

    Deduplicated by what the path resolves *to*, not by how it is spelled: the
    mask lookup offers a mirrored and a flat path, which are one file for an
    image at the root of the tree, and on a case-insensitive filesystem
    ``a.png`` and ``a.PNG`` — both in ``IMAGE_EXTENSIONS`` — are one file under
    two names. The first spelling wins, and the candidates are ordered so that is
    the lowercase one.
    """
    key = Path(rel_key(rel))
    out: list[tuple[str, Path]] = []
    seen: set[tuple[int, int]] = set()
    for tree in WS.EXCLUDED_TREES:
        for path in _FILES[tree](trees.live(tree), key):
            try:
                st = path.stat()
            except OSError:  # not there, or not reachable — the same answer here
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            ident = (st.st_dev, st.st_ino)
            if ident in seen:
                continue
            seen.add(ident)
            out.append((_slot(trees, tree, path), path))
    return out


# ---- moving ------------------------------------------------------------


def _move(src: Path, dst: Path) -> None:
    """``shutil.move``, not ``rename``: the trees are normally one filesystem,
    but a ``dst`` root on another mount must not make this a crash."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _prune(directory: Path, stop: Path) -> None:
    """Remove ``directory`` and its now-empty parents, up to but never including
    ``stop``. What keeps a restored tree from leaving a skeleton of empty folders
    behind under ``_excluded``."""
    while directory != stop and directory.is_relative_to(stop):
        try:
            directory.rmdir()
        except OSError:  # not empty, or gone already — and that ends the walk
            return
        directory = directory.parent


def exclude_one(
    trees: Trees, rel: str, *, note: str = "", apply: bool = True
) -> Result:
    """Take one image out of the pipeline.

    Every file it has moves under ``_excluded`` and the rel goes into the ledger.
    An image with nothing in the workspace yet — excluded before it was ever
    resized — is still recorded: the ledger is the state, and the point is that
    ``resize`` never makes those files in the first place.

    Re-excluding keeps the original ``at`` (that is when it left the dataset) and
    unions what moved, so a file that appeared since is swept up too.
    """
    key = rel_key(rel)
    entries = read_entries(trees.excluded)
    was = entries.get(key)

    moved: list[str] = []
    for slot, live in artifacts(trees, key):
        if apply:
            if not live.is_file():  # vanished since the walk above
                continue
            # The destination is inside ``_excluded``, which this module owns, so
            # a leftover there is ours to replace.
            _move(live, trees.excluded / slot)
        moved.append(slot)

    entry = Entry(
        rel=key,
        at=was.at if was else time.time(),
        note=note or (was.note if was else ""),
        moved=tuple(dict.fromkeys((*(was.moved if was else ()), *moved))),
    )
    if apply:
        entries[key] = entry
        write_entries(trees.excluded, entries)
    return Result(
        rel=key,
        action="already-excluded" if was and not moved else "excluded",
        moved=tuple(moved),
        entry=entry,
    )


def restore_one(trees: Trees, rel: str, *, apply: bool = True) -> Result:
    """Put one image back, and take its rel out of the ledger.

    A slot whose live path is occupied again is **left** under ``_excluded`` and
    reported: the file there is the older one, and choosing between them is a
    curation decision this is not. The rel leaves the ledger either way — the
    image is no longer excluded, whatever is on disk.
    """
    key = rel_key(rel)
    entries = read_entries(trees.excluded)
    was = entries.get(key)
    if was is None:
        return Result(rel=key, action="not-excluded")

    moved: list[str] = []
    skipped: list[str] = []
    for slot in was.moved:
        tree, _, tail = slot.partition("/")
        if not tail or tree not in WS.EXCLUDED_TREES:
            continue
        src = trees.excluded / slot
        dst = trees.live(tree) / tail
        if not src.is_file():
            continue
        if dst.exists():
            skipped.append(slot)
            continue
        if apply:
            _move(src, dst)
            _prune(src.parent, trees.excluded)
        moved.append(slot)

    if apply:
        del entries[key]
        write_entries(trees.excluded, entries)
    return Result(
        rel=key,
        action="restored",
        moved=tuple(moved),
        skipped=tuple(skipped),
        entry=was,
    )


# ---- the CLI -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m anime_tools.exclude",
        description="Take images out of the pipeline, or put them back. An "
        "excluded image's files move into workspace/_excluded/ and its path goes "
        "into the ledger there, which `resize` reads as extra --skip entries — so "
        "no later stage ever sees it again, and Export republishes it under "
        "<out>/_excluded/ rather than into the tree the trainer reads. Dry run by "
        "default.",
    )
    p.add_argument(
        "rels",
        nargs="*",
        metavar="REL",
        help="Image paths relative to --src, forward slashes (char_aki/a.jpg)",
    )
    p.add_argument(
        "--restore",
        action="store_true",
        help="Put these images back instead: every file returns to the tree it "
        "came from and the rel leaves the ledger",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the ledger — one rel per line, with what moved — and exit",
    )
    p.add_argument("--note", default="", help="Why, recorded on the ledger row")
    p.add_argument(
        "--src",
        default=WS.SOURCE_ROOT,
        help=f"Caption master dir the rels are relative to (default: "
        f"{WS.SOURCE_ROOT}). An excluded rel must name a file under it; "
        "--restore does not check, since the source may be gone by then.",
    )
    p.add_argument("--dst", default=WS.RESIZED, help=f"Resized tree ({WS.RESIZED})")
    p.add_argument("--masks", default=WS.MASKS, help=f"Mask tree ({WS.MASKS})")
    p.add_argument("--ocr_dir", default=WS.OCR, help=f"OCR sidecar tree ({WS.OCR})")
    p.add_argument(
        "--excluded_dir",
        default=WS.EXCLUDED,
        help=f"The excluded tree and its ledger (default: {WS.EXCLUDED})",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Move for real (default: say what would move and stop)",
    )
    return p


def _print_ledger(trees: Trees) -> None:
    entries = read_entries(trees.excluded)
    if not entries:
        print(f"nothing excluded ({manifest_path(trees.excluded)})")
        return
    for rel in sorted(entries):
        e = entries[rel]
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.at)) if e.at else "?"
        print(
            f"{rel}  [{when}] {len(e.moved)} file(s){'  — ' + e.note if e.note else ''}"
        )
        for slot in e.moved:
            print(f"    {slot}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trees = Trees.build(
        dst=args.dst, masks=args.masks, ocr=args.ocr_dir, excluded=args.excluded_dir
    )

    if args.list:
        _print_ledger(trees)
        return 0
    if not args.rels:
        print("no images named; pass a REL or --list", file=sys.stderr)
        return 2

    src = resolve_path(args.src)
    results: list[Result] = []
    for raw in args.rels:
        try:
            key = rel_key(raw)
            if not args.restore and not (src / key).is_file():
                raise ExclusionError(f"not in the dataset: {key} (under {src})")
            results.append(
                restore_one(trees, key, apply=args.apply)
                if args.restore
                else exclude_one(trees, key, note=args.note, apply=args.apply)
            )
        except ExclusionError as e:
            print(f"{raw}: {e}", file=sys.stderr)
            return 2

    verb = "would move" if not args.apply else "moved"
    for r in results:
        print(f"{r.rel}: {r.action}, {verb} {len(r.moved)} file(s)")
        for slot in r.moved:
            print(f"    {slot}")
        for slot in r.skipped:
            print(f"    kept in _excluded (live path occupied): {slot}")
    if not args.apply:
        print("\nDry run — nothing moved. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
