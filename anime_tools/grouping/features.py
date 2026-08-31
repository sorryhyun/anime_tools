"""PE-Spatial image-feature extraction + per-image cache (library primitive).

One PE-Spatial embedding path and one on-disk feature cache, shared by near-twin
mining, grouping and dedup. Each image is encoded at PE's native 512x512 bucket →
a global CLS descriptor + a 32x32 patch grid pooled to 16x16 (both L2-normed),
cached per-image as ``.npz`` under ``$NEAR_TWIN_CACHE`` (default
``~/.cache/near_twin/``), keyed by parent-dir hash + stem and stamped with the
source's ``(size, mtime_ns)`` + :data:`FEATURE_CACHE_VER` — see
:func:`_source_stamp` for why the stamp is what makes a rewritten tree a miss.

The encoder is passed in as an :class:`Embedder`; this module never owns the
model lifetime and never imports the PE loader.
``easycontrol_adapters.tools.near_twins.engine`` re-exports these names, so they
are not free to rename.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image

from anime_tools._walk import IMAGE_EXTENSIONS, glob_images_pathlib
from anime_tools.captions.position_clauses import parse_caption
from anime_tools.captions.taxonomy import normalize_tag

CACHE_ROOT = Path(
    os.environ.get("NEAR_TWIN_CACHE", Path.home() / ".cache" / "near_twin")
)
# Derived from the curation walker rather than retyped, so grouping sees the
# same image pool as the stages and the GUI browsing the same tree.
IMAGE_EXTS: tuple[str, ...] = tuple(sorted({e.lower() for e in IMAGE_EXTENSIONS}))
PE_NATIVE = 512  # PE-Spatial-B16-512 square bucket → 32x32 patch grid
GRID_NATIVE = 32
GRID_CACHE = 16  # cached pooled grid edge; any pooled grid <= 16 pools down from here
FEATURE_CACHE_VER = 1  # bump to invalidate every cached .npz when the schema changes


def read_tags(txt_path: Path) -> set[str]:
    """Read a ``.txt`` caption sidecar → set of normalized flat tags ("" → empty).

    Parsed through the caption grammar, never a hand ``split(",")``: the period
    is the clause delimiter, so a naive split would glue a clause header onto the
    previous tag. Clause tags are excluded — this is the flat bag.
    """
    if not txt_path.is_file():
        return set()
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    return {normalize_tag(t) for t in parse_caption(raw).flat_tags if t.strip()}


def caption_text(txt_path: Path) -> str:
    return (
        txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if txt_path.is_file()
        else ""
    )


@dataclass
class Member:
    artist: str
    stem: str
    image_path: Path
    txt_path: Path
    wh: tuple[int, int] = (0, 0)  # native pixel (W, H); (0, 0) = unreadable header


def _image_size(path: Path) -> tuple[int, int]:
    """Native ``(W, H)`` from the image header (no pixel decode); (0,0) on error."""
    try:
        with Image.open(path) as im:
            return im.size  # PIL returns (width, height)
    except Exception:  # noqa: BLE001 — corrupt/unreadable image
        return (0, 0)


def iter_images(root: Path) -> list[Path]:
    """Every image file under ``root`` (recursive), sorted — the shared curation
    glob, so grouping sees the pool the stages process and the GUI browses."""
    if not root.is_dir():
        return []
    return glob_images_pathlib(root, recursive=True)


def gather_members(
    image_dirs: list[Path], artists_filter: set[str] | None
) -> dict[str, list[Member]]:
    """Walk ``<dir>/<artist>/<stem>.<ext>`` trees → ``artist -> [Member]``.

    Scope is ``union`` across all ``image_dirs`` (a twin can straddle the curated
    cut). A ``(artist, stem)`` seen in more than one dir is kept once, first dir
    listed winning, so list the preferred source first.
    """
    seen: dict[tuple[str, str], Member] = {}
    for d in image_dirs:
        if not d.is_dir():
            print(f"  [warn] image dir not found: {d}", file=sys.stderr)
            continue
        for artist_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            artist = artist_dir.name
            if artists_filter and artist not in artists_filter:
                continue
            for img in glob_images_pathlib(artist_dir, recursive=False):
                key = (artist, img.stem)
                if key in seen:
                    continue
                seen[key] = Member(
                    artist, img.stem, img, img.with_suffix(".txt"), _image_size(img)
                )
    by_artist: dict[str, list[Member]] = {}
    for (artist, _), m in seen.items():
        by_artist.setdefault(artist, []).append(m)
    for members in by_artist.values():
        members.sort(key=lambda m: m.stem)
    return by_artist


def keep_size_cohabiting(members: list[Member]) -> list[Member]:
    """Drop members with no exact same-size sibling — they can never form a pair.

    The same-size gate's pre-embedding half: a unique canvas size within an
    artist has nothing to pair against, so embedding it would be wasted work.
    """
    sizes = Counter(m.wh for m in members)
    return [m for m in members if m.wh != (0, 0) and sizes[m.wh] >= 2]


def _dir_hash(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _cache_path(member: Member) -> Path:
    return CACHE_ROOT / _dir_hash(member.image_path.parent) / f"{member.stem}.npz"


def _source_stamp(path: Path) -> tuple[int, int]:
    """``(size, mtime_ns)`` of the source image; ``(-1, -1)`` if unreadable.

    The cache key addresses a *location* (parent-dir hash + stem), so nothing in
    it changes when the pixels underneath are rewritten — which is what a
    regenerated ``workspace/resized/`` does. The stamp is what makes that a miss
    instead of a silent stale hit. ``(-1, -1)`` never matches a real stat, so an
    unreadable source recomputes rather than trusting what was cached.
    """
    try:
        st = path.stat()
    except OSError:
        return (-1, -1)
    return (st.st_size, st.st_mtime_ns)


def _load_512(image_path: Path) -> torch.Tensor:
    """PIL → [3, 512, 512] in [-1, 1] (PE's Normalize(0.5, 0.5))."""
    with Image.open(image_path) as im:
        im = im.convert("RGB").resize((PE_NATIVE, PE_NATIVE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    return t * 2.0 - 1.0


_BAD_TENSOR = torch.zeros(3, PE_NATIVE, PE_NATIVE)  # placeholder for a failed decode


class _ImageDataset(torch.utils.data.Dataset):
    """Decode+resize on DataLoader workers so CPU preprocessing overlaps the GPU
    forward. A corrupt image yields ``ok=False`` (skipped downstream) instead of
    crashing the whole pass."""

    def __init__(self, members: list[Member]):
        self.members = members

    def __len__(self) -> int:
        return len(self.members)

    def __getitem__(self, i: int):
        try:
            return i, _load_512(self.members[i].image_path), True
        except Exception:  # noqa: BLE001 — corrupt/unreadable image
            return i, _BAD_TENSOR, False


def _collate(batch):
    idxs = [b[0] for b in batch]
    tens = torch.stack([b[1] for b in batch])
    oks = [b[2] for b in batch]
    return idxs, tens, oks


@runtime_checkable
class Embedder(Protocol):
    """Image-batch → (global descriptor, pooled patch grid) for the feature cache.

    ``batch`` arrives on ``self.device`` in ``self.dtype`` as ``[B, 3, 512, 512]``
    in ``[-1, 1]``; return ``(cls [B, D] L2-normed float32, grid16 [B, 16, 16, D]
    float16)`` as numpy. The cache format is embedder-agnostic in *name* only —
    switch embedders with a fresh ``$NEAR_TWIN_CACHE`` root.
    """

    device: torch.device
    dtype: torch.dtype

    def __call__(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class Feature:
    cls: np.ndarray  # [768] L2-normed float32
    grid16: np.ndarray  # [16, 16, 768] float16


def _save_feature(cache_path: Path, f: Feature, stamp: tuple[int, int]) -> None:
    """Write one cached feature, stamped with its source's ``(size, mtime_ns)``.

    ``stamp`` is captured before the decode rather than read here: a source
    rewritten mid-run then stores the *old* stamp against the new pixels, so the
    next run re-embeds. Stat-ing at save time would store the new stamp against
    the old pixels and the entry would never be revisited.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    size, mtime_ns = stamp
    np.savez(
        cache_path,
        cls=f.cls.astype(np.float32),
        grid16=f.grid16,
        size=size,
        mtime_ns=mtime_ns,
        ver=FEATURE_CACHE_VER,
    )


def _load_feature(cache_path: Path, stamp: tuple[int, int]) -> Feature | None:
    """Cached feature for ``stamp``, or ``None`` to recompute.

    Anything wrong — no file, an unreadable or truncated ``.npz``, a
    pre-stamp entry, a bumped :data:`FEATURE_CACHE_VER`, or a source whose size
    or mtime moved — means "recompute", never an error.
    """
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path) as z:
            if int(z["ver"]) != FEATURE_CACHE_VER:
                return None
            if (int(z["size"]), int(z["mtime_ns"])) != stamp:
                return None
            return Feature(cls=z["cls"].astype(np.float32), grid16=z["grid16"])
    except (OSError, ValueError, KeyError, EOFError):
        return None


def embed_members(
    embedder: Embedder,
    members: list[Member],
    batch_size: int,
    num_workers: int = 4,
    *,
    root: Path | None = None,
) -> dict[str, Feature]:
    """Load cached features; embed + cache any misses once via ``embedder``.

    Misses stream through a ``DataLoader`` with pinned-memory async H2D and a
    thread pool for the ``.npz`` writes, so decode, copy and forward overlap.

    With ``root`` given, the returned dict is keyed by each member's image path
    relative to it (POSIX string) — pass the tree's source dir whenever members
    can span subfolders, since nothing enforces unique stems tree-wide and two
    subfolders' ``1.webp`` would otherwise silently share one entry. Without
    ``root`` the key is the bare ``stem``, which is only safe when the caller's
    member scope guarantees unique stems. A member whose image fails to decode is
    omitted; the on-disk cache is keyed and stamped independently of this.
    """

    def _key(m: Member) -> str:
        if root is None:
            return m.stem
        return m.image_path.relative_to(root).as_posix()

    feats: dict[str, Feature] = {}
    todo: list[Member] = []
    stamps: dict[int, tuple[int, int]] = {}  # index into ``todo`` -> source stamp
    for m in members:
        stamp = _source_stamp(m.image_path)
        cached = _load_feature(_cache_path(m), stamp)
        if cached is not None:
            feats[_key(m)] = cached
        else:
            stamps[len(todo)] = stamp
            todo.append(m)
    if not todo:
        return feats

    pin = embedder.device.type == "cuda"
    loader = torch.utils.data.DataLoader(
        _ImageDataset(todo),
        batch_size=batch_size,
        num_workers=min(num_workers, len(todo)),
        pin_memory=pin,
        collate_fn=_collate,
        persistent_workers=False,
    )
    # tqdm to stderr so the daemon captures it and the GUI progress-bar tracker
    # (gui/progress.py TQDM_RE) can drive a determinate bar over the embed pass.
    from tqdm import tqdm

    pbar = tqdm(total=len(todo), desc="embedding", unit="img", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=2) as saver:
        for idxs, tens, oks in loader:
            batch = tens.to(embedder.device, embedder.dtype, non_blocking=pin)
            cls_b, grid_b = embedder(batch)
            for k, i in enumerate(idxs):
                if not oks[k]:
                    print(
                        f"  [warn] skipped unreadable {todo[i].image_path}",
                        file=sys.stderr,
                    )
                    continue
                f = Feature(cls=cls_b[k], grid16=grid_b[k])
                feats[_key(todo[i])] = f
                saver.submit(_save_feature, _cache_path(todo[i]), f, stamps[i])
            pbar.update(len(idxs))
    pbar.close()
    return feats
