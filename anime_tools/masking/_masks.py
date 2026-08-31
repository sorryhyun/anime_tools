"""What every mask generator does around the model: flags, plan, write, read.

Declaration *order* is part of the argparse contract (``gui.stages.fields_of``
walks ``parser._actions`` in order and the form follows it), which is why the
flag helpers come in small contiguous blocks the generators interleave their own
flags between rather than one all-or-nothing call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from anime_tools._walk import walk_images

MASK_SUFFIX = "_mask.png"
"""The whole contract between the generators, ``merge_masks`` and the GUI's mask
lookup is this string plus "mirror the source subdir"."""

WALK_HELP = (
    "Walk subfolders under --image-dir. Mask output mirrors the source "
    "subdir structure under --mask-dir."
)
PATTERN_HELP = (
    "fnmatch glob (| to OR-combine) on each image's path relative to "
    "--image-dir, restricting which images get masked. Same semantics "
    "as the training path_pattern."
)


def mask_name(stem: str) -> str:
    return f"{stem}{MASK_SUFFIX}"


# ---- argparse blocks ---------------------------------------------------


def add_mask_dir_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image-dir", type=str, required=True, help="Image directory")
    p.add_argument("--mask-dir", type=str, required=True, help="Output mask directory")


def add_force_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--force", action="store_true", help="Regenerate existing masks")


def add_device_arg(p: argparse.ArgumentParser) -> None:
    """``--device``, ``None`` so the CLI resolves it in-process: the dest is in
    :data:`anime_tools.gui.stages.AUTO_FIELDS`, never shown and never sent.
    """
    p.add_argument("--device", type=str, default=None, help="cuda|cpu (default: auto)")


def add_workers_arg(
    p: argparse.ArgumentParser, *, help: str = "I/O workers (default: 4)"
) -> None:
    p.add_argument("--workers", type=int, default=4, help=help)


def add_walk_args(p: argparse.ArgumentParser, *, pattern_help: str = "") -> None:
    """``--recursive`` / ``--path-pattern``: which images get masked at all.

    ``pattern_help`` appends a stage-specific sentence to :data:`PATTERN_HELP`.
    """
    p.add_argument("--recursive", action="store_true", help=WALK_HELP)
    p.add_argument(
        "--path-pattern",
        type=str,
        default=None,
        help=PATTERN_HELP + pattern_help,
    )


# ---- the write side ----------------------------------------------------


def mask_path_for(image_path: Path, image_dir: Path, mask_dir: Path) -> Path:
    """Where ``image_path``'s mask belongs: the source subdir, mirrored.

    An image outside ``image_dir`` (which ``walk_images`` will not hand back,
    but a direct caller might) lands flat at the root.
    """
    try:
        rel = image_path.parent.relative_to(image_dir)
    except ValueError:
        rel = Path("")
    target_dir = mask_dir if str(rel) in ("", ".") else mask_dir / rel
    return target_dir / mask_name(image_path.stem)


def plan_mask_jobs(
    image_dir: Path,
    mask_dir: Path,
    *,
    recursive: bool = False,
    pattern: str | None = None,
    force: bool = False,
) -> list[tuple[Path, Path]]:
    """``[(image_path, mask_path)]`` for the masks still to write.

    ``walk_images`` raises on same-stem collisions *within* one folder, which
    would have the two images overwrite each other's mask; the same stem across
    folders is fine, since the mirrored layout disambiguates it. Output
    directories are created here, so the caller's write loop is a plain save.
    """
    jobs: list[tuple[Path, Path]] = []
    for image_path in walk_images(image_dir, recursive=recursive, pattern=pattern):
        mask_path = mask_path_for(image_path, image_dir, mask_dir)
        if mask_path.exists() and not force:
            continue
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((image_path, mask_path))
    return jobs


def save_mask(path: Path, alpha_mask: np.ndarray) -> None:
    """Thread-safe enough for the I/O pool."""
    Image.fromarray(alpha_mask, mode="L").save(path)


def write_mask(path: Path, keep: np.ndarray, *, pool=None):
    """Save ``keep`` (1 = train here) as the 8-bit alpha mask ``keep * 255``.

    With ``pool``, the save is submitted to it and the future returned; without
    one it is written inline and ``None`` comes back.
    """
    alpha_mask = (np.asarray(keep) * 255).astype(np.uint8)
    if pool is None:
        save_mask(path, alpha_mask)
        return None
    return pool.submit(save_mask, path, alpha_mask)


def write_ignore_mask(path: Path, detected: np.ndarray, *, pool=None):
    """Save the *inverse* of a detection mask, the polarity the trainer reads.

    detected=1 → alpha=0 (ignored in the loss), no detection → alpha=255
    (trained on). The one home for that inversion.
    """
    return write_mask(path, 1 - np.asarray(detected), pool=pool)


# ---- the read side -----------------------------------------------------


def iter_masks(mask_dir: Path):
    """``(rel_dir, path)`` for every mask under ``mask_dir``, recursively.

    ``rel_dir`` is the mirrored source subdir as a string, ``""`` at the root —
    the key half of ``merge_masks``' ``(rel_dir, name)`` collision rule, so two
    inputs merge only when the mask sits at the same relative path in both.
    """
    for p in sorted(mask_dir.rglob(f"*{MASK_SUFFIX}")):
        rel = p.parent.relative_to(mask_dir)
        yield ("" if str(rel) in ("", ".") else str(rel)), p
