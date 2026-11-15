"""Where a mask is, as paths alone — stdlib only.

Split out of :mod:`._masks` (numpy + PIL at import) for the readers that only
need to *find* a mask: the GUI's dataset listing, the exclusion move, Export's
plan and the OCR stage's component pass. :mod:`._masks` re-exports every name.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["MASK_SUFFIX", "find_mask", "mask_candidates", "mask_name", "mask_path_for"]

MASK_SUFFIX = "_mask.png"
"""The contract between the generators, ``merge_masks`` and the GUI's mask lookup:
this suffix plus "mirror the source subdir"."""


def mask_name(stem: str) -> str:
    return f"{stem}{MASK_SUFFIX}"


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


def mask_candidates(root: Path, rel: Path) -> tuple[Path, Path]:
    """The two places ``rel``'s mask may be under ``root``, in the order every
    reader tries them: the mirrored layout (``<rel dir>/{stem}_mask.png``), then
    the legacy flat one (``{stem}_mask.png`` at the root). One path twice for an
    image at the root of the tree.

    ``rel`` is the image's path relative to its tree; only its directory and
    stem are read, so the image's own suffix does not matter.
    """
    name = mask_name(rel.stem)
    return root / rel.parent / name, root / name


def find_mask(root: Path, rel: Path) -> Path | None:
    """The first of :func:`mask_candidates` that is a file, or ``None``."""
    return next((p for p in mask_candidates(root, rel) if p.is_file()), None)
