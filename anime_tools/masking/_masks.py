"""What every mask generator does around the model: flags, plan, run, write, read.

Declaration *order* is part of the argparse contract (``gui.stages.fields_of``
walks ``parser._actions`` in order and the form follows it), which is why the
flag helpers come in small contiguous blocks the generators interleave their own
flags between rather than one all-or-nothing call.

The flags declared here are also read back here, in :func:`mask_run`: the walk,
the output tree and the I/O pool are this module's own vocabulary, so a
generator that spells one of them differently from the way it was declared is a
mistake that cannot be made in the first place.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from anime_tools._env import resolve_path
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


def add_mask_dir_args(p: argparse.ArgumentParser, *, mask_default: str) -> None:
    """``--image-dir`` / ``--mask-dir``, the two ends of a generator's walk.

    ``mask_default`` is this generator's own tree
    (:data:`anime_tools.workspace.MASKS_SAM` / ``MASKS_MIT``) and is required
    of the caller, because the one thing a mask directory must not be is
    *shared*: two generators writing ``{stem}_mask.png`` at the same relative
    path would overwrite each other, and ``merge_masks`` — which unions them —
    is what fills the ``masks`` root. Defaulted rather than bound to a dataset
    root for the same reason, so the GUI form ships filled in and the operator
    still owns the value.
    """
    p.add_argument("--image-dir", type=str, required=True, help="Image directory")
    p.add_argument(
        "--mask-dir",
        type=str,
        default=mask_default,
        help=f"Output mask directory for this generator alone (default: "
        f"{mask_default}); `merge_masks` unions it with the other's into the "
        f"masks root",
    )


GATE_ATTR = "gui_gate"
"""The attribute :func:`gated_group` stamps a group with, naming the dest that
switches it on. Read back by ``anime_tools.gui.stages.fields_of``, which spells
the string rather than importing it: that module stays free of every stage's
dependencies, exactly as it duplicates ``REPLAY_REPORT_NAME``. The pairing is
pinned by ``tests/test_masking_plan.py``."""


def gated_group(
    p: argparse.ArgumentParser,
    title: str,
    *,
    gate: str,
    default: bool,
    help: str,
) -> argparse._ArgumentGroup:
    """An argument group behind an on/off flag — a *drawer* in the GUI form.

    Returns the group, with the gate declared as its first argument, so the
    flags that only matter while it is on are written inside it and the browser
    can fold them away when it is off. Two detectors in one stage is what this
    is for: the knobs of the one you are not running are noise on the form, and
    :func:`anime_tools.gui.stages.build_argv` drops a shut drawer's values from
    the argv rather than passing flags the stage would ignore.

    ``gate`` is the dest (``use_sam`` → ``--use-sam`` / ``--no-use-sam``), and
    it is what the drawer's checkbox is: it stays a plain flag, so the CLI is
    the same shape with or without a browser in front of it.
    """
    g = p.add_argument_group(title)
    g.add_argument(
        f"--{gate.replace('_', '-')}",
        dest=gate,
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help,
    )
    setattr(g, GATE_ATTR, gate)
    return g


def add_force_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--force", action="store_true", help="Regenerate existing masks")


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


# ---- the run -----------------------------------------------------------


def coverage_pct(mask: np.ndarray) -> float:
    """What share of the frame ``mask`` covers, as a percentage.

    Denominated in the mask's *own* size rather than a ``(w, h)`` passed
    alongside it: every mask here is the frame, and the one way this arithmetic
    goes wrong is a caller handing it the transposed pair. Deliberately
    polarity-blind — the generators call it on a keep mask and on an ignore
    mask, and what the number is *called* on the progress line is theirs to say.
    """
    return 100 * np.count_nonzero(mask) / mask.size


@dataclass(slots=True)
class MaskRun:
    """One generator's pass: where it reads, where it writes, what is left to do.

    ``items`` is the plan (see :func:`plan_mask_jobs`) — a list rather than an
    iterator because a batching generator indexes ahead of the loop to prefetch.
    ``pool`` is the shared I/O pool ``write_mask`` submits saves to; the progress
    bar is private, reached through the two methods below, so ``tqdm`` stays out
    of the stages entirely.
    """

    image_dir: Path
    mask_dir: Path
    items: list[tuple[Path, Path]]
    pool: ThreadPoolExecutor
    _bar: tqdm

    @property
    def total(self) -> int:
        return len(self.items)

    def advance(self) -> None:
        """One image dealt with — called before the branches, so an image that
        gets no mask still moves the bar."""
        self._bar.update(1)

    def note(self, image_path: Path, what: str) -> None:
        """``name: what`` beside the bar. ``what`` is the stage's own wording
        (``train 41.2%``, ``skipped (ctd-gated)``, ``focus not found``); only the
        shape of the line is shared."""
        self._bar.set_postfix_str(f"{image_path.name}: {what}")


@contextmanager
def mask_run(
    args: argparse.Namespace, *, desc: str = "Generating masks"
) -> Iterator[MaskRun]:
    """The scaffolding both generators wrap their inner loop in.

    Owns the four things that are the same whatever the detector is: the two
    roots are home-anchored (the ``--mask-dir`` defaults above are written
    home-relative, so a run from another directory has to mean the same tree the
    GUI and the merge do), the output root exists before the first write, the
    plan is drawn from the walk flags this module declared, and the bar and the
    pool are closed in that order — the bar first, so its line is finished
    before the closing sentence prints over it.

    Draining those saves is the caller's, not this function's: the batching
    generator holds futures it has to see the results of, and a future whose
    exception nobody reads is a mask that silently did not get written. Do it
    inside the ``with``.

    Nothing to do is not an error and not an early return: ``items`` is empty,
    the loop the caller writes does not run, the bar never draws (it is
    constructed disabled) and the closing line says so instead of naming a
    directory nothing landed in.
    """
    image_dir = resolve_path(args.image_dir)
    mask_dir = resolve_path(args.mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    items = plan_mask_jobs(
        image_dir,
        mask_dir,
        recursive=args.recursive,
        pattern=args.path_pattern,
        force=args.force,
    )

    pool = ThreadPoolExecutor(max_workers=args.workers)
    bar = tqdm(total=len(items), desc=desc, disable=not items)
    try:
        yield MaskRun(image_dir, mask_dir, items, pool, bar)
    finally:
        bar.close()
        pool.shutdown()
    # Past the `finally`, so a run that raised does not sign off as if it had not.
    print(f"Masks saved to {mask_dir}/" if items else "No images to process.")


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
