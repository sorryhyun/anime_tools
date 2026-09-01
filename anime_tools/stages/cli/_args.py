"""Shared argparse blocks for the stage CLIs.

:mod:`anime_tools.gui.stages` introspects ``build_parser()`` to build the GUI
form, so a dropped ``dest=``, a drifted default or a lost ``--foo-bar`` alias
silently changes the form rather than failing anything. Declaration *order* is
part of that contract (``fields_of`` walks ``parser._actions`` in order), so
each helper adds its flags as one contiguous block.
"""

from __future__ import annotations

import argparse

from anime_tools import workspace as WS
from anime_tools._device import add_device_arg
from anime_tools.tagger.dbv4_meta import DEFAULT_TAGGER_DIR

PATTERN_HELP = "fnmatch glob (| to OR-combine) on the path relative to {root}"


def add_path_pattern_arg(p: argparse.ArgumentParser, *, help: str) -> None:
    """``--path_pattern`` — the one scope knob the GUI narrows to a single image.

    ``gui.stages.SCOPE_FIELD`` looks for this ``dest``, so the dual spelling and
    the ``dest=`` are load-bearing.
    """
    p.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help=help,
    )


def add_dataset_args(
    p: argparse.ArgumentParser,
    *,
    src_help: str = "Caption master dir",
    dst_help: str = "Resized images",
    pattern_root: str = "--dst",
) -> None:
    """``--src`` / ``--dst`` / ``--path_pattern``: the three dataset roots.

    ``pattern_root`` names the tree the glob is matched against — ``--dst`` for
    every stage that walks the resized tree, ``--src`` for resize.
    """
    p.add_argument("--src", default="image_dataset", help=src_help)
    p.add_argument("--dst", default=WS.RESIZED, help=dst_help)
    add_path_pattern_arg(p, help=PATTERN_HELP.format(root=pattern_root))


def add_apply_args(
    p: argparse.ArgumentParser, *, apply_help: str, from_report_help: str
) -> None:
    """``--apply`` / ``--from_report``: dry-run-by-default and its replay."""
    p.add_argument("--apply", action="store_true", help=apply_help)
    p.add_argument(
        "--from_report",
        "--from-report",
        dest="from_report",
        default=None,
        help=from_report_help,
    )


def add_report_dir_arg(p: argparse.ArgumentParser, default: str) -> None:
    """``--report_dir`` — where ``report.json`` lands."""
    p.add_argument(
        "--report_dir",
        "--report-dir",
        dest="report_dir",
        default=default,
        help=f"Where report.json lands (default: {default})",
    )


def add_model_args(p: argparse.ArgumentParser) -> None:
    """``--tagger_dir`` / ``--device``.

    ``--device`` defaults to ``None`` on purpose: it is in
    ``gui.stages.AUTO_FIELDS`` (never shown, never sent) and each stage resolves
    it at the model-load site, so the torch-free ``--from_report`` replay path
    never pays for the answer.
    """
    p.add_argument(
        "--tagger_dir",
        "--tagger-dir",
        dest="tagger_dir",
        default=None,
        help=f"Anima Tagger checkpoint dir (default: {DEFAULT_TAGGER_DIR})",
    )
    add_device_arg(p)


def make_progress(every: int, *, first: bool = False):
    """A ``progress(index, total, detail)`` that prints one line every ``every``.

    The last line always prints, so a run under ``every`` images still says it
    finished; ``first`` also prints image 1.
    """

    def progress(index: int, total: int, detail: str) -> None:
        if index == total or index % every == 0 or (first and index == 1):
            print(f"  [{index}/{total}] {detail}", flush=True)

    return progress
