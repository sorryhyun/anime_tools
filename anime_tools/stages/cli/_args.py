"""Shared argparse blocks for the stage CLIs.

Every caption stage takes the same dataset roots, the same dry-run/apply pair
and (where it loads the tagger) the same model knobs. Those used to be typed
out once per parser with byte-identical help strings, which matters more than
tidiness: :mod:`anime_tools.gui.stages` introspects ``build_parser()`` to build
the GUI form, so a dropped ``dest=``, a drifted default or a lost ``--foo-bar``
alias silently changes the form rather than failing anything.

Declaration *order* is part of that contract too — ``fields_of`` walks
``parser._actions`` in order and the form follows it — so each helper adds its
flags as one contiguous block, and a caller places the block exactly where the
flags used to sit. Where two stages genuinely say different things about the
same flag (``--src`` is read-only for the clause rewrite, written by autotag)
the help text is an argument, not a fork.
"""

from __future__ import annotations

import argparse

from anime_tools import workspace as WS
from anime_tools.tagger.dbv4_meta import DEFAULT_TAGGER_DIR

PATTERN_HELP = "fnmatch glob (| to OR-combine) on the path relative to {root}"


def add_path_pattern_arg(p: argparse.ArgumentParser, *, help: str) -> None:
    """``--path_pattern`` — the one scope knob the GUI narrows to a single image.

    Its ``dest`` is what :data:`anime_tools.gui.stages.SCOPE_FIELD` looks for,
    so the dual spelling and the ``dest=`` are load-bearing.
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
    every stage that walks the resized tree, ``--src`` for the resize stage
    that populates it.
    """
    p.add_argument("--src", default="image_dataset", help=src_help)
    p.add_argument("--dst", default=WS.RESIZED, help=dst_help)
    add_path_pattern_arg(p, help=PATTERN_HELP.format(root=pattern_root))


def add_apply_args(
    p: argparse.ArgumentParser, *, apply_help: str, from_report_help: str
) -> None:
    """``--apply`` / ``--from_report``: dry-run-by-default and its replay.

    Kept together and separate from :func:`add_report_dir_arg` because the
    multiview audit slots its verdict/confidence gate between the two.
    """
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
    :data:`anime_tools.gui.stages.AUTO_FIELDS`, never shown and never sent, and
    every stage resolves it in-process through
    :func:`anime_tools._device.resolve_device` at the model-load site — so the
    torch-free ``--from_report`` replay path never pays for the answer.
    """
    p.add_argument(
        "--tagger_dir",
        "--tagger-dir",
        dest="tagger_dir",
        default=None,
        help=f"Anima Tagger checkpoint dir (default: {DEFAULT_TAGGER_DIR})",
    )
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")


def make_progress(every: int, *, first: bool = False):
    """A ``progress(index, total, detail)`` that prints one line every ``every``.

    The last line always prints, so a run under ``every`` images still says it
    finished; ``first`` also prints image 1, which is what autotag wants to
    show the model finished loading.
    """

    def progress(index: int, total: int, detail: str) -> None:
        if index == total or index % every == 0 or (first and index == 1):
            print(f"  [{index}/{total}] {detail}", flush=True)

    return progress
