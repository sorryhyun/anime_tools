"""Publish the workspace to the paths the trainer reads.

The one operation in the package that writes outside ``workspace/``: resized
images, masks and captions are copied under ``--out``.

Dry-run by default. ``--apply`` copies for real, re-deciding every row against
the destination, so a file edited since the dry run is reported rather than
clobbered. Taking an export back is the GUI's Undo, not a flag here.
"""

from __future__ import annotations

import argparse

from anime_tools import workspace as WS
from anime_tools.stages.cli._args import (
    add_dataset_args,
    add_report_dir_arg,
)
from anime_tools.stages.requests import DEFAULT_EXPORT_INDEX, ExportRequest

DEFAULT_REPORT_DIR = ExportRequest.report_dir
DEFAULT_INDEX = DEFAULT_EXPORT_INDEX


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(
        p,
        src_help="Caption master dir — where a revised master publishes back to",
        dst_help="Resized tree to publish (the workspace's)",
        pattern_root="--dst",
    )
    p.add_argument(
        "--masks", default=WS.MASKS, help=f"Workspace mask dir (default: {WS.MASKS})"
    )
    p.add_argument(
        "--master",
        default=WS.DEFAULT_ROOTS["master"],
        help=(
            "Revised-master overlay to publish over --src (default: "
            f"{WS.DEFAULT_ROOTS['master']})"
        ),
    )
    p.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help=f"caption_index.json to publish (default: {DEFAULT_INDEX})",
    )
    p.add_argument(
        "--out",
        default=WS.EXPORT_ROOT,
        help=(
            "Export root: resized/, masks/ and captions/ land under it "
            f"(default: {WS.EXPORT_ROOT}). The tree the trainer reads."
        ),
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Copy for real (default: list what would be copied and stop)",
    )
    add_report_dir_arg(p, DEFAULT_REPORT_DIR)
    return p


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_export

    try:
        run_export(ExportRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
