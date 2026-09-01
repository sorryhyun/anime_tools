"""Publish the workspace to the paths the trainer reads.

    python -m anime_tools.stages.cli.export_workspace           # what would publish
    python -m anime_tools.stages.cli.export_workspace --apply   # publish it

Dry-run by default: this is the one operation in the package that writes outside
``workspace/``. It loads no model, hence no ``--from_report`` replay — an Apply
runs the pass again and re-decides every row against disk, so a destination
edited since the dry run is reported rather than clobbered.

See :mod:`anime_tools.stages.export_workspace` for the six artifact kinds.
Taking an export back is :func:`anime_tools.stages.export_workspace.revert_export`
over an ``--apply`` run's report, which is what the GUI's **Undo** calls
(``gui.proposals``); it is not a flag here, because an undo is a *second* write
outside the workspace and belongs beside the run whose report it reads rather
than in an argv typed by hand.
"""

from __future__ import annotations

import argparse

from anime_tools import workspace as WS
from anime_tools._env import resolve_path
from anime_tools.stages.cli._args import (
    add_dataset_args,
    add_report_dir_arg,
    make_progress,
)
from anime_tools.stages.cli._report import (
    print_dry_run_footer,
    stage_report_header,
    write_stage_report,
)
from anime_tools.stages.export_workspace import ExportPaths, run_export

DEFAULT_REPORT_DIR = f"{WS.REPORTS}/export"
DEFAULT_INDEX = f"{WS.REPORTS}/caption_index.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # --src and --dst keep their package-wide meanings; this is just the one
    # stage that reads the resized tree and writes the master.
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


def main() -> None:
    args = build_parser().parse_args()
    apply = bool(args.apply)
    report_dir = resolve_path(args.report_dir)

    paths = ExportPaths(
        resized=resolve_path(args.dst),
        masks=resolve_path(args.masks),
        master=resolve_path(args.master),
        index=resolve_path(args.index),
        src=resolve_path(args.src),
        out=resolve_path(args.out),
    )
    pattern = str(args.path_pattern or "*")

    if not paths.resized.is_dir():
        raise SystemExit(
            f"nothing to export: {paths.resized} does not exist. "
            "Run the Resize stage first."
        )
    rows, stats = run_export(
        paths,
        path_pattern=pattern,
        apply=apply,
        progress=make_progress(50),
    )
    note = f"published: {stats.created} created, {stats.overwrote} overwritten"

    report = {
        **stage_report_header(
            src=paths.src, dst=paths.resized, path_pattern=pattern, apply=apply
        ),
        "out": str(paths.out),
        "stats": stats.to_dict(),
        "rows": [r.to_dict() for r in rows],
    }
    path = write_stage_report(report_dir, report)
    print(f"\nreport → {path}")
    print_dry_run_footer(apply, note)


if __name__ == "__main__":
    main()
