"""Move a pre-workspace tree into ``workspace/``.

``python -m anime_tools.workspace.migrate [--apply]``

An install predating the workspace has ``post_image_dataset/{resized,masks,
captions,groups}`` on the wrong side of the tools-write / Export-publishes line.
This moves those four directories.

Dry-run by default and ``--apply`` for real, like every stage CLI. It is a
directory *rename*, not a copy, so a large resized tree moves instantly on one
volume and the operation is its own undo. It never merges: an existing
destination is reported and skipped, because picking a per-file winner is not
this script's call.

It deliberately does **not** rewrite the GUI settings file — a root explicitly
pinned to a legacy path is named on the way past and left alone, rather than
silently edited out from under the user who typed it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from anime_tools._env import curation_home, workspace_dir
from anime_tools.workspace import EXPORT_ROOT, LEGACY_DIRS, LEGACY_ROOTS


def plan_moves(home: Path, workspace: Path) -> list[tuple[str, Path, Path, str]]:
    """``(what, from, to, status)`` for every directory the workspace claims.

    Status is decided here, not at the move, so the dry run and the ``--apply``
    see the same rows and differ only in whether :func:`shutil.move` runs.
    """
    legacy = {Path(v).name: home / v for v in LEGACY_ROOTS.values()}
    legacy.update({name: home / EXPORT_ROOT / name for name in LEGACY_DIRS})

    rows = []
    for name, src in legacy.items():
        dst = workspace / name
        if not src.is_dir():
            status = "absent"
        elif dst.exists():
            status = "both-exist"
        else:
            status = "would-move"
        rows.append((name, src, dst, status))
    return rows


def pinned_roots(saved: dict) -> list[tuple[str, str]]:
    """Saved dataset roots that still name a pre-workspace path.

    ``(root name, path)``. A blank or absent value is not pinned — it follows
    the defaults.
    """
    return [
        (name, str(saved.get(name)).strip())
        for name, legacy in LEGACY_ROOTS.items()
        if str(saved.get(name) or "").strip() == legacy
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m anime_tools.workspace.migrate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="move for real (default: print what would move and stop)",
    )
    args = p.parse_args(argv)

    home = curation_home()
    workspace = workspace_dir()
    print(f"home:      {home}")
    print(f"workspace: {workspace}\n")

    rows = plan_moves(home, workspace)
    moved = 0
    for name, src, dst, status in rows:
        if status == "would-move" and args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            status = "moved"
            moved += 1
        arrow = f"{src.relative_to(home)} -> {dst.relative_to(home)}"
        print(f"  {status:12} {name:10} {arrow}")

    # Late and defensive: the GUI may never have run here, and a warning is not
    # worth failing a migration over.
    try:
        from anime_tools.gui.dataset import SETTINGS_KEY
        from anime_tools.gui.settings import load_settings, settings_path

        pinned = pinned_roots(load_settings().get(SETTINGS_KEY) or {})
    except Exception:  # noqa: BLE001 - the warning is a courtesy, not a step
        pinned = []
    if pinned:
        print(f"\n! {settings_path()} pins pre-workspace roots:")
        for name, path in pinned:
            print(f"    {name} = {path}")
        print(
            "  Only the defaults moved, so these still point at the old tree.\n"
            "  Clear them in ⚙ Settings › Dataset roots to follow the workspace."
        )

    if not args.apply:
        pending = sum(1 for *_, st in rows if st == "would-move")
        print(f"\nDRY RUN — {pending} to move. Re-run with --apply.")
    else:
        print(f"\nmoved {moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
