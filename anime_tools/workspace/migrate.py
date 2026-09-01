"""Move a pre-workspace tree into ``workspace/``.

Moves ``post_image_dataset/{resized,masks,captions,groups}`` — the four
directories an install predating the workspace left on the Export side of the
line. Dry-run by default; ``--apply`` moves for real. It is a directory rename
and never a merge: an existing destination is reported and skipped. A dataset
root pinned to a legacy path in the GUI settings file is named but not rewritten,
so clear it yourself in ⚙ Settings.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from anime_tools._env import curation_home, workspace_dir
from anime_tools.workspace import EXPORT_ROOT, LEGACY_DIRS, LEGACY_ROOTS


def plan_moves(home: Path, workspace: Path) -> list[tuple[str, Path, Path, str]]:
    """``(what, from, to, status)`` for every directory the workspace claims.

    Status is decided here, not at the move, so the dry run and ``--apply`` see
    the same rows.
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

    # Imported late: the GUI may never have run here.
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
