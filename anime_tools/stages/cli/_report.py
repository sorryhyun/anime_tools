"""Writing a stage's ``report.json``, and the epilogue that follows it.

Every stage ends the same three ways: a JSON report written with the same
options, the four-key header a replay reads back, and a line saying whether
anything was actually written. The header is the load-bearing part —
:func:`anime_tools.stages.replay.validate_report` refuses a report that does
not record the roots it walked or that was itself written by an ``--apply``
run, so a stage that forgets a key produces a report nothing can replay.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from anime_tools._json import write_json

DRY_RUN_NOTE = "\nDry run — no captions written. Re-run with --apply to write."
"""What every caption stage says when it wrote nothing. One string, because
"add --apply" is the single instruction the dry-run default exists to give."""


def stage_report_header(
    *, src: Path, dst: Path, path_pattern: str | None, apply: bool
) -> dict[str, object]:
    """The keys ``validate_report`` checks, in the shape it reads them.

    Both spellings of the applied flag are emitted: the stages disagreed
    (``apply`` for autotag, ``applied`` for the other two) and
    :func:`anime_tools.stages.replay.report_meta` reads either, so writing both
    costs a key and removes the question. The roots are recorded because a
    report's row paths are relative to them — replaying across trees would
    write real text into the wrong place.
    """
    return {
        "applied": bool(apply),
        "apply": bool(apply),
        "src": str(src),
        "dst": str(dst),
        "path_pattern": path_pattern,
    }


def write_stage_report(report_dir: Path, payload: Mapping[str, object]) -> Path:
    """Write ``report.json`` under ``report_dir``, returning where it landed.

    ``ensure_ascii=False`` is not cosmetic: the dataset paths and captions are
    Korean and Japanese, and an escaped report is unreadable in review.
    """
    return write_json(report_dir / "report.json", payload)


def print_dry_run_footer(apply: bool, note: str | None = None) -> None:
    """Close a run: the dry-run instruction, or what an apply just changed."""
    if not apply:
        print(DRY_RUN_NOTE)
    elif note:
        print(note)
