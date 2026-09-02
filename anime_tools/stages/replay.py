"""Re-apply a stage's dry-run ``report.json`` without re-running its models.

A stage report carries everything a write needs — the destination caption path,
the verbatim *before* text, the exact proposed text — so an ``--apply`` pass can
replay it instead of reloading the tagger (and SAM3) to recompute it.

**Torch-free by construction**: a ``--from_report --apply`` run must never
import a model.

Three checks make a replay safe: the report's recorded ``src``/``dst`` must
match this run's (``caption_path`` is relative to them); the recorded
before-text is compared against disk per row (``skip:drifted`` /
``skip:already-applied``); and a report from an ``--apply`` run is refused,
since its before-text describes the pre-apply world. The replay writes its own
report under :data:`REPLAY_REPORT_NAME`.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from anime_tools._env import resolve_path
from anime_tools._json import read_json, write_json
from anime_tools.contract import REPLAY_REPORT_NAME, ReplaySpec
from anime_tools.path_filter import filter_paths_by_glob

from ._caption_io import read_caption, write_caption

__all__ = [
    "REPLAY_REPORT_NAME",
    "ReplaySpec",
]  # re-exported from ``anime_tools.contract``

# Where a stage records whether its own run wrote anything.
_APPLIED_KEYS = ("apply", "applied")


class StaleReportError(RuntimeError):
    """The report cannot be replayed against this run (roots, shape, or state)."""


@dataclass
class ReplayRow:
    """One replayed proposal and what happened to it."""

    image: str = ""
    caption_path: str = ""
    before: str = ""
    after: str = ""
    status: str = "would-write"


@dataclass
class ReplayStats:
    rows: int = 0
    candidates: int = 0
    written: int = 0
    would_write: int = 0
    skipped: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1


def load_report(path: Path) -> dict:
    """Read a stage report, with a legible error for the usual mistakes."""
    if not path.exists():
        raise StaleReportError(f"report not found: {path}")
    try:
        report = read_json(path)
    except (OSError, ValueError) as exc:
        raise StaleReportError(f"report is not readable JSON: {path} ({exc})") from exc
    if not isinstance(report, dict):
        raise StaleReportError(f"report is not a stage report object: {path}")
    return report


def report_meta(report: Mapping[str, object]) -> dict:
    """Flatten a report's scalar metadata: autotag keeps ``src``/``dst``/``apply``
    at the top level, the other two under ``summary``."""

    def scalars(d: Mapping[str, object]) -> dict:
        return {k: v for k, v in d.items() if not isinstance(v, (list, dict))}

    meta = scalars(report)
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        meta.update(scalars(summary))
    return meta


def report_rows(report: Mapping[str, object], spec: ReplaySpec) -> list[dict]:
    rows = report.get(spec.rows_key)
    if not isinstance(rows, list):
        raise StaleReportError(
            f"report has no {spec.rows_key!r} list — is it a {spec.stage} report?"
        )
    return [r for r in rows if isinstance(r, Mapping)]


def validate_report(
    report: Mapping[str, object],
    *,
    spec: ReplaySpec,
    src: Path,
    dst: Path,
    report_path: Path | None = None,
) -> None:
    """Refuse a report that cannot be safely replayed against this run.

    Checked here (per-row drift is checked at write time, in
    :func:`replay_rows`): the roots the report walked match this run's, and the
    report is a dry run rather than one that already wrote.
    """
    where = f" ({report_path})" if report_path is not None else ""
    meta = report_meta(report)

    for key in _APPLIED_KEYS:
        if bool(meta.get(key)):
            raise StaleReportError(
                f"report{where} was itself written by an --apply run "
                f"({key}=true). Its before-text describes the pre-apply state, "
                "so every row would read as drifted. Re-run the dry pass to "
                "get a fresh report."
            )

    for key, current in (("src", src), ("dst", dst)):
        recorded = meta.get(key)
        if recorded is None:
            raise StaleReportError(
                f"report{where} does not record {key!r} — it predates "
                "--from_report. Re-run the dry pass to get a replayable report."
            )
        if Path(str(recorded)).resolve() != Path(current).resolve():
            raise StaleReportError(
                f"report{where} walked {key}={recorded!r} but this run uses "
                f"{str(current)!r}. The recorded caption paths are relative to "
                "those roots — refusing to replay across trees."
            )


def _is_writable(row: Mapping[str, object], spec: ReplaySpec) -> bool:
    if spec.ok_status is not None and row.get("status") != spec.ok_status:
        return False
    return spec.row_filter is None or bool(spec.row_filter(row))


def _keep_by_pattern(
    rows: Sequence[Mapping[str, object]], dst: Path, pattern: str | None
):
    """Apply the stage's own ``--path_pattern`` to a replayed report.

    Matched against the image path relative to the resized dir, as the live pass
    matches it, so a filtered replay selects the same subset.
    """
    paths = [str(dst / str(r.get("image") or "")) for r in rows]
    return filter_paths_by_glob(paths, str(dst), pattern)


# Everything :func:`apply_one` can answer, in the order it decides them;
# ``written``/``would-write`` are the two outcomes that are not a skip.
APPLY_STATUSES = (
    "no-proposal",
    "missing-caption",
    "already-applied",
    "drifted",
    "would-write",
    "written",
)


def apply_one(
    target: Path,
    before: str,
    after: str,
    *,
    apply: bool,
    newline: bool = False,
    drop_variants: bool = False,
    history_by: str | None = None,
) -> str:
    """Write one proposal if — and only if — the file still says what it said.

    The drift ladder, shared by every stage that writes a caption:

    ``no-proposal``
        Nothing to write, or the proposal equals the before-text.
    ``missing-caption``
        The target is gone. An absent target with a non-empty before-text is
        refused; only autotag's ``missing`` mode legitimately creates one.
    ``already-applied``
        The file already holds the proposal, so a crashed apply can be re-run.
    ``drifted``
        Neither the before-text nor the proposal. Never overwritten.
    ``would-write`` / ``written``
        The write is safe; ``apply`` decides whether it happens.

    Both texts are stripped, so trailing whitespace cannot read as drift.
    """
    before = (before or "").strip()
    after = (after or "").strip()
    if not after or after == before:
        return "no-proposal"

    if target.exists():
        current = read_caption(target)
    elif before:
        return "missing-caption"
    else:
        current = ""

    if current == after:
        return "already-applied"
    if current != before:
        return "drifted"
    if not apply:
        return "would-write"
    write_caption(
        target,
        after,
        newline=newline,
        drop_variants=drop_variants,
        history_by=history_by,
    )
    return "written"


def replay_rows(
    report: Mapping[str, object],
    *,
    spec: ReplaySpec,
    src: Path,
    dst: Path,
    path_pattern: str | None = "*",
    apply: bool = False,
) -> tuple[list[ReplayRow], ReplayStats]:
    """Write (or, without ``apply``, describe) the proposals a report recorded.

    Loads no model. Rows the report did not mark writable, and rows the
    ``path_pattern`` excludes, are counted and dropped; every returned row
    reached the on-disk check.
    """
    rows = report_rows(report, spec)
    root = src if spec.target_root == "src" else dst
    stats = ReplayStats(rows=len(rows))
    keep = _keep_by_pattern(rows, dst, path_pattern)

    out: list[ReplayRow] = []
    for row, matched in zip(rows, keep, strict=True):
        if not _is_writable(row, spec):
            stats.skip("not-writable")
            continue
        if not matched:
            stats.skip("filtered")
            continue
        stats.candidates += 1

        caption_path = str(row.get("caption_path") or "")
        before = str(row.get(spec.before_field) or "").strip()
        after = str(row.get(spec.after_field) or "").strip()
        entry = ReplayRow(
            image=str(row.get("image") or ""),
            caption_path=caption_path,
            before=before,
            after=after,
        )
        out.append(entry)

        if not caption_path:
            # The report's problem rather than the file's, so not in
            # ``apply_one``.
            entry.status = "skip:no-caption-path"
            stats.skip("no-caption-path")
            continue

        status = apply_one(
            root / caption_path,
            before,
            after,
            apply=apply,
            newline=spec.newline,
            drop_variants=spec.drop_variants,
            history_by=spec.history_by,
        )
        if status == "written":
            entry.status = status
            stats.written += 1
        elif status == "would-write":
            entry.status = status
            stats.would_write += 1
        else:
            entry.status = f"skip:{status}"
            stats.skip(status)

    return out, stats


def build_replay_report(
    *,
    spec: ReplaySpec,
    source_report: Mapping[str, object],
    source_report_path: Path,
    src: Path,
    dst: Path,
    path_pattern: str | None,
    apply: bool,
    rows: Sequence[ReplayRow],
    stats: ReplayStats,
) -> dict:
    """The replay's own report, in the shape the stage already emits.

    Container keys follow the stage, plus ``from_report`` and ``written`` — the
    relative image paths actually written, which a UI reads to reload them.
    """
    meta = report_meta(source_report)
    stats_block = {
        "stage": spec.stage,
        "mode": "replay",
        "from_report": str(source_report_path),
        "applied": bool(apply),
        "apply": bool(apply),
        "src": str(src),
        "dst": str(dst),
        "path_pattern": path_pattern,
        "report_rows": stats.rows,
        "candidates": stats.candidates,
        "written": stats.written,
        "would_write": stats.would_write,
        "skipped": dict(stats.skipped.most_common()),
    }
    # Carry the knobs that shaped the proposals.
    for key in ("mode", "min_confidence", "rewrite", "attribution_margin", "prompt"):
        if key in meta:
            stats_block[f"source_{key}"] = meta[key]

    report: dict = {
        spec.stats_key: stats_block,
        spec.rows_key: [vars(r) for r in rows],
        "written": [r.image for r in rows if r.status == "written"],
    }
    if spec.stats_key != "summary":
        # autotag keeps its metadata at the top level next to ``stats``.
        report.update(
            {
                "stage": spec.stage,
                "from_report": str(source_report_path),
                "apply": bool(apply),
                "src": str(src),
                "dst": str(dst),
                "path_pattern": path_pattern,
            }
        )
    return report


def write_replay_report(report: Mapping[str, object], report_dir: Path) -> Path:
    """Write the replay report next to (never over) the dry run's."""
    return write_json(report_dir / REPLAY_REPORT_NAME, report)


def print_replay(rows: Sequence[ReplayRow], stats: ReplayStats, *, apply: bool) -> None:
    """Human summary of a replay, naming the rows that were not written."""
    mark = {"written": "->", "would-write": ".."}
    for row in rows:
        if row.status in mark:
            continue
        print(f"  !! {row.status}: {row.image}")
    print(
        f"\nreport rows={stats.rows} candidates={stats.candidates} "
        f"written={stats.written} would_write={stats.would_write}"
    )
    for reason, count in stats.skipped.most_common():
        print(f"  skip:{reason} {count}")
    if not apply:
        print("\nReplay dry run — nothing written. Add --apply to write.")


def run_replay(
    *,
    spec: ReplaySpec,
    report_path: Path,
    src: Path,
    dst: Path,
    report_dir: Path,
    path_pattern: str | None = "*",
    apply: bool = False,
) -> tuple[list[ReplayRow], ReplayStats, Path]:
    """Load → validate → write → report. The one entry point the CLIs call."""
    source = load_report(report_path)
    validate_report(source, spec=spec, src=src, dst=dst, report_path=report_path)
    rows, stats = replay_rows(
        source,
        spec=spec,
        src=src,
        dst=dst,
        path_pattern=path_pattern,
        apply=apply,
    )
    out_path = write_replay_report(
        build_replay_report(
            spec=spec,
            source_report=source,
            source_report_path=report_path,
            src=src,
            dst=dst,
            path_pattern=path_pattern,
            apply=apply,
            rows=rows,
            stats=stats,
        ),
        report_dir,
    )
    return rows, stats, out_path


def run_replay_cli(
    args: argparse.Namespace,
    *,
    spec: ReplaySpec,
    src: Path,
    dst: Path,
    report_dir: Path,
    notes: Sequence[str] = (),
    after_write_note: str | Callable[[ReplayStats], str] | None = None,
) -> tuple[list[ReplayRow], ReplayStats, Path]:
    """:func:`run_replay` wrapped in the ``--from_report`` half of a stage CLI.

    Turns a :class:`StaleReportError` into a ``SystemExit``, prints the rows and
    the output report, and adds the stage's "now re-encode" epilogue only when
    an apply wrote something. ``notes`` prints extra lines before the rows;
    ``after_write_note`` is that epilogue, a string or a callable over the stats.
    """
    try:
        rows, stats, out_path = run_replay(
            spec=spec,
            report_path=resolve_path(args.from_report),
            src=src,
            dst=dst,
            report_dir=report_dir,
            path_pattern=args.path_pattern,
            apply=args.apply,
        )
    except StaleReportError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"replaying {args.from_report} (no model loaded)")
    for note in notes:
        print(note)
    print_replay(rows, stats, apply=args.apply)
    print(f"\nreport: {out_path}")
    if args.apply and stats.written and after_write_note:
        print(
            after_write_note(stats) if callable(after_write_note) else after_write_note
        )
    return rows, stats, out_path
