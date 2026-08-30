"""Re-apply a stage's dry-run ``report.json`` without re-running its models.

Every caption stage here is a dry run by default and a second ``--apply`` run
for real — which, for ``autotag_captions`` / ``position_captions`` /
``audit_multiview``, means loading the Anima Tagger (and SAM3) a second time and
recomputing proposals the dry run already wrote down. The reports already carry
everything a write needs: the destination caption path, the verbatim *before*
text, and the exact proposed text. This module replays them.

**Torch-free by construction** — stdlib plus :mod:`anime_tools.path_filter` and
(lazily) :mod:`anime_tools.captions.variants`. A ``--from_report --apply`` run
must never import a model; ``tests/test_stage_replay.py`` pins that in a
subprocess.

Three things make a replay safe to trust:

*Root agreement*
    The source report records the ``src``/``dst`` it walked. A replay whose own
    roots differ is refused outright — the recorded ``caption_path`` values are
    relative to those roots, so a mismatch would write real text into the wrong
    tree.

*Per-row drift*
    The recorded before-text is compared against what is on disk **now**. A
    caption edited between the two passes is skipped and counted
    (``skip:drifted``), never overwritten. A file that already holds the
    proposal is ``skip:already-applied`` — idempotent, so a crashed replay can
    simply be re-run.

*Already-applied reports*
    A report whose own ``apply``/``applied`` flag is true is refused: its
    before-text describes the pre-apply world, so every row would read as
    drifted and the run would look mysteriously empty. Refusing says why. The
    replay's own report is written under a *different* name
    (:data:`REPLAY_REPORT_NAME`) so a replay never clobbers the dry run it
    reads — re-running is always possible.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from anime_tools.path_filter import filter_paths_by_glob

# The replay pass writes here, never over the dry run's ``report.json`` — the
# usual invocation points ``--from_report`` and ``--report_dir`` at the same
# directory, and clobbering the input would make a re-run impossible.
REPLAY_REPORT_NAME = "apply_report.json"

# Where a stage records whether its own run wrote anything.
_APPLIED_KEYS = ("apply", "applied")


class StaleReportError(RuntimeError):
    """The report cannot be replayed against this run (roots, shape, or state)."""


@dataclass(frozen=True)
class ReplaySpec:
    """How to read one stage's report and where its proposals get written.

    ``target_root`` is the tree ``caption_path`` is relative to: ``"src"`` for
    the stages that write the caption master (autotag, multiview audit),
    ``"dst"`` for the clause rewrite, which only ever touches the derived
    caption under ``post_image_dataset/resized/``.
    """

    stage: str
    # Report container keys — ``rows``/``stats`` (autotag) or ``images``/``summary``
    # (position clauses, multiview audit).
    rows_key: str = "rows"
    stats_key: str = "stats"
    # A row is writable when its status matches, and/or ``row_filter`` says so.
    # The multiview audit gates on verdict/confidence instead of a status field,
    # so it supplies a closure over the CLI's own gate.
    ok_status: str | None = None
    row_filter: Callable[[Mapping[str, object]], bool] | None = None
    before_field: str = "existing"
    after_field: str = "proposed"
    target_root: str = "src"
    # ``audit_multiview`` writes a trailing newline; the other two do not. Kept
    # byte-exact so a replay and a native apply produce identical files.
    newline: bool = False
    # The clause rewrite must drop ``{stem}.variants.txt`` — the sidecar wins
    # over ``{stem}.txt`` at encode time, so a stale one keeps training the
    # pre-clause caption no matter how fresh the caption is.
    drop_variants: bool = False


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
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StaleReportError(f"report is not readable JSON: {path} ({exc})") from exc
    if not isinstance(report, dict):
        raise StaleReportError(f"report is not a stage report object: {path}")
    return report


def report_meta(report: Mapping[str, object]) -> dict:
    """Flatten a report's scalar metadata.

    ``autotag_captions`` keeps ``src``/``dst``/``apply`` at the top level;
    ``position_captions`` and ``audit_multiview`` keep theirs under
    ``summary``. Both are read the same way so callers need not care.
    """

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
    :func:`replay_rows`): the report records the roots it walked and they match
    this run's, and the report is a dry run rather than one that already wrote.
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

    Matched exactly the way the live pass matches it — against the image path
    relative to the resized dir — so filtering a replay selects the same subset
    a filtered live run would have proposed.
    """
    paths = [str(dst / str(r.get("image") or "")) for r in rows]
    return filter_paths_by_glob(paths, str(dst), pattern)


def _write_caption(path: Path, text: str, spec: ReplaySpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if spec.newline else ""), encoding="utf-8")
    if spec.drop_variants:
        from anime_tools.captions.variants import variants_sidecar_path

        sidecar = variants_sidecar_path(path)
        if sidecar.exists():
            sidecar.unlink()


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

    Loads no model — that is the whole point. Rows the report did not mark
    writable, and rows the ``path_pattern`` excludes, are counted and dropped;
    everything else is checked against the file on disk and either written or
    skipped with a reason. Returns ``(rows, stats)`` where every returned row
    reached the on-disk check, so the caller can show exactly what drifted.
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
            entry.status = "skip:no-caption-path"
            stats.skip("no-caption-path")
            continue
        if not after or after == before:
            entry.status = "skip:no-proposal"
            stats.skip("no-proposal")
            continue

        target = root / caption_path
        if not target.exists():
            # An absent file is legitimate only where the dry run also saw none
            # (autotag's ``missing`` mode creates the master caption).
            if before:
                entry.status = "skip:missing-caption"
                stats.skip("missing-caption")
                continue
            current = ""
        else:
            current = target.read_text(encoding="utf-8").strip()

        if current == after:
            entry.status = "skip:already-applied"
            stats.skip("already-applied")
            continue
        if current != before:
            entry.status = "skip:drifted"
            stats.skip("drifted")
            continue

        if apply:
            _write_caption(target, after, spec)
            entry.status = "written"
            stats.written += 1
        else:
            entry.status = "would-write"
            stats.would_write += 1

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

    Container keys follow the stage (``stats``/``rows`` for autotag,
    ``summary``/``images`` for the other two), so an existing reader keeps
    working. Two additions are common to all of them: ``from_report`` (the dry
    run this replayed) and **``written`` — the relative image paths actually
    written**, which is what a UI should read to reload exactly the affected
    dataset items.
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
    # Carry the knobs that shaped the proposals, so the replay report still says
    # what produced the text it wrote.
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
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / REPLAY_REPORT_NAME
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def print_replay(rows: Sequence[ReplayRow], stats: ReplayStats, *, apply: bool) -> None:
    """Human summary of a replay — the drifted rows named, since they are the
    whole reason a replay can be less than a full apply."""
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
