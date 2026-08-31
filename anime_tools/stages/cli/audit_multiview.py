"""Audit `1girl` captions for images that are really several views of one girl.

Thin CLI over ``anime_tools.stages.multiview_audit``: loads SAM3 and the Anima
Tagger (the same two models ``caption-position`` uses, via the same builders),
sweeps the images that pipeline skips as ``single-subject``, and reports every
one where the ``girl`` prompt finds two or more subjects.

Dry-run by default. ``--apply`` writes the missing tag into the **caption
master** — unlike the clause rewrite, which only ever touches the derived
caption — because a missing ``multiple views`` is a fact about the picture that
every later stage should read down from. Follow it with ``make preprocess-te``.

GOTCHA: ``image_dataset/`` is gitignored, so an ``--apply`` is not
git-recoverable. ``report.json`` carries the verbatim before-text of every
caption it touched; keep it.

``--from_report <report.json>`` replays a dry run's findings instead of
re-auditing — the report holds the caption path, the before-text and the
proposal, so the write needs **no SAM3 and no tagger**. The verdict/confidence
gate is still applied at replay time, so one audit pass can be replayed at
several tiers; a caption edited since the audit is skipped and counted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from anime_tools._env import resolve_path
from anime_tools.stages.cli._args import (
    add_apply_args,
    add_dataset_args,
    add_model_args,
    add_report_dir_arg,
    make_progress,
)
from anime_tools.stages.cli._detection import (
    add_checkpoint_arg,
    add_detection_args,
    detection_options,
)
from anime_tools.stages.cli._models import load_tagger
from anime_tools.stages.cli._report import (
    print_dry_run_footer,
    stage_report_header,
    write_stage_report,
)
from anime_tools.stages.cli.position_captions import build_detect_fn
from anime_tools.stages.multiview_audit import (
    DEFAULT_IDENTITY_CONFIDENCE,
    DEFAULT_MULTIVIEW_PROB,
    EXTRA_CHARACTER,
    MULTIPLE_VIEWS,
    apply_findings,
    run_multiview_audit,
)
from anime_tools.stages.position_captions import PositionCaptionOptions
from anime_tools.stages.replay import ReplaySpec, run_replay_cli

DEFAULT_REPORT_DIR = "post_image_dataset/captions/multiview_audit"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p)
    add_apply_args(
        p,
        apply_help="Write the suggested tag into the caption master (default: dry run)",
        from_report_help="Replay a previous dry run's report.json instead of "
        "re-auditing: writes exactly the captions it proposed (still gated by "
        "--apply_verdicts / --apply_confidence) and loads no model. Skips any "
        "caption that changed since. Emits apply_report.json",
    )
    p.add_argument(
        "--apply_verdicts",
        "--apply-verdicts",
        dest="apply_verdicts",
        default=MULTIPLE_VIEWS,
        help=f"Comma-separated verdicts --apply may write "
        f"('{MULTIPLE_VIEWS}', '{EXTRA_CHARACTER}')",
    )
    p.add_argument(
        "--apply_confidence",
        "--apply-confidence",
        dest="apply_confidence",
        default="strong",
        help="Comma-separated confidence tiers --apply may write (strong, weak). "
        "A weak finding has only the geometry behind it — review the crops first",
    )
    add_report_dir_arg(p, DEFAULT_REPORT_DIR)
    p.add_argument(
        "--crops",
        action="store_true",
        help="Export the per-instance crops next to the report (review aid)",
    )
    p.add_argument(
        "--no_sheets",
        "--no-sheets",
        dest="sheets",
        action="store_false",
        help="Skip the per-finding contact sheets. They are the review surface — "
        "boxed original + the crops the tagger saw + the proposed edit, one PNG "
        "per finding under <report_dir>/sheets/, named verdict-first",
    )
    add_checkpoint_arg(p)
    add_model_args(p)
    add_detection_args(
        p,
        score_threshold_help="Subject confidence floor. Raising it trades "
        "recall for a shorter review list; this audit is precision-sensitive "
        "since every hit is read by hand",
        part_prompts_help="Comma-separated body-part prompts, tried only when "
        "'girl' finds fewer than two subjects — recovers a sheet whose second "
        'view is a headless close-up. Off by default; try "buttocks,hips,thighs"',
        name_confidence=True,
    )

    v = p.add_argument_group("verdict")
    v.add_argument(
        "--multiview_threshold",
        "--multiview-threshold",
        dest="multiview_threshold",
        type=float,
        default=DEFAULT_MULTIVIEW_PROB,
        help="Whole-image P(multiple views) at which the tagger counts as a "
        "witness — and, on its own, raises an image detection saw as one box",
    )
    v.add_argument(
        "--identity_confidence",
        "--identity-confidence",
        dest="identity_confidence",
        type=float,
        default=DEFAULT_IDENTITY_CONFIDENCE,
        help="Probability an identity-group winner needs before the verdict "
        "believes it. The group heads are softmax argmaxes, so they name a hair "
        "colour for a headless crop too — lowering this lets those back in",
    )
    v.add_argument(
        "--suggest_counts",
        "--suggest-counts",
        dest="suggest_counts",
        action="store_true",
        help=f"Also propose an 'Ngirls' fix for a '{EXTRA_CHARACTER}' verdict. Off "
        "because the 'girl' prompt does not exclude males — check the "
        "people-count head in the report before trusting any of these",
    )
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _gate(args) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The verdict/confidence tiers ``--apply`` is allowed to write."""
    verdicts = tuple(v.strip() for v in args.apply_verdicts.split(",") if v.strip())
    confidences = tuple(
        c.strip() for c in args.apply_confidence.split(",") if c.strip()
    )
    return verdicts, confidences


# How ``replay`` reads an audit report: ``images``/``summary`` containers and
# the caption **master** (``--src``), which is the one stage that writes it.
# Unlike the other two the writable set is not a row ``status`` but the
# verdict/confidence gate ``apply_findings`` applies, so ``row_filter`` is left
# open here and closed over the CLI's own gate at replay time — replaying one
# audit pass under ``--apply_verdicts "multiple views,extra-character"`` writes
# strictly more of it than the default.
REPLAY_SPEC = ReplaySpec(
    stage="audit_multiview",
    rows_key="images",
    stats_key="summary",
    before_field="caption",
    after_field="proposed",
    target_root="src",
    # ``apply_findings`` writes ``proposed + "\n"``; a replay must be
    # byte-identical to it.
    newline=True,
)


def _run_replay(args, src: Path, dst: Path, report_dir: Path) -> None:
    """Write a previous dry run's findings — no SAM3, no tagger, no pixels."""
    verdicts, confidences = _gate(args)
    run_replay_cli(
        args,
        spec=replace(
            REPLAY_SPEC,
            row_filter=lambda row: (
                row.get("verdict") in verdicts and row.get("confidence") in confidences
            ),
        ),
        src=src,
        dst=dst,
        report_dir=report_dir,
        notes=[f"gate: verdicts={list(verdicts)} confidences={list(confidences)}"],
        after_write_note=lambda stats: (
            f"\n{stats.written} caption(s) written to the master ({src}). Run "
            "`make preprocess-te` now to re-encode. The master is gitignored — "
            "the replayed report holds the before-text if you need to back "
            "this out."
        ),
    )


def main() -> None:
    args = parse_args()
    src = resolve_path(args.src)
    dst = resolve_path(args.dst)
    report_dir = resolve_path(args.report_dir)

    if args.from_report:
        _run_replay(args, src, dst, report_dir)
        return

    detect_fn, part_detect_fn, sam_model, sam_processor = build_detect_fn(args)
    # Loaded here and not in parse_args(): the --from_report replay returns
    # before this and must stay torch-free.
    tagger, vocabulary, _ckpt_dir = load_tagger(args)

    # Two subjects is what the audit is *for*, so it pins min_instances rather
    # than exposing it; everything else is the position stage's detector,
    # verbatim, because the audit sweeps what that stage skipped.
    options = PositionCaptionOptions(
        **detection_options(args, min_instances=2, name_confidence=args.name_confidence)
    )

    rows, stats = run_multiview_audit(
        resized_dir=dst,
        source_dir=src,
        detect_fn=detect_fn,
        tag_fn=tagger.predict,
        vocabulary=vocabulary,
        options=options,
        path_pattern=args.path_pattern,
        crops_dir=(report_dir / "crops") if args.crops else None,
        sheets_dir=(report_dir / "sheets") if args.sheets else None,
        progress=make_progress(200),
        part_detect_fn=part_detect_fn,
        multiview_threshold=args.multiview_threshold,
        identity_confidence=args.identity_confidence,
        suggest_counts=args.suggest_counts,
    )
    del sam_processor, sam_model

    written: list[tuple[str, str, str]] = []
    apply_skipped: dict[str, int] = {}
    if args.apply:
        verdicts, confidences = _gate(args)
        written, skipped = apply_findings(
            rows, source_dir=src, verdicts=verdicts, confidences=confidences
        )
        apply_skipped = dict(skipped.most_common())

    summary = {
        # The header carries the roots so ``--from_report`` can refuse to replay
        # this report against a different pair of trees (the row paths are
        # relative to them).
        **stage_report_header(
            src=src, dst=dst, path_pattern=args.path_pattern, apply=args.apply
        ),
        "seen": stats.seen,
        "audited": stats.audited,
        "findings": stats.findings,
        "verdicts": dict(sorted(stats.verdicts.items(), key=lambda kv: -kv[1])),
        "by_confidence": {
            tier: sum(1 for r in rows if r.confidence == tier)
            for tier in ("strong", "weak")
        },
        "actionable": sum(1 for r in rows if r.suggested_tag),
        "by_source": {
            "detection": sum(1 for r in rows if r.source == "detection"),
            "tagger-only": sum(1 for r in rows if r.source == "tagger-only"),
        },
        "written": len(written),
        # Why a gated row was not written — ``drifted`` (the master moved since
        # the audit) and ``already-applied`` used to be indistinguishable from
        # "the gate rejected it".
        "apply_skipped": apply_skipped,
        "part_prompts": list(options.part_prompts),
        "part_recovered": sum(
            1 for r in rows if any(c.source != "subject" for c in r.crops)
        ),
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
    }
    report_path = write_stage_report(
        report_dir,
        {
            "summary": summary,
            "images": [asdict(r) for r in rows],
            "written": [
                {"caption_path": rel, "before": before, "after": after}
                for rel, before, after in written
            ],
        },
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_path}")
    if args.sheets:
        print(f"sheets: {report_dir / 'sheets'} (one PNG per finding, verdict-first)")
    print_dry_run_footer(
        args.apply,
        f"\n{len(written)} caption(s) written to the master ({src}). Run "
        "`make preprocess-te` now to re-encode. The master is gitignored — "
        "report.json holds the before-text if you need to back this out.",
    )


if __name__ == "__main__":
    main()
