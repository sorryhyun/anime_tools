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
from dataclasses import asdict
from pathlib import Path

import numpy as np

# Monkey-patch numpy for sam3 compatibility (upstream pins numpy<2 and uses np.bool)
if not hasattr(np, "bool"):
    np.bool = np.bool_

from anime_tools._device import resolve_device
from anime_tools._env import resolve_path
from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT
from anime_tools.stages.cli.position_captions import build_detect_fn
from anime_tools.stages.instance_detection import (
    DEFAULT_SUBJECT_PROMPT_EMBED,
)
from anime_tools.stages.multiview_audit import (
    DEFAULT_IDENTITY_CONFIDENCE,
    DEFAULT_MULTIVIEW_PROB,
    EXTRA_CHARACTER,
    MULTIPLE_VIEWS,
    apply_findings,
    run_multiview_audit,
)
from anime_tools.stages.position_captions import (
    PositionCaptionOptions,
    load_clause_vocabulary,
)
from anime_tools.stages.replay import (
    ReplaySpec,
    StaleReportError,
    print_replay,
    run_replay,
)
from anime_tools.tagger.dbv4_meta import DEFAULT_TAGGER_DIR

DEFAULT_REPORT_DIR = "post_image_dataset/captions/multiview_audit"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="image_dataset", help="Caption master dir")
    p.add_argument("--dst", default="post_image_dataset/resized", help="Resized images")
    p.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help="fnmatch glob (| to OR-combine) on the path relative to --dst",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write the suggested tag into the caption master (default: dry run)",
    )
    p.add_argument(
        "--from_report",
        "--from-report",
        dest="from_report",
        default=None,
        help="Replay a previous dry run's report.json instead of re-auditing: "
        "writes exactly the captions it proposed (still gated by "
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
    p.add_argument(
        "--report_dir",
        "--report-dir",
        dest="report_dir",
        default=DEFAULT_REPORT_DIR,
        help=f"Where report.json lands (default: {DEFAULT_REPORT_DIR})",
    )
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
    p.add_argument("--checkpoint", default=DEFAULT_SAM3_CHECKPOINT, help="SAM3 weights")
    p.add_argument(
        "--tagger_dir",
        "--tagger-dir",
        dest="tagger_dir",
        default=None,
        help=f"Anima Tagger checkpoint dir (default: {DEFAULT_TAGGER_DIR})",
    )
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")

    g = p.add_argument_group("detection")
    g.add_argument("--prompt", default="girl", help="SAM3 text prompt for a subject")
    g.add_argument(
        "--prompt_embed",
        default=DEFAULT_SUBJECT_PROMPT_EMBED,
        help="learned soft prompt (.safetensors) standing in for --prompt on the "
        "subject pass (part prompts stay text); default = shipped, `none` = text",
    )
    g.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Subject confidence floor. Raising it trades recall for a shorter "
        "review list; this audit is precision-sensitive since every hit is "
        "read by hand",
    )
    g.add_argument("--retry_score_threshold", type=float, default=0.35)
    g.add_argument(
        "--part_prompts",
        "--part-prompts",
        dest="part_prompts",
        default="",
        help="Comma-separated body-part prompts, tried only when 'girl' finds "
        "fewer than two subjects — recovers a sheet whose second view is a "
        'headless close-up. Off by default; try "buttocks,hips,thighs"',
    )
    g.add_argument("--part_score_threshold", type=float, default=0.5)
    g.add_argument("--part_containment_threshold", type=float, default=0.7)
    g.add_argument("--iou_threshold", type=float, default=0.65)
    g.add_argument("--containment_threshold", type=float, default=1.01)
    g.add_argument(
        "--mask_containment_threshold",
        "--mask-containment-threshold",
        dest="mask_containment_threshold",
        type=float,
        default=0.8,
        help="Suppress a detection whose MASK is this nested inside a kept "
        "one. On by default, unlike its box counterpart: a second girl in "
        "front of the first nests identically by box but her mask is disjoint. "
        ">1.0 disables (the pre-2026-08-19 behaviour). Same default as the "
        "position stage so the audit dedupes detections the same way",
    )
    g.add_argument(
        "--dedupe_fill_ratio",
        type=float,
        default=2.0,
        help="Mask-quality tie-break inside an NMS-matched pair; 0 = off "
        "(score-only survivor). See docs/experimental/multiview_audit.md §5.",
    )
    g.add_argument("--min_area_frac", type=float, default=0.005)
    g.add_argument("--pad", type=float, default=0.06)
    g.add_argument("--row_tol", type=float, default=0.25)
    g.add_argument("--max_instances", type=int, default=8)
    g.add_argument("--name_confidence", type=float, default=0.5)

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


def _run_replay(args, src: Path, dst: Path, report_dir: Path) -> None:
    """Write a previous dry run's findings — no SAM3, no tagger, no pixels.

    Unlike the other two stages, the writable set is not a row ``status`` but
    the same verdict/confidence gate :func:`apply_findings` applies, so the
    tiers are still chosen at replay time: replaying a report under
    ``--apply_verdicts multiple views,extra-character`` writes strictly more of
    it than the default, off one audit pass.
    """
    verdicts, confidences = _gate(args)
    spec = ReplaySpec(
        stage="audit_multiview",
        rows_key="images",
        stats_key="summary",
        row_filter=lambda row: (
            row.get("verdict") in verdicts and row.get("confidence") in confidences
        ),
        before_field="caption",
        after_field="proposed",
        target_root="src",
        # ``apply_findings`` writes ``proposed + "\n"``; a replay must be
        # byte-identical to it.
        newline=True,
    )
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
    print(f"gate: verdicts={list(verdicts)} confidences={list(confidences)}")
    print_replay(rows, stats, apply=args.apply)
    print(f"\nreport: {out_path}")
    if args.apply and stats.written:
        print(
            f"\n{stats.written} caption(s) written to the master ({src}). Run "
            "`make preprocess-te` now to re-encode. The master is gitignored — "
            "the replayed report holds the before-text if you need to back "
            "this out."
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

    from anime_tools.tagger.tagger import AnimaTagger, ensure_tagger_checkpoint

    ckpt_dir = ensure_tagger_checkpoint(
        resolve_path(args.tagger_dir or DEFAULT_TAGGER_DIR)
    )
    print(f"Loading Anima Tagger from {ckpt_dir}...", flush=True)
    # Resolved here and not in parse_args(): the --from_report replay returns
    # before this and must stay torch-free, and resolving imports torch.
    tagger = AnimaTagger(ckpt_dir, device=resolve_device(args.device))
    vocabulary = load_clause_vocabulary(ckpt_dir)

    options = PositionCaptionOptions(
        prompt=args.prompt,
        score_threshold=args.score_threshold,
        retry_score_threshold=args.retry_score_threshold,
        part_prompts=tuple(
            t.strip() for t in args.part_prompts.split(",") if t.strip()
        ),
        part_score_threshold=args.part_score_threshold,
        part_containment_threshold=args.part_containment_threshold,
        iou_threshold=args.iou_threshold,
        containment_threshold=args.containment_threshold,
        mask_containment_threshold=args.mask_containment_threshold,
        dedupe_fill_ratio=args.dedupe_fill_ratio,
        min_area_frac=args.min_area_frac,
        pad=args.pad,
        row_tol=args.row_tol,
        min_instances=2,
        max_instances=args.max_instances,
        name_confidence=args.name_confidence,
    )

    def progress(index: int, total: int, rel: str) -> None:
        if index % 200 == 0 or index == total:
            print(f"  [{index}/{total}] {rel}", flush=True)

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
        progress=progress,
        part_detect_fn=part_detect_fn,
        multiview_threshold=args.multiview_threshold,
        identity_confidence=args.identity_confidence,
        suggest_counts=args.suggest_counts,
    )
    del sam_processor, sam_model

    written: list[tuple[str, str, str]] = []
    if args.apply:
        verdicts, confidences = _gate(args)
        written = apply_findings(
            rows, source_dir=src, verdicts=verdicts, confidences=confidences
        )

    summary = {
        "applied": bool(args.apply),
        # Recorded so ``--from_report`` can refuse to replay this report against
        # a different pair of trees (the row paths are relative to these).
        "src": str(src),
        "dst": str(dst),
        "path_pattern": args.path_pattern,
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
        "part_prompts": list(options.part_prompts),
        "part_recovered": sum(
            1 for r in rows if any(c.source != "subject" for c in r.crops)
        ),
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "images": [asdict(r) for r in rows],
                "written": [
                    {"caption_path": rel, "before": before, "after": after}
                    for rel, before, after in written
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_dir / 'report.json'}")
    if args.sheets:
        print(f"sheets: {report_dir / 'sheets'} (one PNG per finding, verdict-first)")
    if args.apply:
        print(
            f"\n{len(written)} caption(s) written to the master ({src}). Run "
            "`make preprocess-te` now to re-encode. The master is gitignored — "
            "report.json holds the before-text if you need to back this out."
        )
    else:
        print("\nDry run — no captions written. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
