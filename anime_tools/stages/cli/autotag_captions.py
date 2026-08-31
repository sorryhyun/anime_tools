#!/usr/bin/env python3
"""Auto-tag the dataset with the Anima Tagger and write the caption master.

Thin CLI over ``anime_tools.stages.autotag``: loads the tagger (auto-downloading
the checkpoint if missing), walks the resized tree, and proposes a caption per
image. The dataset-wide counterpart to the Dataset tab's per-image autotag
button.

**Dry-run is the default.** Nothing is written until ``--apply`` is passed; a
dry run emits ``report.json`` so the proposals can be eyeballed first.

Three modes (``--mode``):

    missing    only images with no caption sidecar (default — never destructive)
    merge      tag everything, append only tags the caption lacks (clauses kept)
    overwrite  replace the caption with the tagger's output (destructive)

After an ``--apply`` run the TE caches are stale but *look* current — caption
edits do not invalidate them. Follow with ``make preprocess-te``, which
regenerates the ``.variants.txt`` sidecars first (those override the CLI dropout
rate, so a stale sidecar would keep training the pre-tag caption).

``--from_report <report.json>`` replays a dry run instead of re-tagging: the
report already carries the destination path and the exact proposed text, so the
apply pass writes them **without loading the tagger**. Rows whose caption
changed since the dry run are skipped and counted, never overwritten; the replay
writes ``apply_report.json`` so it can never clobber the report it read.

    make caption-autotag                              # dry run, missing-only
    make caption-autotag ARGS="--apply"               # write them
    make caption-autotag ARGS="--mode merge --apply"  # top up existing captions
    make preprocess-te                                # re-encode (required)

    # the same two-step, paying for the tagger once:
    python -m anime_tools.stages.cli.autotag_captions
    python -m anime_tools.stages.cli.autotag_captions --apply \
        --from_report post_image_dataset/captions/autotag/report.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from anime_tools._device import resolve_device
from anime_tools._env import resolve_path
from anime_tools.stages.autotag import (
    MODES,
    AutotagOptions,
    build_tag_fn,
    run_autotag_captions,
)
from anime_tools.stages.cli._args import (
    add_apply_args,
    add_dataset_args,
    add_model_args,
    add_report_dir_arg,
    make_progress,
)
from anime_tools.stages.cli._report import (
    print_dry_run_footer,
    stage_report_header,
    write_stage_report,
)
from anime_tools.stages.replay import ReplaySpec, run_replay_cli

DEFAULT_REPORT_DIR = "post_image_dataset/captions/autotag"
TE_NOTE = "captions changed — run `make preprocess-te` to re-encode."


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p)
    p.add_argument(
        "--mode",
        choices=MODES,
        default="missing",
        help="missing: only uncaptioned images (default). merge: append novel "
        "tags to every caption, keeping its position clauses. overwrite: "
        "replace the caption outright (discards hand-written text)",
    )
    p.add_argument(
        "--min_confidence",
        "--min-confidence",
        dest="min_confidence",
        type=float,
        default=0.0,
        help="Extra probability floor on top of the tagger's per-tag F1 "
        "thresholds (0-1). 0 leaves its own decisions untouched",
    )
    add_apply_args(
        p,
        apply_help="Write the proposed captions into the master (default: dry run)",
        from_report_help="Replay a previous dry run's report.json instead of "
        "re-tagging: writes exactly the captions it proposed and loads no "
        "model. Skips any row whose caption changed since. Emits "
        "apply_report.json (never clobbers the report it reads)",
    )
    add_report_dir_arg(p, DEFAULT_REPORT_DIR)
    add_model_args(p)
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


# How ``replay`` reads an autotag report: ``rows``/``stats`` containers, ``ok``
# is the writable status, and the proposal lands in the caption **master**
# (``--src``), which is what this stage writes.
REPLAY_SPEC = ReplaySpec(
    stage="autotag_captions",
    rows_key="rows",
    stats_key="stats",
    ok_status="ok",
    before_field="existing",
    after_field="proposed",
    target_root="src",
)


def _replay(args, *, src: Path, dst: Path, report_dir: Path) -> None:
    """Write a previous dry run's proposals — no tagger, no images opened."""
    run_replay_cli(
        args,
        spec=REPLAY_SPEC,
        src=src,
        dst=dst,
        report_dir=report_dir,
        after_write_note=TE_NOTE,
    )


def main() -> None:
    args = parse_args()
    resized_dir = resolve_path(args.dst)
    source_dir = resolve_path(args.src)
    if not resized_dir.exists():
        raise SystemExit(
            f"resized dir not found: {resized_dir} — run `make preprocess-resize` first"
        )

    report_dir = resolve_path(args.report_dir)
    if args.from_report:
        _replay(args, src=source_dir, dst=resized_dir, report_dir=report_dir)
        return

    options = AutotagOptions(mode=args.mode, min_confidence=args.min_confidence)
    # Resolved here and not in parse_args(): --from_report returned above
    # without importing torch, and resolve_device would have done exactly that.
    device = resolve_device(args.device)
    print(f"Loading Anima Tagger ({device})...", flush=True)
    tag_fn, info = build_tag_fn(
        args.tagger_dir, device=device, min_confidence=args.min_confidence
    )

    rows, stats = run_autotag_captions(
        resized_dir=resized_dir,
        source_dir=source_dir,
        tag_fn=tag_fn,
        options=options,
        path_pattern=args.path_pattern,
        apply=args.apply,
        progress=make_progress(50, first=True),
    )

    report_path = write_stage_report(
        report_dir,
        {
            "mode": args.mode,
            "min_confidence": args.min_confidence,
            **stage_report_header(
                src=source_dir,
                dst=resized_dir,
                path_pattern=args.path_pattern,
                apply=args.apply,
            ),
            **dict(info),
            "stats": {
                "seen": stats.seen,
                "candidates": stats.candidates,
                "proposed": stats.proposed,
                "written": stats.written,
                "skipped": dict(stats.skipped),
            },
            "rows": [asdict(r) for r in rows],
        },
    )

    print(
        f"\nseen={stats.seen} candidates={stats.candidates} "
        f"proposed={stats.proposed} written={stats.written}"
    )
    for reason, count in stats.skipped.most_common():
        print(f"  skip:{reason} {count}")
    print(f"report: {report_path}")
    print_dry_run_footer(args.apply, TE_NOTE if stats.written else None)


if __name__ == "__main__":
    main()
