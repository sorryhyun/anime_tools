"""Read the text in each image with PP-OCRv6 and record what it says.

Thin CLI over ``anime_tools.stages.ocr``: walks the resized tree, detects and
recognizes every text line, writes ``{stem}.ocr.txt`` beside the revised caption
and adds the script tag the caption earns. **Dry-run is the default** — a dry run
emits ``report.json`` carrying every line it would have written, so the sidecars
can be eyeballed before they exist.

Two destinations, because a run answers two questions (see
``anime_tools/stages/ocr.py``): the recognized text goes in the sidecar, and the
caption gets at most a tag — ``english text`` / ``chinese text`` /
``bilingual text``, never the string. Japanese earns no tag, Danbooru having none
for it, so a Japanese corpus fills sidecars and proposes nothing; ``--no-tags``
makes that the mode for every language.

The weights have no flag. Both halves are read from the download catalog's
``ppocr_det`` / ``ppocr_rec`` rows, for the reason the MIT stage's ``--ctd-gate``
net has none: a path you could point elsewhere is a Download button aimed at a
directory the loader does not read.

After an ``--apply`` run the TE caches are stale but *look* current — caption
edits do not invalidate them. Follow with ``make preprocess-te``.

``--from_report`` replays a dry run's caption tags without loading a model, skips
any row whose caption changed since, and writes ``apply_report.json``. It does
**not** replay sidecars: those describe the image rather than the caption, so
re-deriving them means re-reading the pixels, which is the run itself.

    python -m anime_tools.stages.cli.ocr_captions
    python -m anime_tools.stages.cli.ocr_captions --apply --lang ja
    python -m anime_tools.stages.cli.ocr_captions --apply --no-tags
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_tools import workspace as WS
from anime_tools._device import add_device_arg, resolve_device
from anime_tools._env import resolve_path
from anime_tools.ocr.script import LANGS, parse_langs
from anime_tools.stages.cli._args import (
    add_apply_args,
    add_dataset_args,
    add_report_dir_arg,
    make_progress,
)
from anime_tools.stages.cli._report import (
    print_dry_run_footer,
    stage_report_header,
    write_stage_report,
)
from anime_tools.stages.ocr import HISTORY_BY, OcrOptions, run_ocr_captions
from anime_tools.stages.replay import ReplaySpec, run_replay_cli

DEFAULT_REPORT_DIR = f"{WS.REPORTS}/ocr"
DEFAULT_LANGS = "en,ja,zh"
"""Every language the shipped recognizer can read — its dictionary has both kana
and 15,565 han characters and nothing else, so this is the whole vocabulary
rather than a selection from it."""

TE_NOTE = "captions changed — run `make preprocess-te` to re-encode."


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p)
    p.add_argument(
        "--lang",
        dest="lang",
        default=DEFAULT_LANGS,
        help=f"Comma-separated allowlist; a line in any other language is "
        f"dropped whole. One of {', '.join(LANGS)} "
        f"(default: {DEFAULT_LANGS})",
    )
    p.add_argument(
        "--tags",
        dest="tags",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Add the script tag the caption earns (english text / chinese "
        "text / bilingual text). --no-tags writes sidecars only and touches "
        "no caption",
    )
    p.add_argument(
        "--min_score",
        "--min-score",
        dest="min_score",
        type=float,
        default=0.6,
        help="Drop a recognized line below this mean per-character confidence "
        "(0-1). The one filter that catches vertical Japanese, which this "
        "pipeline reads sideways and scores low",
    )
    p.add_argument(
        "--min_box_px",
        "--min-box-px",
        dest="min_box_px",
        type=int,
        default=12,
        help="Ignore a detected box whose longest side is under this many "
        "pixels — screentone and hatching, not text",
    )
    p.add_argument(
        "--max_boxes",
        "--max-boxes",
        dest="max_boxes",
        type=int,
        default=64,
        help="Recognize at most this many boxes per image, largest first, so "
        "one misread texture cannot cost a thousand crops",
    )
    p.add_argument(
        "--det_limit_side",
        "--det-limit-side",
        dest="det_limit_side",
        type=int,
        default=960,
        help="Longest side the detector sees; larger finds smaller text and "
        "costs quadratically",
    )
    p.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=8,
        help="Line crops recognized per forward pass",
    )
    add_apply_args(
        p,
        apply_help="Write the sidecars and the caption tags (default: dry run)",
        from_report_help="Replay a previous dry run's caption tags instead of "
        "re-reading the images: loads no model and writes no sidecars. Skips "
        "any row whose caption changed since. Emits apply_report.json (never "
        "clobbers the report it reads)",
    )
    add_report_dir_arg(p, DEFAULT_REPORT_DIR)
    add_device_arg(p)
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


# The tag lands in the **revised** caption (``--dst``), like the clause rewrite
# and unlike autotag: this stage never creates a caption master.
REPLAY_SPEC = ReplaySpec(
    stage="ocr_captions",
    rows_key="rows",
    stats_key="stats",
    ok_status="ok",
    before_field="existing",
    after_field="proposed",
    target_root="dst",
    drop_variants=True,
    history_by=HISTORY_BY,
)


def _replay(args, *, src: Path, dst: Path, report_dir: Path) -> None:
    """Write a previous dry run's caption tags — no model, no images opened."""
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

    try:
        langs = parse_langs(args.lang)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Not in parse_args(): --from_report returns above without importing
    # onnxruntime, and both of these would have loaded it.
    from anime_tools.ocr import OcrWeightsMissing, load_ocr

    device = resolve_device(args.device)
    print(f"Loading PP-OCRv6 ({device})...", flush=True)
    try:
        engine = load_ocr(
            device=device,
            min_score=args.min_score,
            min_box_px=args.min_box_px,
            max_boxes=args.max_boxes,
            limit_side=args.det_limit_side,
            batch_size=args.batch_size,
        )
    except OcrWeightsMissing as exc:
        raise SystemExit(str(exc)) from exc

    options = OcrOptions(langs=langs, tags=args.tags)
    rows, stats = run_ocr_captions(
        resized_dir=resized_dir,
        source_dir=source_dir,
        read_fn=engine.read,
        options=options,
        path_pattern=args.path_pattern,
        apply=args.apply,
        progress=make_progress(25, first=True),
    )

    report_path = write_stage_report(
        report_dir,
        {
            "lang": list(langs),
            "tags": bool(args.tags),
            "min_score": args.min_score,
            "min_box_px": args.min_box_px,
            "max_boxes": args.max_boxes,
            "det_limit_side": args.det_limit_side,
            **stage_report_header(
                src=source_dir,
                dst=resized_dir,
                path_pattern=args.path_pattern,
                apply=args.apply,
            ),
            "stats": {
                "seen": stats.seen,
                "candidates": stats.candidates,
                "with_text": stats.with_text,
                "lines": stats.lines,
                "proposed": stats.proposed,
                "written": stats.written,
                "sidecars": stats.sidecars,
                "langs": dict(stats.langs),
                "skipped": dict(stats.skipped),
            },
            "rows": [r.to_row() for r in rows],
        },
    )

    print(
        f"\nseen={stats.seen} candidates={stats.candidates} "
        f"with_text={stats.with_text} lines={stats.lines} "
        f"proposed={stats.proposed} written={stats.written}"
    )
    for lang, count in stats.langs.most_common():
        print(f"  lang:{lang} {count}")
    for reason, count in stats.skipped.most_common():
        print(f"  skip:{reason} {count}")
    print(f"report: {report_path}")
    print_dry_run_footer(args.apply, TE_NOTE if stats.written else None)


if __name__ == "__main__":
    main()
