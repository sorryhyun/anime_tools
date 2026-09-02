"""One ``run_<stage>(req)`` per caption stage — the in-process surface the CLIs
in ``cli/`` are shells over.

Each takes its request from :mod:`anime_tools.stages.requests`, runs the stage's
library function, writes ``report.json`` and prints the epilogue the CLI
always printed. The models are imported inside the call, so importing this
module stays torch-free and a ``from_report`` replay never loads one.

A missing input tree raises :class:`FileNotFoundError`; the CLIs turn it into a
``SystemExit`` with the same message.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from anime_tools._env import curation_home, resolve_path
from anime_tools._json import write_json
from anime_tools._progress import phase
from anime_tools.contract import REPLAY_SHAPES
from anime_tools.stages.cli._args import make_progress
from anime_tools.stages.cli._report import (
    print_dry_run_footer,
    stage_report_header,
    write_stage_report,
)
from anime_tools.stages.requests import (
    AuditRequest,
    AutotagRequest,
    CorrectRequest,
    ExportRequest,
    OcrRequest,
    PositionRequest,
    ResizeRequest,
)

__all__ = [
    "run_audit",
    "run_autotag",
    "run_correct",
    "run_export",
    "run_ocr",
    "run_position",
    "run_resize",
]

RESIZE_FIRST = "run `make preprocess-resize` (or the Resize stage) first"


def _resized(dst: str) -> Path:
    resized_dir = resolve_path(dst)
    if not resized_dir.exists():
        raise FileNotFoundError(
            f"resized dir not found: {resized_dir} — {RESIZE_FIRST}"
        )
    return resized_dir


# ---- autotag -------------------------------------------------------------

AUTOTAG_TE_NOTE = "captions changed — run `make preprocess-te` to re-encode."


def run_autotag(req: AutotagRequest):
    """Tag the resized tree and propose (with ``apply``, write) the revised
    caption. Returns ``(rows, stats)`` from the tagging or replay pass."""
    from anime_tools.stages.autotag import (
        AutotagOptions,
        build_tag_fn,
        run_autotag_captions,
    )
    from anime_tools.stages.replay import run_replay_cli

    resized_dir = _resized(req.dst)
    source_dir = resolve_path(req.src)
    report_dir = resolve_path(req.report_dir)
    if req.from_report:
        # Write a previous dry run's proposals — no tagger, no images opened.
        rows, stats, _ = run_replay_cli(
            req,
            spec=REPLAY_SHAPES["autotag"],
            src=source_dir,
            dst=resized_dir,
            report_dir=report_dir,
            after_write_note=AUTOTAG_TE_NOTE,
        )
        return rows, stats

    tag_fn, info = build_tag_fn(
        req.tagger_dir, device=req.device, min_confidence=req.min_confidence
    )
    rows, stats = run_autotag_captions(
        resized_dir=resized_dir,
        source_dir=source_dir,
        tag_fn=tag_fn,
        options=AutotagOptions(mode=req.mode, min_confidence=req.min_confidence),
        path_pattern=req.path_pattern,
        apply=req.apply,
        progress=make_progress(50, first=True),
    )

    report_path = write_stage_report(
        report_dir,
        {
            "mode": req.mode,
            "min_confidence": req.min_confidence,
            **stage_report_header(
                src=source_dir,
                dst=resized_dir,
                path_pattern=req.path_pattern,
                apply=req.apply,
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
    print_dry_run_footer(req.apply, AUTOTAG_TE_NOTE if stats.written else None)
    return rows, stats


# ---- position clauses ----------------------------------------------------

POSITION_TE_NOTE = (
    "\nWritten to the resized captions (the master is untouched). Run "
    "`make preprocess-te` now to regenerate the variant sidecars and "
    "re-encode."
)


def _run_flatten(req: PositionRequest, src: Path, dst: Path, report_dir: Path):
    """The inverse pass — text only, so it short-circuits before any model load."""
    from anime_tools.stages.position_captions import flatten_captions

    rows, stats = flatten_captions(
        resized_dir=dst, source_dir=src, path_pattern=req.path_pattern, apply=req.apply
    )
    summary = {
        "mode": "flatten",
        "applied": bool(req.apply),
        "seen": stats.seen,
        "with_clauses": stats.candidates,
        "flattened": stats.proposed,
        "written": stats.written,
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
    }
    # Its own name, not ``report.json``: replaying a flatten would write the
    # clauses back.
    write_json(report_dir / "flatten_report.json", {"summary": summary, "images": rows})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_dir / 'flatten_report.json'}")
    print_dry_run_footer(req.apply, POSITION_TE_NOTE)
    return rows, stats


def run_position(req: PositionRequest):
    """Detect → order → crop → tag → compose over the resized tree, or the
    ``flatten`` / ``from_report`` text-only passes. Returns ``(rows, stats)``."""
    from anime_tools.stages.replay import run_replay_cli

    src = resolve_path(req.src)
    dst = resolve_path(req.dst)
    report_dir = resolve_path(req.report_dir)

    if req.flatten:
        return _run_flatten(req, src, dst, report_dir)
    if req.from_report:
        # ``drop_variants`` mirrors the stage's own write: a stale
        # ``{stem}.variants.txt`` outranks ``{stem}.txt`` at encode time.
        rows, stats, _ = run_replay_cli(
            req,
            spec=REPLAY_SHAPES["position"],
            src=src,
            dst=dst,
            report_dir=report_dir,
            after_write_note=POSITION_TE_NOTE,
        )
        return rows, stats

    from anime_tools.stages._models import load_tagger
    from anime_tools.stages.detector import build_detect_fn
    from anime_tools.stages.instance_detection import (
        prompt_embed_sha256,
        resolve_prompt_embed,
    )
    from anime_tools.stages.position_captions import run_position_captions

    # Both stay resident: the pipeline is per-image (detect -> crop -> tag), not
    # two dataset-wide passes.
    detect_fn, part_detect_fn, sam_model, sam_processor = build_detect_fn(
        req.detection, device=req.device
    )
    tagger, vocabulary, _ckpt_dir = load_tagger(req)

    token_count_fn = None
    if req.qwen3:
        from anime_tools.captions.tokenizers import load_qwen3_tokenizer_from_dir

        tokenizer = load_qwen3_tokenizer_from_dir(req.qwen3)

        def token_count_fn(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    options = req.options()
    rows, stats = run_position_captions(
        resized_dir=dst,
        source_dir=src,
        detect_fn=detect_fn,
        part_detect_fn=part_detect_fn,
        tag_fn=tagger.predict,
        vocabulary=vocabulary,
        options=options,
        path_pattern=req.path_pattern,
        apply=req.apply,
        crops_dir=(report_dir / "crops") if req.crops else None,
        token_count_fn=token_count_fn,
        progress=make_progress(200),
    )
    del sam_processor, sam_model

    over_budget = [
        r for r in rows if r.tokens is not None and r.tokens > req.max_tokens
    ]
    embed_path = resolve_prompt_embed(req.detection.prompt_embed)
    summary = {
        **stage_report_header(
            src=src, dst=dst, path_pattern=req.path_pattern, apply=req.apply
        ),
        "rewrite": bool(req.rewrite),
        # A soft prompt is a file: two runs only compare when the sha matches.
        "prompt": req.detection.prompt,
        "prompt_embed": str(embed_path) if embed_path else None,
        "prompt_embed_sha256": prompt_embed_sha256(embed_path),
        "attribution_margin": req.attribution_margin,
        "seen": stats.seen,
        "candidates": stats.candidates,
        "proposed": stats.proposed,
        "written": stats.written,
        # How much of the flat bag the clauses took. Zero under --no_rewrite.
        "rewritten": stats.rewritten,
        "moved_tags": stats.moved_tags,
        "max_novel_tags": req.max_novel_tags,
        "clause_tags": stats.clause_tags,
        "novel_tags": stats.novel_tags,
        "reuse_ratio": (
            round(1.0 - stats.novel_tags / stats.clause_tags, 3)
            if stats.clause_tags
            else None
        ),
        "pinned_tags": dict(sorted(stats.pinned_tags.items(), key=lambda kv: -kv[1])),
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
        "part_prompts": list(options.part_prompts),
        # Images with at least one bound instance from a part prompt.
        "part_recovered": sum(
            1 for r in rows if any(i.source != "subject" for i in r.instances)
        ),
        "max_tokens": max(
            (r.tokens for r in rows if r.tokens is not None), default=None
        ),
        "over_token_budget": [r.image for r in over_budget],
    }
    report_path = write_stage_report(
        report_dir, {"summary": summary, "images": [asdict(r) for r in rows]}
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_path}")
    if over_budget:
        print(
            f"WARNING: {len(over_budget)} caption(s) exceed {req.max_tokens} tokens — "
            "the tail truncates silently at TE-cache time."
        )
    print_dry_run_footer(req.apply, POSITION_TE_NOTE)
    if req.apply and req.rewrite and stats.moved_tags:
        print(
            f"{stats.moved_tags} tag(s) moved out of the flat bag across "
            f"{stats.rewritten} caption(s). To back that out: "
            '`make caption-position ARGS="--flatten --apply"`.'
        )
    return rows, stats


# ---- multiview audit -----------------------------------------------------


def _audit_written_note(src: Path, written: int, holder: str) -> str:
    return (
        f"\n{written} caption(s) written to the master ({src}). Run "
        "`make preprocess-te` now to re-encode. The master is gitignored — "
        f"{holder} holds the before-text if you need to back this out."
    )


def run_audit(req: AuditRequest):
    """Sweep the single-subject images for several views of one girl and (with
    ``apply``) write the missing tag into the caption master. Returns
    ``(rows, stats)``."""
    from anime_tools.stages.replay import run_replay_cli

    src = resolve_path(req.src)
    dst = resolve_path(req.dst)
    report_dir = resolve_path(req.report_dir)
    verdicts, confidences = req.apply_verdicts, req.apply_confidence

    if req.from_report:
        # The writable set is the verdict/confidence gate, not a row ``status``,
        # so ``row_filter`` is closed over the gate here.
        rows, stats, _ = run_replay_cli(
            req,
            spec=replace(
                REPLAY_SHAPES["audit"],
                row_filter=lambda row: (
                    row.get("verdict") in verdicts
                    and row.get("confidence") in confidences
                ),
            ),
            src=src,
            dst=dst,
            report_dir=report_dir,
            notes=[f"gate: verdicts={list(verdicts)} confidences={list(confidences)}"],
            after_write_note=lambda stats: _audit_written_note(
                src, stats.written, "the replayed report"
            ),
        )
        return rows, stats

    from anime_tools.stages._models import load_tagger
    from anime_tools.stages.detector import build_detect_fn
    from anime_tools.stages.multiview_audit import apply_findings, run_multiview_audit

    detect_fn, part_detect_fn, sam_model, sam_processor = build_detect_fn(
        req.detection, device=req.device
    )
    tagger, vocabulary, _ckpt_dir = load_tagger(req)
    options = req.options()

    rows, stats = run_multiview_audit(
        resized_dir=dst,
        source_dir=src,
        detect_fn=detect_fn,
        tag_fn=tagger.predict,
        vocabulary=vocabulary,
        options=options,
        path_pattern=req.path_pattern,
        crops_dir=(report_dir / "crops") if req.crops else None,
        sheets_dir=(report_dir / "sheets") if req.sheets else None,
        progress=make_progress(200),
        part_detect_fn=part_detect_fn,
        multiview_threshold=req.multiview_threshold,
        identity_confidence=req.identity_confidence,
        suggest_counts=req.suggest_counts,
    )
    del sam_processor, sam_model

    written: list[tuple[str, str, str]] = []
    apply_skipped: dict[str, int] = {}
    if req.apply:
        written, skipped = apply_findings(
            rows, source_dir=src, verdicts=verdicts, confidences=confidences
        )
        apply_skipped = dict(skipped.most_common())

    summary = {
        **stage_report_header(
            src=src, dst=dst, path_pattern=req.path_pattern, apply=req.apply
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
        # Why a row was not written: ``drifted`` / ``already-applied`` / gated.
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
    if req.sheets:
        print(f"sheets: {report_dir / 'sheets'} (one PNG per finding, verdict-first)")
    print_dry_run_footer(
        req.apply, _audit_written_note(src, len(written), "report.json")
    )
    return rows, stats


# ---- correction + variants -----------------------------------------------


def run_correct(req: CorrectRequest):
    """Mirror the master into corrected revised captions (plus variant
    sidecars). Returns the :class:`~anime_tools.stages.captions.CaptionStats`."""
    from anime_tools.captions.correction import (
        CaptionCorrectionOptions,
        find_tag_csv,
        load_tag_knowledge_base,
    )
    from anime_tools.captions.tag_drop_groups import parse_drop_groups
    from anime_tools.stages.captions import write_corrected_preprocess_captions

    csv_path = Path(req.tag_csv) if req.tag_csv else find_tag_csv(curation_home())
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            "danbooru_tags_classified.csv not found. Run "
            "`python -m anime_tools.downloads danbooru_tags` first "
            "(or the GUI's Settings > Models > Danbooru tag KB)."
        )

    # The erasure pool (identity-randomize only) needs both tokenizers, loaded
    # tokenizer-only — no encoder weights.
    qwen3_tokenizer = t5_tokenizer = None
    if req.randomizes:
        from anime_tools.captions.tokenizers import (
            load_qwen3_tokenizer_from_dir,
            load_t5_tokenizer_from_dir,
        )

        qwen3_tokenizer = load_qwen3_tokenizer_from_dir(req.qwen3)
        t5_tokenizer = load_t5_tokenizer_from_dir(req.t5_tokenizer_path)

    stats = write_corrected_preprocess_captions(
        Path(req.src),
        Path(req.dst),
        load_tag_knowledge_base(csv_path),
        options=CaptionCorrectionOptions(
            insert_no_artist=req.caption_insert_no_artist,
            trigger_word=req.caption_trigger_word,
            trigger_at_front=req.caption_trigger_at_front,
            drop_groups=parse_drop_groups(req.caption_drop_groups),
        ),
        recursive=req.recursive,
        path_pattern=req.path_pattern or "*",
        correct=not req.no_correct,
        num_variants=req.caption_shuffle_variants,
        tag_dropout_rate=req.caption_tag_dropout_rate,
        tag_randomize_rate=req.caption_tag_randomize_rate,
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
    )
    print(
        "Corrected preprocess captions: "
        f"{stats.written} written, {stats.unchanged} unchanged, "
        f"{stats.missing_source} missing source, {stats.removed_stale} stale removed, "
        f"{stats.variants_written} variant sidecars, "
        f"{stats.clauses_preserved} position clauses kept "
        f"({stats.seen} resized images)"
    )
    return stats


# ---- OCR -----------------------------------------------------------------


def run_ocr(req: OcrRequest):
    """Read the text in every resized image and (with ``apply``) write the
    ``{stem}.ocr.txt`` sidecars. Returns ``(rows, stats)``."""
    resized_dir = _resized(req.dst)
    ocr_dir = resolve_path(req.ocr_dir)
    report_dir = resolve_path(req.report_dir)

    # Deferred: onnxruntime is the heaviest thing this stage touches.
    from anime_tools.ocr import load_ocr, resolve_onnx_device
    from anime_tools.stages.ocr import run_ocr as read_tree

    # Not `_device.resolve_device`: its torch probe would cost this run 1.8x for an
    # answer onnxruntime already has.
    device = resolve_onnx_device(req.device)
    print(f"Loading PP-OCRv6 ({device})...", flush=True)
    with phase("load ocr"):
        engine = load_ocr(
            device=device,
            min_score=req.min_score,
            min_chars=req.min_chars,
            skip_en=req.skip_en,
            join_cjk=req.join_cjk,
            min_box_px=req.min_box_px,
            max_boxes=req.max_boxes,
            limit_side=req.det_limit_side,
            batch_size=req.batch_size,
        )

    rows, stats = read_tree(
        resized_dir=resized_dir,
        ocr_dir=ocr_dir,
        read_fn=engine.read,
        read_iter_fn=engine.read_iter,
        path_pattern=req.path_pattern,
        apply=req.apply,
        progress=make_progress(25, first=True),
    )

    report_path = write_stage_report(
        report_dir,
        {
            "min_score": req.min_score,
            "min_chars": req.min_chars,
            "skip_en": bool(req.skip_en),
            "join_cjk": bool(req.join_cjk),
            "min_box_px": req.min_box_px,
            "max_boxes": req.max_boxes,
            "det_limit_side": req.det_limit_side,
            "applied": bool(req.apply),
            "apply": bool(req.apply),
            "dst": str(resized_dir),
            "ocr_dir": str(ocr_dir),
            "path_pattern": req.path_pattern,
            "stats": {
                "seen": stats.seen,
                "with_text": stats.with_text,
                "lines": stats.lines,
                "sidecars": stats.sidecars,
                "skipped": dict(stats.skipped),
            },
            "rows": [r.to_row() for r in rows],
        },
    )

    print(
        f"\nseen={stats.seen} with_text={stats.with_text} "
        f"lines={stats.lines} sidecars={stats.sidecars}"
    )
    for reason, count in stats.skipped.most_common():
        print(f"  skip:{reason} {count}")
    print(f"report: {report_path}")
    if req.apply:
        print(f"sidecars: {ocr_dir}")
    else:
        print("\nDry run — no sidecars written. Re-run with --apply to write.")
    return rows, stats


# ---- resize --------------------------------------------------------------


def run_resize(req: ResizeRequest):
    """Resize the master into the bucket tree. Always writes. Returns the
    :class:`~anime_tools.stages.resize.ResizeStats`."""
    from anime_tools.stages.resize import ResizeOptions, run_resize_images

    src = resolve_path(req.src)
    dst = resolve_path(req.dst)
    if not src.is_dir():
        raise FileNotFoundError(f"source dir not found: {src}")

    options = ResizeOptions.build(
        target_res=req.target_res,
        crop_anchor=req.resize_crop_anchor,
        crop_margins=req.resize_crop_margins,
        max_ratio=req.freefit_max_ratio,
    )
    stats = run_resize_images(
        src=src,
        dst=dst,
        options=options,
        path_pattern=req.path_pattern or "*",
        recursive=req.recursive,
        min_pixels=req.min_pixels,
        copy_captions=req.copy_captions,
        overwrite=req.overwrite,
        workers=req.workers,
        # Every line: there is no other per-image output.
        progress=make_progress(1),
    )

    write_stage_report(
        resolve_path(req.report_dir),
        {
            "src": str(src),
            "dst": str(dst),
            "path_pattern": req.path_pattern or "*",
            "target_res": list(options.target_res),
            "crop_anchor": options.crop_anchor,
            "crop_margins": list(options.crop_margins),
            "max_ratio": options.max_ratio,
            "min_pixels": req.min_pixels,
            "overwrite": req.overwrite,
            "stats": {
                "seen": stats.seen,
                "written": stats.written,
                "skipped_current": stats.skipped_current,
                "skipped_small": stats.skipped_small,
                "failed": stats.failed,
            },
            "buckets": dict(sorted(stats.buckets.items())),
            "failures": stats.failures,
            "too_small": stats.too_small,
        },
    )

    print(
        f"Resized: {stats.written} written, "
        f"{stats.skipped_current} already current, "
        f"{stats.skipped_small} below {req.min_pixels:,} px, "
        f"{stats.failed} failed ({stats.seen} images seen)"
    )
    for line in stats.failures:
        print(f"  fail: {line}")
    # A skip here means invisible to every later stage, so name each file rather
    # than counting them.
    for line in stats.too_small:
        print(f"  too small: {line}")
    if stats.buckets:
        print("Bucket distribution:")
        for reso, count in sorted(stats.buckets.items()):
            w, h = (int(v) for v in reso.split("x"))
            print(f"  {reso:>10}: {count:>3d} images  ({(w // 16) * (h // 16)} tokens)")
    return stats


# ---- export --------------------------------------------------------------


def run_export(req: ExportRequest):
    """Publish the workspace under ``out``. Returns ``(rows, stats)``."""
    from anime_tools.stages.export_workspace import ExportPaths
    from anime_tools.stages.export_workspace import run_export as publish

    paths = ExportPaths(
        resized=resolve_path(req.dst),
        masks=resolve_path(req.masks),
        master=resolve_path(req.master),
        index=resolve_path(req.index),
        src=resolve_path(req.src),
        out=resolve_path(req.out),
    )
    if not paths.resized.is_dir():
        raise FileNotFoundError(
            f"nothing to export: {paths.resized} does not exist. Run the Resize stage first."
        )
    pattern = req.path_pattern or "*"
    rows, stats = publish(
        paths, path_pattern=pattern, apply=req.apply, progress=make_progress(50)
    )
    path = write_stage_report(
        resolve_path(req.report_dir),
        {
            **stage_report_header(
                src=paths.src, dst=paths.resized, path_pattern=pattern, apply=req.apply
            ),
            "out": str(paths.out),
            "stats": stats.to_dict(),
            "rows": [r.to_dict() for r in rows],
        },
    )
    print(f"\nreport → {path}")
    print_dry_run_footer(
        req.apply, f"published: {stats.created} created, {stats.overwrote} overwritten"
    )
    return rows, stats
