#!/usr/bin/env python3
"""Rewrite multi-subject captions into position clauses (SAM3 + Anima Tagger).

Thin CLI over ``anime_tools.stages.position_captions``: loads SAM3 and the Anima
Tagger, drives the detect -> order -> crop+blank -> tag -> compose pipeline over
the resized dataset, and writes a review report.

An attributable tag is MOVED out of the flat bag into its clause, so each
attribute is asserted exactly once and bound to its subject (v2). ``--no_rewrite``
restores the additive v1 behaviour for a training A/B; ``--flatten`` is the
inverse pass that merges clauses back into the bag.

The caption master (``image_dataset/``) is never written — clauses land on the
derived caption next to the resized image (``--dst``, i.e.
``post_image_dataset/resized/<rel>.txt``), the same file the TE step encodes.

Dry-run is the default: nothing is written until ``--apply`` is passed. An
``--apply`` run bumps the caption's mtime (so TE caches go correctly stale) and
drops stale ``.variants.txt`` sidecars — follow it with ``make preprocess-te``
to actually re-encode.

``--from_report <report.json>`` replays a dry run instead of redoing the whole
detect → crop → tag pass: the report already holds the destination caption and
the exact proposed text, so the apply writes them **without loading SAM3 or the
tagger**. A caption edited since the dry run is skipped and counted, never
overwritten; the replay writes ``apply_report.json`` so it cannot clobber the
report it read.

    make caption-position                      # dry run over the whole dataset
    make caption-position ARGS="--apply"       # write the clauses
    make preprocess-te                         # re-encode (required after apply)
    make caption-position ARGS="--flatten --apply"   # back the rewrite out

    # the same two-step, paying for SAM3 + the tagger once:
    python -m anime_tools.stages.cli.position_captions
    python -m anime_tools.stages.cli.position_captions --apply \
        --from_report post_image_dataset/captions/position/report.json
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
from anime_tools.stages.instance_detection import (
    DEFAULT_SUBJECT_PROMPT_EMBED,
    load_soft_prompt,
    prompt_embed_sha256,
    resolve_prompt_embed,
)
from anime_tools.stages.position_captions import (
    Detection,
    PositionCaptionOptions,
    flatten_captions,
    load_clause_vocabulary,
    run_position_captions,
)
from anime_tools.stages.replay import (
    ReplaySpec,
    StaleReportError,
    print_replay,
    run_replay,
)
from anime_tools.tagger.dbv4_meta import DEFAULT_TAGGER_DIR

DEFAULT_REPORT_DIR = "post_image_dataset/captions/position"
# Both tokenizers pad to this (``--qwen3_max_token_length`` / ``--t5_…``); a
# caption past it is silently truncated, and the padding invariant means the
# tail simply never reaches the model.
DEFAULT_MAX_TOKENS = 512


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default="image_dataset",
        help="Caption master dir (read-only fallback — never written)",
    )
    p.add_argument(
        "--dst",
        default="post_image_dataset/resized",
        help="Resized images — and where the rewritten caption is written",
    )
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
        help="Write the proposed clauses into the resized captions (default: dry run)",
    )
    p.add_argument(
        "--from_report",
        "--from-report",
        dest="from_report",
        default=None,
        help="Replay a previous dry run's report.json instead of re-running "
        "SAM3 + the tagger: writes exactly the captions it proposed and loads "
        "no model. Skips any caption that changed since. Emits "
        "apply_report.json (never clobbers the report it reads)",
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
        help="Also export the mask-blanked crops next to the report (review aid)",
    )
    p.add_argument(
        "--flatten",
        action="store_true",
        help="Inverse pass: merge every caption's clauses back into its flat bag "
        "and drop the clauses. Text-only (no SAM3, no tagger) — this is how an "
        "--apply run is backed out, and how the clause-free control corpus for a "
        "training A/B is built. Flattens hand-written clauses too.",
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
        help="learned soft prompt (.safetensors) used in place of --prompt for "
        "the subject pass; part prompts stay textual. Default = the shipped "
        f"{DEFAULT_SUBJECT_PROMPT_EMBED}; pass `none` for the plain text prompt",
    )
    g.add_argument("--score_threshold", type=float, default=0.5)
    g.add_argument(
        "--retry_score_threshold",
        type=float,
        default=0.35,
        help="Retry threshold when detection undershoots the expected count. "
        "This is SAM3's own confidence floor, not a post-filter — see "
        "build_detect_fn",
    )
    g.add_argument(
        "--part_prompts",
        "--part-prompts",
        dest="part_prompts",
        default="",
        help="Comma-separated body-part prompts, tried only when the subject "
        "prompt undershoots — recovers headless close-up panels (a hip / "
        "backside crop next to one full body) that 'girl' cannot see at any "
        'threshold. Off by default; try "buttocks,hips,thighs"',
    )
    g.add_argument(
        "--part_score_threshold",
        "--part-score-threshold",
        dest="part_score_threshold",
        type=float,
        default=0.5,
        help="Confidence floor for a body-part box (kept separate from the "
        "subject threshold — part prompts are the looser concept)",
    )
    g.add_argument(
        "--part_containment_threshold",
        "--part-containment-threshold",
        dest="part_containment_threshold",
        type=float,
        default=0.7,
        help="Drop a part box this nested inside an already-kept box. Unlike "
        "--containment_threshold this is safe to leave on: a part inside a "
        "subject is that subject's own body, never a second subject",
    )
    g.add_argument("--iou_threshold", type=float, default=0.65)
    g.add_argument(
        "--containment_threshold",
        "--containment-threshold",
        dest="containment_threshold",
        type=float,
        default=1.01,
        help="Suppress a box this nested inside a kept one (intersection over "
        "the smaller box). Off by default (>1.0 disables): a real second "
        "subject is as nested as a group box — enabling it cost 32 real "
        "subjects to save 12 group boxes",
    )
    g.add_argument(
        "--mask_containment_threshold",
        "--mask-containment-threshold",
        dest="mask_containment_threshold",
        type=float,
        default=0.8,
        help="Suppress a detection whose MASK is this nested inside a kept "
        "one. On by default, unlike its box counterpart: a second girl in "
        "front of the first nests identically by box but her mask is disjoint. "
        ">1.0 disables (the pre-2026-08-19 behaviour)",
    )
    g.add_argument(
        "--dedupe_fill_ratio",
        "--dedupe-fill-ratio",
        dest="dedupe_fill_ratio",
        type=float,
        default=2.0,
        help="Mask-quality tie-break inside an NMS-matched pair; 0 = off "
        "(score-only survivor). See docs/experimental/multiview_audit.md §5.",
    )
    g.add_argument(
        "--min_area_frac",
        "--min-area-frac",
        dest="min_area_frac",
        type=float,
        default=0.005,
        help="Drop detections smaller than this fraction of the image — an "
        "inset (a character on a phone screen) is not a bindable subject",
    )
    g.add_argument("--pad", type=float, default=0.06, help="bbox padding fraction")
    g.add_argument(
        "--no_blank_crops",
        "--no-blank-crops",
        dest="blank_crops",
        action="store_false",
        help="Skip mask-blanking (probe B: this is what caused the hair-color misses)",
    )
    g.add_argument(
        "--row_tol",
        type=float,
        default=0.25,
        help="Minimum fractional overlap (of the narrower box extent) for two "
        "subjects to share a row — and a column, on magazine layouts where a "
        "full-height subject bridges a stack of panels",
    )
    g.add_argument("--min_instances", type=int, default=2)
    g.add_argument("--max_instances", type=int, default=8)
    g.add_argument(
        "--no_strict_count",
        "--no-strict-count",
        dest="strict_count",
        action="store_false",
        help="Propose clauses even when detection disagrees with the girls-count",
    )

    c = p.add_argument_group("clause composition")
    c.add_argument("--max_clause_tags", type=int, default=8)
    c.add_argument(
        "--max_novel_tags",
        "--max-novel-tags",
        dest="max_novel_tags",
        type=int,
        default=1,
        help="How many tags a clause may introduce that the caption never "
        "contained. The rest of the clause is filled from the flat bag first. "
        "Only a bag tag can MOVE — a novel one is a pure addition the curated "
        "caption never made. 0 = never invent, --max_clause_tags = the old "
        "bag-blind behaviour (46%% novel; on ama_mitsuki, 1 vs 8 cut novel tags "
        "515 to 115 and the caption 40%% shorter, with the moved set unchanged)",
    )
    c.add_argument(
        "--name_confidence",
        type=float,
        default=0.5,
        help="Confidence floor for putting a character name in a clause",
    )
    c.add_argument(
        "--allow_unlisted_names",
        "--allow-unlisted-names",
        dest="allow_unlisted_names",
        action="store_true",
        help="Allow a clause name the flat caption never mentions (off: probe B "
        "scored names 4/7, so an unlisted one is most likely a crop artifact)",
    )
    c.add_argument(
        "--keep_shared_tags",
        "--keep-shared-tags",
        dest="discriminative_only",
        action="store_false",
        help="Keep tags every crop agrees on in every clause. Off by default: on "
        "a multiple-views sheet all views share the character, hair and eyes, so "
        "repeating them binds nothing and crowds out the outfit that differs "
        "(they stay in the flat bag either way — v1 never removes anything).",
    )
    c.add_argument(
        "--ungated_identity",
        "--ungated-identity",
        dest="bag_gated_identity",
        action="store_false",
        help="Let a clause carry a hair/eye color the flat caption never listed. "
        "Gated by default: the caption is the curated ground truth, the crop "
        "tagger guesses one for every crop including headless ones, and "
        "discriminative-only then promotes the guess precisely because it "
        "disagrees — 520 of 1600 identity clause tags in the first full-corpus "
        "dry run contradicted the caption",
    )
    c.add_argument(
        "--bind_view_traits",
        "--bind-view-traits",
        dest="multi_view_gate",
        action="store_false",
        help="On a repeated-subject layout (`multiple views`, comic panels), "
        "let a clause carry the character's name and traits (hair, eyes, body, "
        "anatomy). Gated by default: every view or panel is the SAME girl, so "
        "those belong to her, not to a view — 45%% of the multiple-views clause "
        "tags in the first full-corpus dry run were view-invariant, and the ones "
        "that survived shared-tag suppression did so precisely because a crop "
        "disagreed",
    )
    c.add_argument(
        "--gate_view_anatomy",
        "--gate-view-anatomy",
        dest="bind_view_anatomy",
        action="store_false",
        help="On a repeated-subject layout, keep anatomy (`ass`, `thighs`, "
        "`body_parts`) out of every clause — the pre-2026-08-19 behaviour, when "
        "`body_parts` sat in the view-invariant set. Bound by default: unlike "
        "hair color, what anatomy is *visible* is a fact about the panel, so on "
        "a sheet of one girl from the front and the same girl from behind it is "
        "the tag that separates them",
    )
    c.add_argument(
        "--no_framing",
        "--no-framing",
        dest="bind_framing",
        action="store_false",
        help="Keep `framing` out of every clause (the pre-2026-08-19 behaviour). "
        "On by default: it is the only group that says a view is a headless "
        "close-up rather than a whole figure, which on a `multiple views` sheet "
        "of one full body plus a hip/backside panel is the single thing that "
        "tells the clauses apart. Off restores the A side for an A/B.",
    )
    c.add_argument(
        "--no_rewrite",
        "--no-rewrite",
        dest="rewrite",
        action="store_false",
        help="Additive v1: append the clauses but leave the flat bag untouched, "
        "so every bound attribute is asserted twice. Default is the v2 rewrite, "
        "which moves an attributable tag out of the bag into its clause. Kept for "
        "the training A/B arm",
    )
    c.add_argument(
        "--bag_relax",
        "--bag-relax",
        dest="bag_relax",
        type=float,
        default=0.35,
        help="Multiplier on the tagger's per-tag keep threshold for tags the "
        "flat bag already contains (they can only MOVE into a clause, never be "
        "invented, so the curated caption corroborates them — the crop only "
        "attributes). 1.0 = off, the pre-2026-08-19 behaviour. Applied to "
        "every crop before the attributable/shared census, so a rival crop's "
        "borderline score also blocks a move the strict kept sets would have "
        "granted. Motivating case: 5828184's `black panties` scored 0.498 "
        "against a 0.800 threshold on the lying crop and stayed unbound; the "
        "0.35 default is what recovers pose tags off mask-blanked crops",
    )
    c.add_argument(
        "--bag_word_relax",
        "--bag-word-relax",
        dest="bag_word_relax",
        type=float,
        default=0.85,
        help="Extra threshold multiplier per word beyond the first, compounding "
        "with --bag_relax (`black panties` is more specific than `panties`, so "
        "a sub-threshold hit on it is less likely noise). 1.0 = off",
    )
    c.add_argument(
        "--bag_relax_min_score",
        "--bag-relax-min-score",
        dest="bag_relax_min_score",
        type=float,
        default=0.3,
        help="Absolute score floor under the bag relaxation: a relaxed "
        "admission still needs at least this raw probability, however low "
        "bag_relax × bag_word_relax drags the per-tag threshold. Blocks "
        "near-noise fires (measured: `white gloves` bound to a crop with no "
        "hands in frame at a ~0.16 relaxed floor) while keeping the genuine "
        "recoveries (`black panties` at 0.498). Only the relax path is "
        "floored. 0.0 = off, the pre-floor behaviour",
    )
    c.add_argument(
        "--attribution_margin",
        "--attribution-margin",
        dest="attribution_margin",
        type=float,
        default=0.25,
        help="How far the winning crop's probability must clear every other "
        "crop's, RELATIVE to its own (1 - rival/winner), before a tag may LEAVE "
        "the flat bag (the clause carries it either way). Applies on top of the "
        "hard rule that no other crop kept the tag; 0.0 trusts the tagger's "
        "per-tag thresholds alone. Guards the one thing v2 can get wrong that "
        "v1 cannot: removing an attribute the other subjects also have",
    )
    c.add_argument(
        "--qwen3",
        default=None,
        help="Qwen3 tokenizer directory — enables the token-budget column in the report",
    )
    c.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def build_options_from_args(args: argparse.Namespace) -> PositionCaptionOptions:
    """Parsed CLI -> the options one pass runs under.

    Split out of ``main`` so a second entry point (``ab_position_captions.py``,
    which builds two of these from two flag sets) reuses the shipping
    construction instead of a copy that would silently drift the moment a knob
    is added.
    """
    return PositionCaptionOptions(
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
        blank_crops=args.blank_crops,
        row_tol=args.row_tol,
        max_clause_tags=args.max_clause_tags,
        max_novel_tags=args.max_novel_tags,
        name_confidence=args.name_confidence,
        allow_unlisted_names=args.allow_unlisted_names,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        strict_count=args.strict_count,
        discriminative_only=args.discriminative_only,
        bag_gated_identity=args.bag_gated_identity,
        multi_view_gate=args.multi_view_gate,
        bind_framing=args.bind_framing,
        bind_view_anatomy=args.bind_view_anatomy,
        rewrite=args.rewrite,
        attribution_margin=args.attribution_margin,
        bag_relax=args.bag_relax,
        bag_relax_min_score=args.bag_relax_min_score,
        bag_word_relax=args.bag_word_relax,
    )


def build_detect_fn(args: argparse.Namespace, *, model=None, processor=None):
    """SAM3 text-prompt detector returning per-instance boxes + masks.

    Pass ``model``/``processor`` from a previous call to build a second
    detector (different ``--prompt`` / ``--prompt_embed``) on the same loaded
    SAM3 — the A/B script uses this for a detector-side A/B.

    GOTCHA 1: ``Sam3Processor`` carries its own ``confidence_threshold`` and
    applies it before the caller ever sees the boxes, so filtering the result
    against a *lower* retry threshold is a no-op unless the processor itself is
    built at the lowest threshold we might ask for (which it is here — the
    score gate is then applied on top, in ``detect``/``part_detect``).

    GOTCHA 2: ``detect_subjects`` calls back into this per retry and per
    body-part prompt on the same image — encoding and raw detections are
    memoised per image/prompt so a retry is a pure re-filter and a part prompt
    costs only one grounding pass.

    Returns ``(detect, part_detect, model, processor)``. ``part_detect`` takes
    the prompt as an argument; ``detect`` is pinned to ``args.prompt``.
    """
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    floor = min(
        args.score_threshold, args.retry_score_threshold, args.part_score_threshold
    )
    device = resolve_device(args.device)
    if model is None:
        print("Loading SAM3...", flush=True)
        model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            checkpoint_path=str(resolve_path(args.checkpoint)),
            load_from_HF=False,
        )
    if processor is None or processor.confidence_threshold > floor:
        processor = Sam3Processor(model, confidence_threshold=floor)
    soft_prompt = None
    embed_path = resolve_prompt_embed(getattr(args, "prompt_embed", None))
    if embed_path is not None:
        soft_prompt = load_soft_prompt(embed_path, device)
        print(f"soft prompt: {embed_path} (replaces {args.prompt!r})", flush=True)
    cache: dict[str, object] = {"key": None, "state": None, "dets": {}}

    def _ground(image, prompt: str) -> list[Detection]:
        """Raw detections for one prompt, reusing this image's encoded state."""
        if cache["key"] is not image:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cache["state"] = processor.set_image(image)
            cache["key"] = image
            cache["dets"] = {}
        memo: dict = cache["dets"]  # type: ignore[assignment]
        if prompt in memo:
            return memo[prompt]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if soft_prompt is not None and prompt == args.prompt:
                # Learned prompt tensor stands in for the subject phrase; the
                # processor's text encode is skipped and its grounding pass
                # reused as-is (bench/sam3_soft_prompt/).
                state = cache["state"]
                state["backbone_out"].update(soft_prompt)
                state.setdefault("geometric_prompt", model._get_dummy_prompt())
                out = processor._forward_grounding(state)
            else:
                out = processor.set_text_prompt(prompt=prompt, state=cache["state"])
        masks = out.get("masks")
        source = "subject" if prompt == args.prompt else prompt
        dets: list[Detection] = []
        for i, (box, score) in enumerate(zip(out["boxes"], out["scores"])):
            coords = box.tolist() if torch.is_tensor(box) else list(box)
            mask = None
            if masks is not None and i < len(masks):
                m = masks[i]
                mask = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
            dets.append(
                Detection(
                    box=tuple(float(v) for v in coords),
                    score=float(score),
                    mask=mask,
                    source=source,
                )
            )
        memo[prompt] = dets
        return dets

    def detect(image, score_threshold: float) -> list[Detection]:
        return [d for d in _ground(image, args.prompt) if d.score >= score_threshold]

    def part_detect(image, prompt: str, score_threshold: float) -> list[Detection]:
        return [d for d in _ground(image, prompt) if d.score >= score_threshold]

    return detect, part_detect, model, processor


def _run_flatten(args, src: Path, dst: Path, report_dir: Path) -> None:
    """The inverse pass — text only, so it short-circuits before any model load."""
    rows, stats = flatten_captions(
        resized_dir=dst,
        source_dir=src,
        path_pattern=args.path_pattern,
        apply=args.apply,
    )
    summary = {
        "mode": "flatten",
        "applied": bool(args.apply),
        "seen": stats.seen,
        "with_clauses": stats.candidates,
        "flattened": stats.proposed,
        "written": stats.written,
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "flatten_report.json").write_text(
        json.dumps({"summary": summary, "images": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_dir / 'flatten_report.json'}")
    if args.apply:
        print(
            "\nWritten to the resized captions (the master is untouched). Run "
            "`make preprocess-te` now to regenerate the variant sidecars and "
            "re-encode."
        )
    else:
        print("\nDry run — no captions written. Re-run with --apply to write.")


# How ``replay`` reads a position report: ``images``/``summary`` containers,
# ``proposed`` is the writable status, and the rewrite lands on the **derived**
# caption under ``--dst`` (the master is never written). ``drop_variants``
# mirrors ``_write_derived_caption``: a stale ``{stem}.variants.txt`` outranks
# ``{stem}.txt`` at encode time, so the replay must unlink it too.
REPLAY_SPEC = ReplaySpec(
    stage="position_captions",
    rows_key="images",
    stats_key="summary",
    ok_status="proposed",
    before_field="original",
    after_field="proposed",
    target_root="dst",
    drop_variants=True,
)


def _run_replay(args, src: Path, dst: Path, report_dir: Path) -> None:
    """Write a previous dry run's clauses — no SAM3, no tagger, no pixels."""
    try:
        rows, stats, out_path = run_replay(
            spec=REPLAY_SPEC,
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
    print_replay(rows, stats, apply=args.apply)
    print(f"\nreport: {out_path}")
    if args.apply and stats.written:
        print(
            "\nWritten to the resized captions (the master is untouched). Run "
            "`make preprocess-te` now to regenerate the variant sidecars and "
            "re-encode."
        )


def main() -> None:
    args = parse_args()
    src = resolve_path(args.src)
    dst = resolve_path(args.dst)
    report_dir = resolve_path(args.report_dir)

    if args.flatten:
        if args.from_report:
            raise SystemExit(
                "--flatten and --from_report are mutually exclusive: the "
                "flatten pass is already text-only, so there is no model pass "
                "to skip."
            )
        _run_flatten(args, src, dst, report_dir)
        return

    if args.from_report:
        _run_replay(args, src, dst, report_dir)
        return

    # SAM3 first, tagger second: both stay resident since the pipeline is
    # per-image (detect -> crop -> tag), not two dataset-wide passes.
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

    token_count_fn = None
    if args.qwen3:
        from anime_tools.captions.tokenizers import load_qwen3_tokenizer_from_dir

        tokenizer = load_qwen3_tokenizer_from_dir(args.qwen3)

        def token_count_fn(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    options = build_options_from_args(args)

    def progress(index: int, total: int, rel: str) -> None:
        if index % 200 == 0 or index == total:
            print(f"  [{index}/{total}] {rel}", flush=True)

    rows, stats = run_position_captions(
        resized_dir=dst,
        source_dir=src,
        detect_fn=detect_fn,
        part_detect_fn=part_detect_fn,
        tag_fn=tagger.predict,
        vocabulary=vocabulary,
        options=options,
        path_pattern=args.path_pattern,
        apply=args.apply,
        crops_dir=(report_dir / "crops") if args.crops else None,
        token_count_fn=token_count_fn,
        progress=progress,
    )
    del sam_processor, sam_model

    over_budget = [
        r for r in rows if r.tokens is not None and r.tokens > args.max_tokens
    ]
    embed_path = resolve_prompt_embed(args.prompt_embed)
    summary = {
        "applied": bool(args.apply),
        "rewrite": bool(args.rewrite),
        # Recorded so ``--from_report`` can refuse to replay this report against
        # a different pair of trees (the row paths are relative to these).
        "src": str(src),
        "dst": str(dst),
        "path_pattern": args.path_pattern,
        # Which detector produced these boxes — a soft prompt is a file, so
        # two runs are only comparable when the sha matches.
        "prompt": args.prompt,
        "prompt_embed": str(embed_path) if embed_path else None,
        "prompt_embed_sha256": prompt_embed_sha256(embed_path),
        "attribution_margin": args.attribution_margin,
        "seen": stats.seen,
        "candidates": stats.candidates,
        "proposed": stats.proposed,
        "written": stats.written,
        # v2: how much of the flat bag the clauses actually took, and which of
        # the two safety rules pinned the rest. Zero under --no_rewrite.
        "rewritten": stats.rewritten,
        "moved_tags": stats.moved_tags,
        # Clause composition. ``reuse_ratio`` is the headline for the
        # move-don't-invent rule: a bag tag is a candidate move, a novel one can
        # only ever be an addition.
        "max_novel_tags": args.max_novel_tags,
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
        # Images the body-part fallback actually rescued: at least one bound
        # instance whose box came from a part prompt. Zero with the feature off.
        "part_recovered": sum(
            1 for r in rows if any(i.source != "subject" for i in r.instances)
        ),
        "max_tokens": max(
            (r.tokens for r in rows if r.tokens is not None), default=None
        ),
        "over_token_budget": [r.image for r in over_budget],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(
        json.dumps(
            {"summary": summary, "images": [asdict(r) for r in rows]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_dir / 'report.json'}")
    if over_budget:
        print(
            f"WARNING: {len(over_budget)} caption(s) exceed {args.max_tokens} tokens — "
            "the tail truncates silently at TE-cache time."
        )
    if args.apply:
        print(
            "\nWritten to the resized captions (the master is untouched). Run "
            "`make preprocess-te` now to regenerate the variant sidecars and "
            "re-encode."
        )
        if args.rewrite and stats.moved_tags:
            print(
                f"{stats.moved_tags} tag(s) moved out of the flat bag across "
                f"{stats.rewritten} caption(s). To back that out: "
                '`make caption-position ARGS="--flatten --apply"`.'
            )
    else:
        print("\nDry run — no captions written. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
