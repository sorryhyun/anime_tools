"""Rewrite multi-subject captions into position clauses (SAM3 + Anima Tagger).

Thin CLI over ``anime_tools.stages.position_captions``: detect -> order ->
crop+blank -> tag -> compose over the resized dataset, plus a review report.
An attributable tag is MOVED out of the flat bag into its clause (v2);
``--no_rewrite`` keeps the additive v1 behaviour and ``--flatten`` is the inverse
pass. See ``docs/position_captions.md``.

Clauses land on the revised caption under ``--dst``; the caption master is never
written. Dry-run is the default — ``--apply`` writes, bumps the caption's mtime
and drops stale ``.variants.txt`` sidecars, so follow it with a TE re-encode.
``--from_report`` replays a dry run's report without loading SAM3 or the tagger,
skips any caption edited since, and writes ``apply_report.json``.

    make caption-position                      # dry run over the whole dataset
    make caption-position ARGS="--apply"       # write the clauses
    make preprocess-te                         # re-encode (required after apply)
    make caption-position ARGS="--flatten --apply"   # back the rewrite out

    # the same two-step, paying for SAM3 + the tagger once:
    python -m anime_tools.stages.cli.position_captions
    python -m anime_tools.stages.cli.position_captions --apply \
        --from_report workspace/captions/position/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from anime_tools import workspace as WS
from anime_tools._device import resolve_device
from anime_tools._env import resolve_path
from anime_tools._json import write_json

# Importing _sam3 also installs the `np.bool` alias sam3 needs before it loads.
from anime_tools.masking._sam3 import (
    add_checkpoint_arg,
    ground_with_soft_prompt,
    load_sam3,
    make_processor,
)
from anime_tools.stages.cli._args import (
    add_apply_args,
    add_dataset_args,
    add_model_args,
    add_report_dir_arg,
    make_progress,
)
from anime_tools.stages.cli._detection import (
    add_detection_args,
    detection_options,
)
from anime_tools.stages.cli._models import load_tagger
from anime_tools.stages.cli._report import (
    print_dry_run_footer,
    stage_report_header,
    write_stage_report,
)
from anime_tools.stages.instance_detection import (
    load_soft_prompt,
    prompt_embed_sha256,
    resolve_prompt_embed,
)
from anime_tools.stages.position_captions import (
    Detection,
    PositionCaptionOptions,
    flatten_captions,
    run_position_captions,
)
from anime_tools.stages.replay import ReplaySpec, run_replay_cli

DEFAULT_REPORT_DIR = f"{WS.REPORTS}/position"
TE_NOTE = (
    "\nWritten to the resized captions (the master is untouched). Run "
    "`make preprocess-te` now to regenerate the variant sidecars and "
    "re-encode."
)
# Both tokenizers pad to this; a caption past it truncates silently, so the tail
# never reaches the model.
DEFAULT_MAX_TOKENS = 512


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(
        p,
        src_help="Caption master dir (read-only fallback — never written)",
        dst_help="Resized images — and where the rewritten caption is written",
    )
    add_apply_args(
        p,
        apply_help="Write the proposed clauses into the resized captions "
        "(default: dry run)",
        from_report_help="Replay a previous dry run's report.json instead of "
        "re-running SAM3 + the tagger: writes exactly the captions it proposed "
        "and loads no model. Skips any caption that changed since. Emits "
        "apply_report.json (never clobbers the report it reads)",
    )
    add_report_dir_arg(p, DEFAULT_REPORT_DIR)
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
    add_checkpoint_arg(p)
    add_model_args(p)
    add_detection_args(
        p,
        part_prompts_help="Comma-separated body-part prompts, tried only when "
        "the subject prompt undershoots — recovers headless close-up panels (a "
        "hip / backside crop next to one full body) that 'girl' cannot see at "
        'any threshold. Off by default; try "buttocks,hips,thighs"',
        blank_crops=True,
        min_instances=True,
        strict_count=True,
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

    Shared with ``ab_position_captions.py`` so a new knob cannot reach one entry
    point and not the other.
    """
    return PositionCaptionOptions(
        **detection_options(args),
        max_clause_tags=args.max_clause_tags,
        max_novel_tags=args.max_novel_tags,
        name_confidence=args.name_confidence,
        allow_unlisted_names=args.allow_unlisted_names,
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


def options_from_flag_string(
    flags: str,
) -> tuple[PositionCaptionOptions, argparse.Namespace]:
    """Parse a flag *string* through this CLI's own parser.

    The A/B and review sheets pin themselves to :func:`parse_args` rather than a
    second parser, hence the ``sys.argv`` swap. Returns ``(options, args)`` — the
    namespace too, since the detector is built from it.
    """
    argv = sys.argv
    sys.argv = [argv[0], *flags.split()]
    try:
        args = parse_args()
    finally:
        sys.argv = argv
    return build_options_from_args(args), args


def build_detect_fn(args: argparse.Namespace, *, model=None, processor=None):
    """SAM3 text-prompt detector returning per-instance boxes + masks.

    Pass ``model``/``processor`` from a previous call to build a second detector
    (different prompt) on the same loaded SAM3.

    GOTCHA 1: ``Sam3Processor`` applies its own ``confidence_threshold`` before
    the caller sees the boxes, so it must be built at the *lowest* threshold any
    retry might ask for; the score gate is applied on top in ``detect``.

    GOTCHA 2: ``detect_subjects`` calls back per retry and per part prompt on the
    same image, so encoding and raw detections are memoised per image/prompt.

    Returns ``(detect, part_detect, model, processor)``; ``detect`` is pinned to
    ``args.prompt``.
    """
    import torch

    floor = min(
        args.score_threshold, args.retry_score_threshold, args.part_score_threshold
    )
    device = resolve_device(args.device)
    if model is None:
        print("Loading SAM3...", flush=True)
        model, fresh = load_sam3(
            resolve_path(args.checkpoint), device, confidence_threshold=floor
        )
        if processor is None:
            processor = fresh
    if processor is None or processor.confidence_threshold > floor:
        processor = make_processor(model, floor)
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
                # Learned prompt tensor stands in for the subject phrase: the
                # text encode is skipped, the grounding pass reused as-is.
                out = ground_with_soft_prompt(
                    processor, model, cache["state"], soft_prompt
                )
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
    # Its own name, not ``report.json``: replaying a flatten would write the
    # clauses straight back.
    write_json(report_dir / "flatten_report.json", {"summary": summary, "images": rows})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_dir / 'flatten_report.json'}")
    print_dry_run_footer(args.apply, TE_NOTE)


# ``drop_variants`` mirrors the stage's own write: a stale
# ``{stem}.variants.txt`` outranks ``{stem}.txt`` at encode time.
# ``history_by`` mirrors it too: a replay files the version it replaces under
# the same name the live pass would have.
REPLAY_SPEC = ReplaySpec(
    stage="position_captions",
    rows_key="images",
    stats_key="summary",
    ok_status="proposed",
    before_field="original",
    after_field="proposed",
    target_root="dst",
    drop_variants=True,
    history_by="position",
)


def _run_replay(args, src: Path, dst: Path, report_dir: Path) -> None:
    """Write a previous dry run's clauses — no SAM3, no tagger, no pixels."""
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
    # Not in parse_args(): the --from_report replay returns above and must stay
    # torch-free.
    tagger, vocabulary, _ckpt_dir = load_tagger(args)

    token_count_fn = None
    if args.qwen3:
        from anime_tools.captions.tokenizers import load_qwen3_tokenizer_from_dir

        tokenizer = load_qwen3_tokenizer_from_dir(args.qwen3)

        def token_count_fn(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    options = build_options_from_args(args)

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
        progress=make_progress(200),
    )
    del sam_processor, sam_model

    over_budget = [
        r for r in rows if r.tokens is not None and r.tokens > args.max_tokens
    ]
    embed_path = resolve_prompt_embed(args.prompt_embed)
    summary = {
        # Row paths are relative to these roots, so ``--from_report`` can refuse
        # to replay the report against a different pair of trees.
        **stage_report_header(
            src=src, dst=dst, path_pattern=args.path_pattern, apply=args.apply
        ),
        "rewrite": bool(args.rewrite),
        # A soft prompt is a file: two runs only compare when the sha matches.
        "prompt": args.prompt,
        "prompt_embed": str(embed_path) if embed_path else None,
        "prompt_embed_sha256": prompt_embed_sha256(embed_path),
        "attribution_margin": args.attribution_margin,
        "seen": stats.seen,
        "candidates": stats.candidates,
        "proposed": stats.proposed,
        "written": stats.written,
        # How much of the flat bag the clauses took. Zero under --no_rewrite.
        "rewritten": stats.rewritten,
        "moved_tags": stats.moved_tags,
        # ``reuse_ratio`` is the headline for move-don't-invent: a bag tag can
        # move, a novel one can only ever be an addition.
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
            f"WARNING: {len(over_budget)} caption(s) exceed {args.max_tokens} tokens — "
            "the tail truncates silently at TE-cache time."
        )
    print_dry_run_footer(args.apply, TE_NOTE)
    if args.apply and args.rewrite and stats.moved_tags:
        print(
            f"{stats.moved_tags} tag(s) moved out of the flat bag across "
            f"{stats.rewritten} caption(s). To back that out: "
            '`make caption-position ARGS="--flatten --apply"`.'
        )


if __name__ == "__main__":
    main()
