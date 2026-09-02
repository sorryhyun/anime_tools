"""Rewrite multi-subject captions into position clauses (SAM3 + Anima Tagger).

Detect -> order -> crop+blank -> tag -> compose over the resized dataset, moving
each attributable tag out of the flat bag into its clause (``--flatten`` is the
inverse pass). Clauses land on the revised caption under ``--dst``, never on the
caption master. See ``docs/position_captions.md``.

Dry-run by default; ``--apply`` writes and drops stale ``.variants.txt``
sidecars, so follow it with a TE re-encode.
"""

from __future__ import annotations

import argparse

from anime_tools.contract import REPLAY_SHAPES

# Importing _sam3 also installs the `np.bool` alias sam3 needs before it loads.
from anime_tools.masking._sam3 import (
    add_checkpoint_arg,
)
from anime_tools.stages.cli._args import (
    add_apply_args,
    add_dataset_args,
    add_model_args,
    add_report_dir_arg,
)
from anime_tools.stages.cli._detection import (
    add_detection_args,
)
from anime_tools.stages.position_captions import PositionCaptionOptions
from anime_tools.stages.requests import DEFAULT_MAX_TOKENS, PositionRequest

# ``drop_variants`` mirrors the stage's own write: a stale
# ``{stem}.variants.txt`` outranks ``{stem}.txt`` at encode time.
REPLAY_SPEC = REPLAY_SHAPES["position"]
"""The shape ``stages.run`` replays this stage's report through — the same
object ``gui/proposals.py`` reads from ``contract.REPLAY_SHAPES``."""

DEFAULT_REPORT_DIR = PositionRequest.report_dir


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


def options_from_flag_string(
    flags: str,
) -> tuple[PositionCaptionOptions, PositionRequest]:
    """Parse a flag *string* through this CLI's own parser (the A/B and review
    CLIs take one per arm). Returns ``(options, request)`` — the request too,
    since the detector and tagger are built from it."""
    req = PositionRequest.from_argv(build_parser(), flags.split())
    return req.options(), req


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_position

    try:
        run_position(PositionRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
