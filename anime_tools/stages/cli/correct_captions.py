"""Write corrected captions next to resized preprocessing images."""

from __future__ import annotations

import argparse

from anime_tools.captions.tag_drop_groups import drop_group_names
from anime_tools.stages.cli._args import add_path_pattern_arg
from anime_tools.stages.requests import CorrectRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Raw source image directory")
    parser.add_argument("--dst", required=True, help="Resized image directory")
    parser.add_argument(
        "--tag_csv",
        default=None,
        help="danbooru_tags_classified.csv path (default: models/ lookup)",
    )
    add_path_pattern_arg(
        parser,
        help="Only write captions for resized images matching this relative glob",
    )
    parser.add_argument("--recursive", action="store_true", help="Walk subfolders")
    parser.add_argument(
        "--caption_insert_no_artist",
        "--caption-insert-no-artist",
        dest="caption_insert_no_artist",
        action="store_true",
        help="Insert @no-artist at the artist slot when no artist marker exists",
    )
    parser.add_argument(
        "--caption_trigger_word",
        "--caption-trigger-word",
        dest="caption_trigger_word",
        default="",
        help="Trigger tag to move into the caption order",
    )
    parser.add_argument(
        "--caption_trigger_at_front",
        "--caption-trigger-at-front",
        dest="caption_trigger_at_front",
        action="store_true",
        help="Place caption_trigger_word at the very front instead of artist slot",
    )
    parser.add_argument(
        "--caption_drop_groups",
        "--caption-drop-groups",
        dest="caption_drop_groups",
        default="",
        help=(
            "Comma-separated tag groups to strip from every mirrored caption "
            "(the master is never edited). Slugs: "
            + ", ".join(drop_group_names())
            + "; anything else is a literal taxonomy-path prefix from "
            "danbooru_tags_classified.csv (e.g. '효과/연출 > 조명')."
        ),
    )
    parser.add_argument(
        "--no_correct",
        "--no-correct",
        dest="no_correct",
        action="store_true",
        help=(
            "Skip bucket-reordering — mirror the raw source caption verbatim as "
            "v0 (the variant-only path: shuffle sidecars without reordering)."
        ),
    )
    parser.add_argument(
        "--caption_shuffle_variants",
        "--caption-shuffle-variants",
        dest="caption_shuffle_variants",
        type=int,
        default=0,
        help=(
            "Number of caption variants to materialize as {stem}.variants.txt "
            "sidecars (0 = none). v0 is the corrected caption; v1..v{N-1} are "
            "smart-shuffled. The TE step encodes these verbatim."
        ),
    )
    parser.add_argument(
        "--caption_tag_dropout_rate",
        "--caption-tag-dropout-rate",
        dest="caption_tag_dropout_rate",
        type=float,
        default=0.0,
        help="Per-tag dropout probability for v1..v{N-1} (ignored without variants).",
    )
    parser.add_argument(
        "--caption_tag_randomize_rate",
        "--caption-tag-randomize-rate",
        dest="caption_tag_randomize_rate",
        type=float,
        default=0.0,
        help=(
            "Per-tag identity-randomize probability — emits an r-family alongside "
            "the v-family. Needs --qwen3 + --t5_tokenizer_path to build the "
            "dual-single erasure pool. Ignored without >=2 variants."
        ),
    )
    parser.add_argument(
        "--qwen3",
        type=str,
        default=None,
        help="Qwen3 tokenizer directory (tokenizer-only load; required for randomize).",
    )
    parser.add_argument(
        "--t5_tokenizer_path",
        "--t5-tokenizer-path",
        dest="t5_tokenizer_path",
        type=str,
        default=None,
        help="T5 tokenizer directory (spiece.model + tokenizer.json).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_correct

    try:
        run_correct(CorrectRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
