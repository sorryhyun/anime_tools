"""CLI entry — argparse + mode dispatcher.

External-corpus paths are resolved via the ``CAPTION_CORPUS_DIR`` env var
(typically set in ``anima_lora/.env``). The corpus directory is expected to
contain ``retrieved/`` (raw caption pool), ``selected/`` (curated subset),
``tag_rules.yaml`` (caption normalization rules), and ``.tag_cache.json``
(per-tag Booru-style category cache, indexed under ``retrieved/``). All of
these can be overridden individually by CLI flags.

Modes (selected by ``--mode``):

* ``build_vocab``    — scan caption sources, intersect with the tag-taxonomy
                       cache, snapshot ``tag_rules.yaml``, emit ``vocab.json``
                       (label space) plus a per-stem ``dataset.json`` manifest
                       that carries the fixed train/val split.
* ``predict``        — single-image debug entry (any backend).
* ``scan_role_markers`` / ``derive_groups`` — vocab curation helpers.

The PE-head training modes (``build_features`` / ``train`` / ``calibrate`` /
``embed_tags``) were archived 2026-08-27 with the dbv4 backend migration —
see ``_archive/anima_tagger_training/`` and
``docs/experimental/anima_tagger.md``. Sidecar training is
``anime_tools/tagger/cli/train_sidecar.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from anime_tools._env import load_dotenv  # noqa: E402
from anime_tools._env import setup_logging  # noqa: E402

# Pull CAPTION_CORPUS_DIR from anima_lora/.env before argparse builds defaults;
# CLI flags still win over env values.
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)


def _corpus_default(rel: str):
    """Resolve ``$CAPTION_CORPUS_DIR/<rel>`` for argparse defaults.

    Returns ``None`` when the env var is unset so argparse renders an
    explicit '(unset)' marker in --help instead of a misleading empty path.
    """
    root = os.environ.get("CAPTION_CORPUS_DIR")
    if not root:
        return None
    return str(Path(root) / rel)


def _default_tag_cache():
    """Default tag-taxonomy source for ``--tag_cache``.

    Prefers the corpus JSON when ``$CAPTION_CORPUS_DIR`` is set; otherwise falls
    back to the publicly downloadable ``models/danbooru_tags_classified.csv`` KB
    (``make download-danbooru-tags``), so the vocab build works without the
    private crawl. Returns ``None`` only when neither is resolvable.
    """
    corpus = _corpus_default("retrieved/.tag_cache.json")
    if corpus:
        return corpus
    csv_kb = (
        Path(__file__).resolve().parents[2] / "models" / "danbooru_tags_classified.csv"
    )
    return str(csv_kb) if csv_kb.exists() else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Anima tagger vocab / predict / curation CLI"
    )
    p.add_argument(
        "--mode",
        choices=[
            "build_vocab",
            "predict",
            "scan_role_markers",
            "derive_groups",
        ],
        default="build_vocab",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device for predict (default: cuda if available).",
    )
    # Vocab-build inputs default to subpaths of ``$CAPTION_CORPUS_DIR``.
    raw_default = _corpus_default("retrieved")
    curated_default = _corpus_default("selected")
    p.add_argument(
        "--caption_roots",
        nargs="+",
        default=[d for d in (curated_default, raw_default, "image_dataset") if d],
        help="Directories to scan recursively for *.txt caption files. "
        "First-match-wins by stem when a duplicate appears across roots, so "
        "list curated roots before raw ones. Defaults: "
        "$CAPTION_CORPUS_DIR/selected + $CAPTION_CORPUS_DIR/retrieved + "
        "image_dataset/.",
    )
    p.add_argument(
        "--tag_cache",
        default=_default_tag_cache(),
        help="Tag-taxonomy source mapping tag → Danbooru type ID. Accepts the "
        "corpus JSON ($CAPTION_CORPUS_DIR/retrieved/.tag_cache.json) or the "
        "public danbooru_tags_classified.csv KB. Default: the corpus JSON when "
        "$CAPTION_CORPUS_DIR is set, else models/danbooru_tags_classified.csv.",
    )
    p.add_argument(
        "--rules",
        default=_corpus_default("tag_rules.yaml"),
        help="Caption-normalization rules (snapshotted into out_dir at "
        "build time). Default: $CAPTION_CORPUS_DIR/tag_rules.yaml.",
    )
    p.add_argument(
        "--groups",
        default=_corpus_default("tag_groups.yaml"),
        help="Tag-groups YAML (typed groupings — eye_color, hair_color, "
        "rating, …). Resolved against the kept vocab and embedded into "
        "vocab.json[groups]; the YAML is snapshotted to out_dir/groups.yaml. "
        "Optional — pass empty / unset to build a flat-vocab checkpoint. "
        "Default: $CAPTION_CORPUS_DIR/tag_groups.yaml.",
    )
    p.add_argument("--min_freq", type=int, default=20)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--image",
        default=None,
        help="Image path for --mode predict.",
    )
    p.add_argument(
        "--show_scores",
        action="store_true",
        help="Predict mode: also print rating distribution + top-K kept tags.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Predict mode: number of top kept tags to show with --show_scores.",
    )

    # scan_role_markers: high solo co-occurrence ratio → likely a class marker
    # mis-typed as character.
    p.add_argument(
        "--min_solo",
        type=int,
        default=5,
        help="scan_role_markers: drop tags with fewer than this many solo "
        "occurrences (default: 5).",
    )
    p.add_argument(
        "--min_ratio",
        type=float,
        default=0.5,
        help="scan_role_markers: drop tags whose conditional co-occurrence "
        "ratio with another character on solo images is below this (default: 0.5).",
    )
    p.add_argument(
        "--top_partners",
        type=int,
        default=3,
        help="scan_role_markers: how many top co-occurring partners to print "
        "per row (default: 3).",
    )
    p.add_argument(
        "--min_role_partners",
        type=int,
        default=5,
        help="scan_role_markers: a candidate with at least this many distinct "
        "co-occurrence partners is classified D_role (broad pool → "
        "affiliation marker). Default: 5.",
    )
    p.add_argument(
        "--pair_dominance",
        type=float,
        default=0.6,
        help="scan_role_markers: a candidate whose top-1 partner accounts for "
        "at least this fraction of co-occurrences is classified C_pair "
        "(narrow pool → genuine couple/sibling). Default: 0.6.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=200,
        help="scan_role_markers: cap rows printed in the table (default: 200).",
    )
    p.add_argument(
        "--out_yaml",
        default=None,
        help="scan_role_markers: optional path for a YAML stub of candidates, "
        "ready to paste into tag_rules.yaml. derive_groups: path for the "
        "candidate groups.yaml.",
    )

    # derive_groups: bucket general vocab by danbooru 소분류 taxonomy → group
    # candidates; co-occurrence on solo images picks softmax vs multilabel.
    p.add_argument(
        "--min_group_size",
        type=int,
        default=3,
        help="derive_groups: minimum members for a taxonomy bucket to become a "
        "candidate group (default: 3).",
    )
    p.add_argument(
        "--min_member_freq",
        type=int,
        default=50,
        help="derive_groups: drop group members appearing in fewer than this "
        "many training captions (default: 50).",
    )
    p.add_argument(
        "--min_group_support",
        type=int,
        default=30,
        help="derive_groups: a group seen on fewer than this many solo images "
        "can't be trusted for exclusivity → defaults to multilabel (default: 30).",
    )
    p.add_argument(
        "--softmax_cooc_max",
        type=float,
        default=0.05,
        help="derive_groups: a group whose members co-occur on at most this "
        "fraction of single-subject images is mutually exclusive → "
        "softmax_when_solo (default: 0.05).",
    )
    p.add_argument(
        "--borderline_cooc_max",
        type=float,
        default=0.20,
        help="derive_groups: groups with multi-rate between --softmax_cooc_max "
        "and this are flagged 'borderline' (attribute families inflated by "
        "hierarchical/mixed tags) — emitted multilabel but tagged PROMOTE? for "
        "review (default: 0.20).",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="derive_groups: print a coverage + per-group table to stdout.",
    )
    p.add_argument(
        "--derive_groups",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build_vocab: derive tag-groups from the danbooru taxonomy + the "
        "scanned captions and merge onto --groups (preserved verbatim), writing "
        "<out_dir>/groups.yaml and baking it into vocab.json — folds the "
        "derive_groups step into the build. On by default; pass "
        "--no-derive_groups to build a flat-vocab checkpoint or use a static "
        "--groups file. Skipped with a warning when the danbooru CSV KB is "
        "absent. (As a --mode, derive_groups runs standalone for review.)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="derive_groups: write a curated, English-keyed groups.yaml that "
        "merges the derived groups onto --preserve_groups (kept verbatim) "
        "instead of the raw candidate. Destination is --out_yaml or "
        "<out_dir>/groups.yaml (backed up to .bak first).",
    )
    p.add_argument(
        "--preserve_groups",
        default="models/captioners/anima-tagger-dbv4/groups.yaml",
        help="derive_groups --apply: existing groups.yaml whose groups are "
        "preserved verbatim and claim their tags first (no regression).",
    )

    # --out_dir holds the checkpoint + vocab (build_vocab writes here, the
    # other modes read from it).
    p.add_argument(
        "--out_dir",
        default="models/captioners/anima-tagger-dbv4",
    )

    args = p.parse_args()

    if args.mode == "build_vocab":
        missing = [
            name
            for name, val in (
                ("--tag_cache", args.tag_cache),
                ("--rules", args.rules),
            )
            if not val
        ]
        if missing or not args.caption_roots:
            raise SystemExit(
                "build_vocab needs CAPTION_CORPUS_DIR set in anima_lora/.env "
                f"(or {', '.join(missing) or '--caption_roots'} passed "
                "explicitly). Add a line like\n"
                "    CAPTION_CORPUS_DIR=/path/to/corpus\n"
                "to anima_lora/.env, or pass the paths via CLI flags."
            )

    return args


def main() -> None:
    args = parse_args()
    if args.mode == "build_vocab":
        from .vocab import cmd_build_vocab

        cmd_build_vocab(args)
    elif args.mode == "predict":
        from .predict import cmd_predict

        cmd_predict(args)
    elif args.mode == "scan_role_markers":
        from .role_markers import cmd_scan_role_markers

        cmd_scan_role_markers(args)
    elif args.mode == "derive_groups":
        from .derive_groups import cmd_derive_groups

        cmd_derive_groups(args)
    else:
        raise SystemExit(f"unknown --mode={args.mode!r}")


if __name__ == "__main__":
    main()
