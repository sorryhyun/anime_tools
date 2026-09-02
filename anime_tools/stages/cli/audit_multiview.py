"""Audit `1girl` captions for images that are really several views of one girl.

Sweeps the images the position stage skips as ``single-subject`` and reports
every one where the ``girl`` prompt finds two or more subjects. See
``docs/multiview_audit.md``.

Dry-run by default; ``--apply`` writes the missing tag into the caption master,
so follow it with a TE re-encode. ``image_dataset/`` is gitignored, so an apply
is not git-recoverable — keep ``report.json``, it holds the before-text.
"""

from __future__ import annotations

import argparse

from anime_tools.contract import REPLAY_SHAPES
from anime_tools.masking._sam3 import add_checkpoint_arg
from anime_tools.stages.cli._args import (
    add_apply_args,
    add_dataset_args,
    add_model_args,
    add_report_dir_arg,
)
from anime_tools.stages.cli._detection import (
    add_detection_args,
)
from anime_tools.stages.multiview_audit import (
    DEFAULT_IDENTITY_CONFIDENCE,
    DEFAULT_MULTIVIEW_PROB,
    EXTRA_CHARACTER,
    MULTIPLE_VIEWS,
)
from anime_tools.stages.requests import AuditRequest

# The writable set is the verdict/confidence gate, not a row ``status``, so
# ``row_filter`` is left open here and closed over the gate at replay time.
REPLAY_SPEC = REPLAY_SHAPES["audit"]
"""The shape ``stages.run`` replays this stage's report through — the same
object ``gui/proposals.py`` reads from ``contract.REPLAY_SHAPES``."""

DEFAULT_REPORT_DIR = AuditRequest.report_dir


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


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_audit

    try:
        run_audit(AuditRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
