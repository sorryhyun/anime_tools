"""Auto-tag the dataset with the Anima Tagger and write the revised caption.

Walks the resized tree and proposes a caption per image, in one of three
``--mode``s: ``missing`` (images no caption speaks for, the default), ``merge``
(append novel tags, keeping clauses) or ``overwrite`` (replace the caption
outright). The caption it reads is the revised one, falling back to the master;
the caption it writes is always the revised one, and what that replaced is kept
as a ``{stem}.history.txt`` version.

Dry-run by default; ``--apply`` writes. The TE caches go stale but still look
current, so follow a real apply with ``make preprocess-te``.
"""

from __future__ import annotations

import argparse

from anime_tools.contract import REPLAY_SHAPES
from anime_tools.stages.autotag import (
    MODES,
)
from anime_tools.stages.cli._args import (
    add_apply_args,
    add_dataset_args,
    add_model_args,
    add_report_dir_arg,
)
from anime_tools.stages.requests import AutotagRequest

# The proposal lands on the **revised** caption (``--dst``); the master is the
# read-only fallback the tagger merged into, so the drift baseline is the target's
# own text (``target_before``), not what spoke for the image.
REPLAY_SPEC = REPLAY_SHAPES["autotag"]
"""The shape ``stages.run`` replays this stage's report through — the same
object ``gui/proposals.py`` reads from ``contract.REPLAY_SHAPES``."""

DEFAULT_REPORT_DIR = AutotagRequest.report_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(p)
    p.add_argument(
        "--mode",
        choices=MODES,
        default="missing",
        help="missing: only images no caption speaks for (default). merge: "
        "append novel tags to every caption, keeping its position clauses. "
        "overwrite: replace the caption outright (the replaced text is kept as "
        "a history version)",
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
        apply_help="Write the proposed captions into the revised tree "
        "(default: dry run)",
        from_report_help="Replay a previous dry run's report.json instead of "
        "re-tagging: writes exactly the captions it proposed and loads no "
        "model. Skips any row whose caption changed since. Emits "
        "apply_report.json (never clobbers the report it reads)",
    )
    add_report_dir_arg(p, DEFAULT_REPORT_DIR)
    add_model_args(p)
    return p


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_autotag

    try:
        run_autotag(AutotagRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
