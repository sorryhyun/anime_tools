"""``python -m anime_tools.stages.cli.position_captions`` — the shell over
:class:`anime_tools.stages.requests.PositionRequest`, which carries the flags
and the doc (``--help`` prints it). See ``docs/position_captions.md``."""

from __future__ import annotations

import argparse

from anime_tools.contract import REPLAY_SHAPES
from anime_tools.stages.position_captions import PositionCaptionOptions
from anime_tools.stages.requests import DEFAULT_MAX_TOKENS, PositionRequest

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_REPORT_DIR",
    "REPLAY_SPEC",
    "PositionRequest",
    "build_parser",
    "main",
    "options_from_flag_string",
]

# ``drop_variants`` mirrors the stage's own write: a stale
# ``{stem}.variants.txt`` outranks ``{stem}.txt`` at encode time.
REPLAY_SPEC = REPLAY_SHAPES["position"]
"""The shape ``stages.run`` replays this stage's report through — the same
object ``gui/proposals.py`` reads from ``contract.REPLAY_SHAPES``."""

DEFAULT_REPORT_DIR = PositionRequest.report_dir


def build_parser() -> argparse.ArgumentParser:
    return PositionRequest.parser()


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
