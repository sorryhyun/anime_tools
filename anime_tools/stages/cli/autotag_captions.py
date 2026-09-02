"""``python -m anime_tools.stages.cli.autotag_captions`` — the shell over
:class:`anime_tools.stages.requests.AutotagRequest`, which carries the flags and
the doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.contract import REPLAY_SHAPES
from anime_tools.stages.requests import AutotagRequest

# The proposal lands on the **revised** caption (``--dst``); the master is the
# read-only fallback the tagger merged into, so the drift baseline is the target's
# own text (``target_before``), not what spoke for the image.
REPLAY_SPEC = REPLAY_SHAPES["autotag"]
"""The shape ``stages.run`` replays this stage's report through — the same
object ``gui/proposals.py`` reads from ``contract.REPLAY_SHAPES``."""

DEFAULT_REPORT_DIR = AutotagRequest.report_dir


def build_parser() -> argparse.ArgumentParser:
    return AutotagRequest.parser()


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_autotag

    try:
        run_autotag(AutotagRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
