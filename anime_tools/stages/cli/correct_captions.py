"""``python -m anime_tools.stages.cli.correct_captions`` — the shell over
:class:`anime_tools.stages.requests.CorrectRequest`, which carries the flags and
the doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.contract import REPLAY_SHAPES
from anime_tools.stages.requests import CorrectRequest

# The correction lands on the **revised** caption (``--dst``); the master is the
# read-only fallback it mirrors from, so the drift baseline is the target's own
# text (``target_before``), not what spoke for the image.
REPLAY_SPEC = REPLAY_SHAPES["correct"]
"""The shape this stage's ``report.json`` is read back through — the same object
``gui/proposals.py`` reads from ``contract.REPLAY_SHAPES``. This stage has no
``--from_report``, so the only reader is the GUI's Undo."""

DEFAULT_REPORT_DIR = CorrectRequest.report_dir


def build_parser() -> argparse.ArgumentParser:
    return CorrectRequest.parser()


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_correct

    try:
        run_correct(CorrectRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
