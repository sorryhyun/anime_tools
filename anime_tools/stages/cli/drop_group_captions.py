"""``python -m anime_tools.stages.cli.drop_group_captions`` — the shell over
:class:`anime_tools.stages.requests.DropGroupRequest`, which carries the flags and
the doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.contract import REPLAY_SHAPES
from anime_tools.stages.requests import DropGroupRequest

REPLAY_SPEC = REPLAY_SHAPES["drop_groups"]
"""The shape this stage's ``report.json`` is read back through — the same object
``gui/proposals.py`` reads from ``contract.REPLAY_SHAPES``. This stage has no
``--from_report``, so the only reader is the GUI's Undo."""

DEFAULT_REPORT_DIR = DropGroupRequest.report_dir


def build_parser() -> argparse.ArgumentParser:
    return DropGroupRequest.parser()


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.drop_groups import run_drop_groups

    parser = build_parser()
    try:
        run_drop_groups(DropGroupRequest.from_argv(parser, argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
