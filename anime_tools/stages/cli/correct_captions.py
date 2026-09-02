"""``python -m anime_tools.stages.cli.correct_captions`` — the shell over
:class:`anime_tools.stages.requests.CorrectRequest`, which carries the flags and
the doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.stages.requests import CorrectRequest


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
