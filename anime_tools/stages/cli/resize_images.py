"""``python -m anime_tools.stages.cli.resize_images`` — the shell over
:class:`anime_tools.stages.requests.ResizeRequest`, which carries the flags and
the doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.stages.requests import ResizeRequest

DEFAULT_REPORT_DIR = ResizeRequest.report_dir


def build_parser() -> argparse.ArgumentParser:
    return ResizeRequest.parser()


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_resize

    try:
        run_resize(ResizeRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
