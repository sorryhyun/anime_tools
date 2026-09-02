"""``python -m anime_tools.stages.cli.export_workspace`` — the shell over
:class:`anime_tools.stages.requests.ExportRequest`, which carries the flags and
the doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.stages.requests import DEFAULT_EXPORT_INDEX, ExportRequest

DEFAULT_REPORT_DIR = ExportRequest.report_dir
DEFAULT_INDEX = DEFAULT_EXPORT_INDEX


def build_parser() -> argparse.ArgumentParser:
    return ExportRequest.parser()


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_export

    try:
        run_export(ExportRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
