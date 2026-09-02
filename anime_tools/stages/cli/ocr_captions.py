"""``python -m anime_tools.stages.cli.ocr_captions`` — the shell over
:class:`anime_tools.stages.requests.OcrRequest`, which carries the flags and the
doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.stages.requests import OcrRequest

DEFAULT_REPORT_DIR = OcrRequest.report_dir


def build_parser() -> argparse.ArgumentParser:
    return OcrRequest.parser()


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_ocr

    try:
        run_ocr(OcrRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
