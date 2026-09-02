"""``python -m anime_tools.masking.cli.generate_masks`` — the shell over
:class:`anime_tools.masking.requests.SamMaskRequest`, which carries the flags and
the doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.masking._sam3 import prompt_list
from anime_tools.masking.requests import SamMaskRequest

__all__ = ["build_parser", "main", "prompt_list"]


def build_parser() -> argparse.ArgumentParser:
    return SamMaskRequest.parser(formatter_class=argparse.RawDescriptionHelpFormatter)


def main(argv: list[str] | None = None) -> None:
    from anime_tools.masking.sam import run_sam_masks

    run_sam_masks(SamMaskRequest.from_argv(build_parser(), argv))


if __name__ == "__main__":
    main()
