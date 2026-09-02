"""``python -m anime_tools.masking.cli.merge_masks`` — the shell over
:class:`anime_tools.masking.requests.MergeMasksRequest`, which carries the flags
and the doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.masking.requests import MergeMasksRequest

DEFAULT_INPUTS = list(MergeMasksRequest.mask_dirs)
"""The two generators' own ``--mask-dir`` defaults, in the order they run."""


def build_parser() -> argparse.ArgumentParser:
    return MergeMasksRequest.parser()


def main(argv: list[str] | None = None) -> None:
    from anime_tools.masking.merge import run_merge_masks

    run_merge_masks(MergeMasksRequest.from_argv(build_parser(), argv))


if __name__ == "__main__":
    main()
