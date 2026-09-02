"""``python -m anime_tools.masking.cli.generate_masks_mit`` — the shell over
:class:`anime_tools.masking.requests.MitMaskRequest`, which carries the flags,
the two drawers and the doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.masking._sam3 import prompt_list
from anime_tools.masking.requests import MitMaskRequest, prompts_flag

__all__ = ["DEFAULT_SAM_PROMPTS", "build_parser", "detectors", "main", "prompt_list"]

DEFAULT_SAM_PROMPTS = prompts_flag(MitMaskRequest.sam_prompts)


def build_parser() -> argparse.ArgumentParser:
    return MitMaskRequest.parser(formatter_class=argparse.RawDescriptionHelpFormatter)


def request(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> MitMaskRequest:
    """The parsed flags as a request — or exit saying why not.

    Both drawers shut is a run that would load no model, walk the whole tree and write
    nothing — caught here, before the first weight is read.
    """
    try:
        return MitMaskRequest.from_namespace(args)
    except ValueError as e:
        parser.error(str(e))


def detectors(parser: argparse.ArgumentParser, args) -> tuple[bool, tuple[str, ...]]:
    """``(run the segmenter, the SAM3 prompts)``."""
    req = request(parser, args)
    return req.use_mit, req.active_sam_prompts


def main(argv: list[str] | None = None) -> None:
    from anime_tools.masking.mit import run_mit_masks

    parser = build_parser()
    run_mit_masks(request(parser, parser.parse_args(argv)))


if __name__ == "__main__":
    main()
