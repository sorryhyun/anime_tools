"""``python -m anime_tools.grouping.cli.build_groups`` — the shell over
:class:`anime_tools.grouping.requests.GroupRequest`, which carries the flags and
the doc (``--help`` prints it)."""

from __future__ import annotations

import argparse

from anime_tools.grouping.requests import DEFAULT_EMBEDDER, GroupRequest

__all__ = ["DEFAULT_EMBEDDER", "build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    return GroupRequest.parser()


def main(argv: list[str] | None = None) -> None:
    from anime_tools.grouping.groups import run_groups

    run_groups(GroupRequest.from_argv(build_parser(), argv))


if __name__ == "__main__":
    main()
