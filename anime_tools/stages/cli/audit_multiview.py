"""``python -m anime_tools.stages.cli.audit_multiview`` — the shell over
:class:`anime_tools.stages.requests.AuditRequest`, which carries the flags and
the doc (``--help`` prints it). See ``docs/multiview_audit.md``."""

from __future__ import annotations

import argparse

from anime_tools.contract import REPLAY_SHAPES
from anime_tools.stages.requests import AuditRequest

# The writable set is the verdict/confidence gate, not a row ``status``, so
# ``row_filter`` is left open here and closed over the gate at replay time.
REPLAY_SPEC = REPLAY_SHAPES["audit"]
"""The shape ``stages.run`` replays this stage's report through — the same
object ``gui/proposals.py`` reads from ``contract.REPLAY_SHAPES``."""

DEFAULT_REPORT_DIR = AuditRequest.report_dir


def build_parser() -> argparse.ArgumentParser:
    return AuditRequest.parser()


def main(argv: list[str] | None = None) -> None:
    from anime_tools.stages.run import run_audit

    try:
        run_audit(AuditRequest.from_argv(build_parser(), argv))
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
