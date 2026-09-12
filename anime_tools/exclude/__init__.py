"""Taking an image out of the pipeline, and putting it back.

An exclusion is a curation decision, not a stage: no model, no report, no diff to
agree to. It moves every file the image has out of the live workspace trees into
``workspace/_excluded/`` and writes down where each came from
(:data:`~anime_tools.workspace.EXCLUDED`); un-excluding moves them back.

| Half | What is in it |
|---|---|
| :mod:`._ledger` | the ledger — the state an exclusion *is*, and the rel that keys it |
| :mod:`._artifacts` | what an image is made of: every file that has to move |
| :mod:`._move` | the move engine — :func:`exclude_one` / :func:`restore_one` |
| :mod:`._cli` | ``python -m anime_tools.exclude`` |

The enforcement is one chokepoint: ``resize`` reads :func:`excluded_rels` into
its own ``--skip``, and every other stage walks ``workspace/resized/``, which an
excluded image has left. Torch-free.
"""

from __future__ import annotations

from anime_tools.exclude._artifacts import artifacts
from anime_tools.exclude._cli import build_parser, main
from anime_tools.exclude._ledger import (
    MANIFEST_VERSION,
    Entry,
    ExclusionError,
    Trees,
    excluded_rels,
    is_excluded,
    manifest_path,
    read_entries,
    rel_key,
    write_entries,
)
from anime_tools.exclude._move import Result, exclude_one, restore_one

__all__ = [
    "MANIFEST_VERSION",
    "Entry",
    "ExclusionError",
    "Result",
    "Trees",
    "artifacts",
    "build_parser",
    "exclude_one",
    "excluded_rels",
    "is_excluded",
    "main",
    "manifest_path",
    "read_entries",
    "rel_key",
    "restore_one",
    "write_entries",
]
