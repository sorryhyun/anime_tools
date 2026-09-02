"""The progress line the stage CLIs print — the one thing the CLIs still share
now that their flags are request fields (``stages/requests.py``).

``gui/jobs.py`` reads the ``  [done/total] detail`` format back off the child's
stdout for the panel's progress bar, so it is a contract, not a style.
"""

from __future__ import annotations


def make_progress(every: int, *, first: bool = False):
    """A ``progress(index, total, detail)`` that prints one line every ``every``.

    The last line always prints, so a run under ``every`` images still says it
    finished; ``first`` also prints image 1.
    """

    def progress(index: int, total: int, detail: str) -> None:
        if index == total or index % every == 0 or (first and index == 1):
            print(f"  [{index}/{total}] {detail}", flush=True)

    return progress
