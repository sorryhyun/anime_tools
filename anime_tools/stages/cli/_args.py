"""The progress callback the stage CLIs pass down — the one thing the CLIs
still share now that their flags are request fields (``stages/requests.py``).

The format itself and the thinning live in :mod:`anime_tools._progress`, which
is also where the stages that walk their own loop get :class:`~anime_tools.
_progress.ProgressBar`; this is the callback shape a library function takes.
"""

from __future__ import annotations

from anime_tools import _progress


def make_progress(every: int, *, first: bool = False) -> _progress.Progress:
    """A ``progress(index, total, detail)`` that prints one line every ``every``.

    The last line always prints, so a run under ``every`` images still says it
    finished; ``first`` also prints image 1.
    """

    def progress(index: int, total: int, detail: str) -> None:
        _progress.progress_line(index, total, detail, every=every, first=first)

    return progress
