"""The one place a stage writes a caption file.

Every caption write carries the same two invariants:

*Trailing newline*
    ``audit_multiview`` writes ``text + "\\n"``; autotag and the clause rewrite
    write ``text`` bare. Not style but compatibility: a replay must reproduce
    the byte-exact file its native apply would have written, or the next run
    reads the difference as drift.

*The variants sidecar*
    ``{stem}.variants.txt`` wins over ``{stem}.txt`` at encode time, so a
    caption rewritten without dropping its sidecar keeps training the *old*
    text. Every write into the revised tree passes ``drop_variants=True``;
    writes into the caption master never need it, because the sidecar lives
    beside the revised caption.

*The history sidecar*
    A run writes for real -- there is no Apply gating it -- so the text it
    replaces has to survive the write or it is simply gone. ``history_by``
    names who is writing and pushes the current text into
    ``{stem}.history.txt`` first; the GUI's caption ladder shows those as
    ``revised@1``, ``revised@2`` … beside the live caption. It is per call site
    rather than automatic because a rung with no history rung on the ladder
    would only be accumulating a file nothing reads -- which is the caption
    master today (``image_dataset/`` is the input tree, and Phase 2 is what
    moves its writes into the workspace).

Torch-free, and deliberately import-light: both sidecar helpers are imported
inside the function so :mod:`anime_tools.stages.replay` stays importable
without :mod:`anime_tools.captions`.
"""

from __future__ import annotations

from pathlib import Path


def read_caption(path: Path) -> str:
    """The stripped text of a caption file. Every before/after comparison goes
    through this, so trailing whitespace can never read as drift."""
    return path.read_text(encoding="utf-8").strip()


def write_caption(
    path: Path,
    text: str,
    *,
    newline: bool = False,
    drop_variants: bool = False,
    history_by: str | None = None,
) -> None:
    """Write one caption, creating its directory and honouring the invariants.

    See the module docstring for what ``newline``, ``drop_variants`` and
    ``history_by`` are protecting. The history push happens *before* the write,
    since what it records is the text about to be replaced.
    """
    if history_by is not None and path.is_file():
        from anime_tools.captions.history import push_history

        push_history(path, read_caption(path), by=history_by)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if newline else ""), encoding="utf-8")
    if drop_variants:
        from anime_tools.captions.variants import variants_sidecar_path

        sidecar = variants_sidecar_path(path)
        if sidecar.exists():
            sidecar.unlink()
