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
    text. Every write into the derived tree passes ``drop_variants=True``;
    writes into the caption master never need it, because the sidecar lives
    beside the derived caption.

Torch-free, and deliberately import-light: the sidecar path helper is imported
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
) -> None:
    """Write one caption, creating its directory and honouring both invariants.

    See the module docstring for what ``newline`` and ``drop_variants`` are
    protecting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if newline else ""), encoding="utf-8")
    if drop_variants:
        from anime_tools.captions.variants import variants_sidecar_path

        sidecar = variants_sidecar_path(path)
        if sidecar.exists():
            sidecar.unlink()
