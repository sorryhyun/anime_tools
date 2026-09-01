"""Caption reads and writes for stages, with the write invariants.

*Trailing newline*: ``audit_multiview`` writes ``text + "\\n"``, autotag and the
clause rewrite write ``text`` bare. A replay must reproduce the byte-exact file
its native apply would have written, or the next run reads the difference as
drift.

*Variants sidecar*: ``{stem}.variants.txt`` wins over ``{stem}.txt`` at encode
time, so every write into the revised tree passes ``drop_variants=True``.

*History sidecar*: ``history_by`` pushes the text about to be replaced onto
``{stem}.history.txt``; without it the replaced text is gone, since nothing
gates the write.

Torch-free: both sidecar helpers are imported inside the function so
:mod:`anime_tools.stages.replay` stays importable without
:mod:`anime_tools.captions`.
"""

from __future__ import annotations

from pathlib import Path


def read_caption(path: Path) -> str:
    """The stripped text of a caption file, so trailing whitespace never reads
    as drift."""
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

    The history push happens *before* the write, since what it records is the
    text about to be replaced.
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
