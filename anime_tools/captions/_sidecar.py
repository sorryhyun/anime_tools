"""The tab-delimited caption sidecar — one format, stated once.

``{stem}.variants.txt`` and ``{stem}.history.txt`` are not two file formats that
happen to look alike; they are one format with two vocabularies. A comment
header, then one tab-delimited record per line, in a file named after the
caption it hangs off. What differs between them is only what a record *means* —
a variant is a caption the TE step will encode, a history entry is a caption
that stopped being true — and that is what :mod:`anime_tools.captions.variants`
and :mod:`anime_tools.captions.history` are each left holding.

Two rules live here rather than in both of them:

* **The multi-dot stem.** A sidecar's name is built with ``with_name``, never
  ``with_suffix``, so ``a.b.png`` yields ``a.b.variants.txt`` and not
  ``a.variants.txt`` — a caption whose stem has a dot in it must keep it, or the
  sidecar belongs to a different image than the one that wrote it.
* **The hand-edit tolerance.** A reader skips blank lines, comment lines and any
  line that does not carry the expected field count, rather than raising: these
  files sit beside the captions in plain sight and get opened, so a damaged one
  must cost you the sidecar, never a run. ``\\r`` is stripped off each line for
  the same reason — a sidecar edited on Windows is still a sidecar.

Deliberately **private and stdlib-only**. Both callers are import-light on
purpose (:mod:`anime_tools.stages._caption_io` imports them from *inside* its
functions, so :mod:`anime_tools.stages.replay` stays importable without
:mod:`anime_tools.captions` behind it), and a shared module that pulled anything
in would undo that for both at once.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


def sidecar_header(kind: str) -> str:
    """The comment line every caption sidecar opens with.

    ``kind`` is the sidecar's own word (``variants``, ``history``). The line is
    a header for a person, not a parser: readers skip *any* ``#`` line, so this
    is what the file says to whoever opens it, and the only place it is spelled.
    """
    return f"# anima caption {kind} — auto-generated, do not hand-edit"


def sidecar_path(path: Path, suffix: str) -> Path:
    """``{stem}{suffix}`` beside ``path`` — an image or its ``.txt`` caption.

    ``with_name``, not ``with_suffix``, so a multi-dot stem survives:
    ``a.b.png`` → ``a.b.variants.txt``. An extensionless path keeps its whole
    name.
    """
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.with_name(stem + suffix)


def read_rows(path: Path, fields: int) -> list[list[str]]:
    """Parse a sidecar into its records, in file order.

    ``fields`` is how many tab-separated fields a record has; the split is
    bounded at that many, so only the *first* ``fields - 1`` tabs are
    delimiters and the last field may contain tabs of its own. A line carrying
    some other number of fields is skipped, as are blank and comment lines —
    see the module docstring on why a reader is tolerant here.

    Fields arrive verbatim: stripping a label, or typing a sequence number, is
    the caller's vocabulary rather than the format's. ``path`` must exist —
    whether a missing sidecar is empty or an error differs between the two
    callers, so neither answer is given here.
    """
    out: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\r")
        if not line or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t", fields - 1)
        if len(parts) != fields:
            continue
        out.append(parts)
    return out


def write_rows(path: Path, header: str, rows: Iterable[Sequence[str]]) -> None:
    """Write ``header`` and one tab-joined line per row, creating the parent dir.

    Fields are joined, not escaped: a caption is comma-separated and never
    contains a tab, which is what makes the split on the way back in
    unambiguous. The trailing newline is part of the format.
    """
    lines = [header]
    lines += ["\t".join(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
