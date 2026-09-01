"""Previous versions of one caption — the ``{stem}.history.txt`` sidecar.

A caption file holds one text: whatever the last write left there. That was
survivable while a stage run was a *proposal* you inspected and then applied,
because the moment before the write was a thing you could look at. It is not
survivable once Run writes directly, so the write records what it replaced:
every superseded text lands here, oldest first, and the GUI's caption ladder
expands them into one badge apiece (``revised@1``, ``revised@2`` …) beside the
version rungs above them.

Deliberately the *same shape* as ``{stem}.variants.txt`` — a comment header and
tab-delimited lines beside the caption it belongs to — because it is the same
kind of thing: a generated sidecar naming captions that are not the file. Two
consequences follow from living beside the caption rather than in a tree of its
own. The history moves with its rung (Phase 2 relocates the revised master and
its history follows with no code change), and the only path a writer needs is
the one it was already holding, which is what keeps
:func:`anime_tools.stages._caption_io.write_caption` a function of one path.

Torch-free, stdlib-only, and import-light for the same reason ``variants`` is:
:mod:`anime_tools.stages.replay` imports it from inside a function.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

HISTORY_SIDECAR_SUFFIX = ".history.txt"
_SIDECAR_HEADER = "# anima caption history — auto-generated, do not hand-edit"

HISTORY_LIMIT = 10
"""How many superseded versions one caption keeps.

A cap rather than an archive: this is the "what did that run replace?" readout,
and a badge row nobody can count is no more legible than no history at all. The
oldest entries fall off the front; sequence numbers do **not** renumber, so a
badge you are looking at cannot become a different version while you read it.
"""


@dataclass(frozen=True)
class HistoryEntry:
    """One superseded caption: what it said, when it stopped saying it, and what
    replaced it.

    ``label`` is the badge — the rung's own name with the sequence appended, so
    ``revised@3`` reads as *the third thing ``revised`` used to be*.
    """

    seq: int
    at: str
    by: str
    text: str

    def label(self, kind: str) -> str:
        return f"{kind}@{self.seq}"

    def note(self) -> str:
        """The badge's tooltip line: who replaced it, and when."""
        return f"{self.by} · {self.at}" if self.by else self.at


def history_sidecar_path(caption_path: Path) -> Path:
    """``{stem}.history.txt`` beside a caption (or its image).

    ``with_name``, not ``with_suffix``, so a multi-dot stem survives:
    ``a.b.txt`` → ``a.b.history.txt``. Same rule as
    :func:`anime_tools.captions.variants.variants_sidecar_path`, and for the
    same reason.
    """
    p = caption_path
    stem = p.name[: -len(p.suffix)] if p.suffix else p.name
    return p.with_name(stem + HISTORY_SIDECAR_SUFFIX)


def read_history(path: Path) -> list[HistoryEntry]:
    """Parse a history sidecar into an ordered list, oldest first.

    ``path`` is the *sidecar*. Comment and blank lines are skipped and a line
    that does not carry the four fields is ignored, the way the variants reader
    tolerates a hand-edit: a damaged sidecar costs you history, never a run.
    """
    if not path.is_file():
        return []
    out: list[HistoryEntry] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\r")
        if not line or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t", 3)
        if len(parts) != 4:
            continue
        seq, at, by, text = parts
        try:
            n = int(seq)
        except ValueError:
            continue
        out.append(HistoryEntry(seq=n, at=at.strip(), by=by.strip(), text=text))
    return out


def write_history(path: Path, entries: list[HistoryEntry]) -> None:
    """Write the sidecar, or delete it when there is nothing left to say."""
    if not entries:
        if path.is_file():
            path.unlink()
        return
    lines = [_SIDECAR_HEADER]
    lines += [f"{e.seq}\t{e.at}\t{e.by}\t{e.text}" for e in entries]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def push_history(
    caption_path: Path,
    text: str,
    *,
    by: str = "",
    limit: int = HISTORY_LIMIT,
) -> HistoryEntry | None:
    """Record ``text`` as a superseded version of ``caption_path``.

    Called with the text the caller is *about to overwrite*, so the newest entry
    is always the immediately previous caption. Returns the entry recorded, or
    ``None`` when there was nothing worth recording:

    * an empty text — a caption being created replaces nothing;
    * a text the newest entry already holds, so a re-run that rewrites the same
      caption twice does not push the same version twice.

    Sequence numbers continue from the highest ever recorded rather than from
    the list length, so trimming the front cannot make two different versions
    share a badge.
    """
    body = (text or "").strip()
    if not body:
        return None
    sidecar = history_sidecar_path(caption_path)
    entries = read_history(sidecar)
    if entries and entries[-1].text == body:
        return None
    entry = HistoryEntry(
        seq=(entries[-1].seq + 1) if entries else 1,
        at=datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M"),
        by=by,
        text=body,
    )
    entries.append(entry)
    write_history(sidecar, entries[-max(1, int(limit)) :])
    return entry


def drop_history(caption_path: Path) -> None:
    """Delete a caption's history. Only for a caption that is itself being
    deleted (an Undo of a run that *created* one): the versions of a file that
    no longer exists are versions of nothing."""
    sidecar = history_sidecar_path(caption_path)
    if sidecar.is_file():
        sidecar.unlink()
