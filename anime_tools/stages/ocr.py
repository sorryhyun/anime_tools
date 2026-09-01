"""Batch OCR: image → PP-OCRv6 → a text sidecar, and a script tag in the caption.

Walks the resized tree and answers two different questions per image, which is
why it writes to two different places:

*What does the picture say?*
    Every recognized line, with its box, language and confidence, into
    ``{stem}.ocr.txt`` beside the revised caption
    (:mod:`anime_tools.captions.ocr_sidecar`). Curation-private, not in the
    contract, and nothing downstream encodes it.

*What should the caption say about that?*
    At most a tag — ``english text``, ``chinese text``, ``bilingual text`` —
    appended to the flat bag. Never the recognized string: the caption grammar
    has no clause that could hold one, and adding it is a two-repo change to a
    frozen contract. See :mod:`anime_tools.ocr.script` for why Japanese earns no
    tag at all and still fills a sidecar.

So a Japanese-only corpus is the shape that looks wrong and is not: every image
gets a sidecar, no caption changes, and the run honestly reports zero proposals.

The caption write is the same one the clause rewrite makes — into the **revised**
tree, never the master, dropping the stale ``{stem}.variants.txt`` and pushing the
replaced text onto ``{stem}.history.txt`` so a run needs no Apply gate in front
of it. **Dry-run is the default** (the caller passes ``apply``), and a dry run
still reports every line it would have written, so the sidecar can be reviewed
before it exists.

An applied run changes captions, so it must be followed by the trainer's TE
re-encode like every other caption stage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from anime_tools.captions.ocr_sidecar import OcrLine, write_ocr_for
from anime_tools.captions.position_clauses import compose_caption, parse_caption
from anime_tools.captions.taxonomy import normalize_tag
from anime_tools.ocr.script import tags_for

from ._caption_io import write_caption
from ._walk_captions import iter_captions

HISTORY_BY = "ocr"
"""What the superseded caption is filed under, so a ``revised@N`` badge says
which stage replaced it. Shared with the replay spec, or an Undo would file its
own write under a different hand than the run it is undoing."""


@dataclass
class OcrOptions:
    """Policy for one OCR pass — what to keep and what to say about it."""

    langs: tuple[str, ...] = ("en", "ja", "zh")
    """Which languages survive into the sidecar. A line in any other one is
    dropped whole: an allowlist rather than a preference, because the point of
    running OCR over an anime corpus is usually one script."""

    tags: bool = True
    """Write the script tags into the caption. Off makes the stage a pure
    reader — sidecars only, no caption touched, nothing to undo — which is the
    honest mode for a first pass over a corpus nobody has looked at yet."""

    def __post_init__(self) -> None:
        if not self.langs:
            raise ValueError("langs must name at least one language")


@dataclass
class OcrProposal:
    """One image's OCR, its caption before/after, and what happened."""

    image: str = ""
    caption_path: str = ""
    existing: str = ""
    proposed: str = ""
    lines: tuple[OcrLine, ...] = ()
    added: tuple[str, ...] = ()
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_row(self) -> dict[str, object]:
        """The report row. ``lines`` are expanded here rather than held as
        dataclasses so a dry run's report *is* the sidecar it would write."""
        return {
            "image": self.image,
            "caption_path": self.caption_path,
            "existing": self.existing,
            "proposed": self.proposed,
            "lines": [line.to_dict() for line in self.lines],
            "added": list(self.added),
            "status": self.status,
        }


@dataclass
class OcrStats:
    seen: int = 0
    candidates: int = 0
    with_text: int = 0
    lines: int = 0
    proposed: int = 0
    written: int = 0
    sidecars: int = 0
    langs: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1


def keep_lines(lines: Sequence[OcrLine], langs: Sequence[str]) -> tuple[OcrLine, ...]:
    """The allowed lines, renumbered from 1 in the order they arrived.

    Renumbering after the filter and not before is what keeps a sidecar's ``seq``
    a contiguous reading order rather than a record of which lines were thrown
    away — the sidecar is a readout, and a gap in it reads as a bug.
    """
    allowed = set(langs)
    kept = [line for line in lines if line.lang in allowed]
    return tuple(
        OcrLine(seq=i, box=ln.box, lang=ln.lang, score=ln.score, text=ln.text)
        for i, ln in enumerate(kept, 1)
    )


def add_script_tags(
    caption: str, lines: Sequence[OcrLine]
) -> tuple[str, tuple[str, ...]]:
    """Append the script tags ``lines`` earn, keeping the caption's clauses.

    Returns ``(caption, added)``, and ``added`` is empty whenever the caption
    already says it — presence is keyed on
    :func:`anime_tools.captions.taxonomy.normalize_tag`, so the ``english_text``
    a hand-written master spells with an underscore counts as the
    ``english text`` this would have added. A tag already bound inside a position
    clause counts as present too, for the reason
    :func:`anime_tools.stages.autotag.merge_tags` says: a stage that runs after
    the clause rewrite must not re-flatten a bound tag back into the bag.
    """
    parsed = parse_caption(caption)
    seen = set(parsed.tag_keys)
    for clause in parsed.clauses:
        seen.update(normalize_tag(t) for t in clause.tags)

    added = [
        tag
        for tag in tags_for(ln.lang for ln in lines)
        if normalize_tag(tag) not in seen
    ]
    if not added:
        return caption, ()
    return compose_caption((*parsed.flat_tags, *added), parsed.clauses), tuple(added)


def run_ocr_captions(
    *,
    resized_dir: Path,
    source_dir: Path,
    read_fn: Callable[[Path], list[OcrLine]],
    options: OcrOptions | None = None,
    path_pattern: str | None = None,
    apply: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[OcrProposal], OcrStats]:
    """Walk the resized tree, read every image, and (with ``apply``) write.

    Only images that already have a caption are read, because the two things a
    run produces both hang off one: the sidecar is named after the caption and
    the tag goes inside it. An uncaptioned image is ``skip:no-caption``, the same
    answer every other stage on this tree gives.

    The sidecar is written whenever ``apply`` is on and the caption tag is
    decided separately, so ``--no-tags`` still fills sidecars and a
    Japanese-only image still gets one while its caption is left alone.
    """
    options = options or OcrOptions()
    stats = OcrStats()
    rows: list[OcrProposal] = []

    walked = list(iter_captions(resized_dir, source_dir, path_pattern, stats))
    for index, (image_path, rel, dst_caption, caption) in enumerate(walked, 1):
        if progress is not None:
            progress(index, len(walked), str(rel))
        stats.candidates += 1

        lines = keep_lines(read_fn(image_path), options.langs)
        stats.lines += len(lines)
        for line in lines:
            stats.langs[line.lang] += 1
        if lines:
            stats.with_text += 1

        proposal = OcrProposal(
            image=str(image_path.relative_to(resized_dir)),
            caption_path=str(rel),
            existing=caption,
            proposed=caption,
            lines=lines,
        )
        if options.tags and lines:
            proposal.proposed, proposal.added = add_script_tags(caption, lines)
        if not proposal.added:
            proposal.status = "no-tags" if lines else "no-text"
        rows.append(proposal)

        if apply:
            # Unconditional, and before the caption write: the sidecar describes
            # the image rather than the caption, so it is just as true for a row
            # whose caption gains nothing — and `write_ocr_for` deletes a stale
            # one when this pass found no text, which is what makes a re-run over
            # re-cropped pixels self-correcting.
            write_ocr_for(dst_caption, lines)
            stats.sidecars += 1

        if not proposal.ok:
            stats.skip(proposal.status)
            continue
        stats.proposed += 1
        if apply:
            write_caption(
                dst_caption,
                proposal.proposed,
                drop_variants=True,
                history_by=HISTORY_BY,
            )
            stats.written += 1

    return rows, stats
