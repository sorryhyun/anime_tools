"""Batch auto-tagging: image → Anima Tagger → the revised caption.

Walks the resized tree and writes ``.txt`` sidecars beside the resized image —
the **revised** caption, like every other caption stage. The master — the
``master_dir`` overlay, then the hand-written one under ``source_dir`` — is read
(as ``resolve_caption``'s fallback) and never written, so a run cannot lose text nobody can get back: what it replaces is
pushed onto ``{stem}.history.txt`` and is one badge away in the GUI.

"Unchanged" is measured against what already *speaks* for the image — the
revised caption, or the master behind it — so a proposal equal to the master
(which is what tagging an image whose master the tagger itself wrote produces)
writes nothing at all. A revised copy of an identical master is a second
spelling of one caption for a later hand-edit of the master to go stale behind,
and it buys nothing: Export reads the same ladder, publishing the master for an
image that has no revised caption.

Three modes:

``missing``
    Tag only images no caption speaks for — neither a revised one nor a master.
``merge``
    Tag everything, but only *append* tags the caption does not already carry.
    Clause tags count as present, so a merge after ``caption-position`` cannot
    re-flatten a bound tag back into the bag — which is why the read goes
    through ``resolve_caption``: the clauses live in the revised caption.
``overwrite``
    Replace the caption with the tagger's output.

Two filters sit between the tagger and the caption. ``exclude_tags`` drops the
tags it names from every proposal (:func:`drop_excluded`). ``mask_dir`` tags each
image that has a mask with the mask's ignored region painted out
(:func:`painted_out`), so a speech bubble or background the trainer's masked loss
never sees is not described either; the blank itself reads as ``white
background`` / ``simple background``, so the image is also tagged as it is and
a background tag only the painted copy has is dropped
(:func:`drop_blank_backgrounds`).

**Dry-run is the default** (the caller passes ``apply``). An applied run must be
followed by ``make preprocess-te``, since caption edits do not invalidate the TE
caches.

:func:`run_autotag` is the stage runner over
:class:`~anime_tools.stages.requests.AutotagRequest`; ``cli/autotag_captions.py``
is the shell over it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from anime_tools._env import resolve_path
from anime_tools.captions.position_clauses import compose_caption, parse_caption
from anime_tools.captions.taxonomy import RATING_LITERALS, normalize_tag
from anime_tools.contract import AUTOTAG_MODES, REPLAY_SHAPES
from anime_tools.masking._mask_paths import find_mask

from ._caption_io import write_caption
from ._options import DEFAULT_BATCH_SIZE
from ._progress import make_progress
from ._report import (
    TE_NOTE,
    print_dry_run_footer,
    stage_report_header,
    write_stage_report,
)
from ._walk_captions import CaptionItem, iter_captions

if TYPE_CHECKING:
    from anime_tools.stages.requests import AutotagRequest

MODES = AUTOTAG_MODES


@dataclass
class AutotagOptions:
    """Policy knobs for one batch autotag pass."""

    mode: str = "missing"
    # Extra probability floor on top of the tagger's per-tag F1 thresholds.
    # Forwarded to ``predict_caption``; 0.0 leaves its own decisions untouched.
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0.0 <= float(self.min_confidence) <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0, 1], got {self.min_confidence!r}"
            )


@dataclass
class AutotagProposal:
    """One image's before/after, whether or not it was written."""

    image: str = ""
    caption_path: str = ""
    existing: str = ""
    """What already speaks for this image — the revised caption, or the master
    when there is no revised one yet. What ``merge`` merges into."""
    target_before: str = ""
    """What the *write target* holds right now (``""`` when it does not exist
    yet). The replay's drift baseline, which is not ``existing``: a merge into a
    master writes a revised caption that was never there."""
    proposed: str = ""
    added: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    """Tagger tags ``exclude_tags`` took out of this proposal."""
    masked: bool = False
    """Whether the tags came from the mask-painted image."""
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class AutotagStats:
    seen: int = 0
    candidates: int = 0
    proposed: int = 0
    written: int = 0
    masked: int = 0
    skipped: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1


def _has_rating(tags: tuple[str, ...]) -> bool:
    # Keyed on ``normalize_tag`` like every other "same tag?" test, even though
    # the rating literals are spaceless words where the underscore fold is moot.
    return any(normalize_tag(t) in RATING_LITERALS for t in tags)


def merge_tags(existing: str, tagged: str) -> tuple[str, tuple[str, ...]]:
    """Append the tagger's novel tags to ``existing``, preserving its clauses.

    Returns ``(caption, added)``. A tag already bound inside a position clause
    counts as present, so merging after ``caption-position`` never duplicates a
    clause tag back into the flat bag. The tagger's leading rating is dropped
    when the caption already carries one — two ratings contradict.

    Presence is keyed on :func:`normalize_tag`: the tagger emits ``speech
    bubble`` where a hand-written master may hold ``speech_bubble``, and any
    other key reads one tag as two.
    """
    parsed = parse_caption(existing)
    seen = set(parsed.tag_keys)
    for clause in parsed.clauses:
        seen.update(normalize_tag(t) for t in clause.tags)

    keep_rating = not _has_rating(parsed.flat_tags)
    added: list[str] = []
    for index, tag in enumerate(t.strip() for t in tagged.split(",")):
        if not tag:
            continue
        key = normalize_tag(tag)
        if key in seen:
            continue
        # ``predict_caption`` always puts the rating in slot 0.
        if index == 0 and key in RATING_LITERALS and not keep_rating:
            continue
        seen.add(key)
        added.append(tag)

    if not added:
        return existing, ()
    return compose_caption((*parsed.flat_tags, *added), parsed.clauses), tuple(added)


def _open_rgb(image_path: Path) -> Image.Image:
    with Image.open(image_path) as handle:
        return handle.convert("RGB")


def _tag_pattern(pattern: str) -> re.Pattern[str]:
    """``pattern`` keyed like a tag, with ``*`` as the one wildcard.

    Not ``fnmatch``: ``?`` and ``[…]`` are literal in a tag.
    """
    parts = normalize_tag(pattern).split("*")
    return re.compile(".*".join(re.escape(p) for p in parts))


def drop_excluded(tagged: str, patterns: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Take the tags ``patterns`` name out of the tagger's flat output.

    Returns ``(kept, dropped)``. Matching is on :func:`normalize_tag`, so
    ``speech_bubble`` excludes ``speech bubble``; ``*`` matches any run.
    """
    compiled = [_tag_pattern(p) for p in patterns if p.strip()]
    tags = [t.strip() for t in tagged.split(",") if t.strip()]
    if not compiled:
        return ", ".join(tags), ()
    kept: list[str] = []
    dropped: list[str] = []
    for tag in tags:
        key = normalize_tag(tag)
        (dropped if any(c.fullmatch(key) for c in compiled) else kept).append(tag)
    return ", ".join(kept), tuple(dropped)


BLANK_FILL = (255, 255, 255)
"""What a mask's ignored region is painted with: the colour the tagger letterboxes
with and the position stage blanks a crop with."""


def painted_out(image: Image.Image, mask_path: Path | None) -> Image.Image | None:
    """``image`` with the mask's ignored region (alpha < 128) painted
    :data:`BLANK_FILL`, or ``None`` when there is no mask or it ignores nothing.

    A mask at another size than the image is scaled to it (nearest).
    """
    if mask_path is None:
        return None
    with Image.open(mask_path) as handle:
        mask = handle.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    ignored = np.asarray(mask) < 128
    if not ignored.any():
        return None
    pixels = np.asarray(image).copy()
    pixels[ignored] = BLANK_FILL
    return Image.fromarray(pixels)


def drop_blank_backgrounds(masked: str, plain: str) -> str:
    """The painted image's tags minus the background tags the blank made.

    A tag ending in ``background`` is kept only when the unpainted image has it
    too: painting a region white makes the tagger see ``white background`` /
    ``simple background`` whether or not the picture had one.
    """
    plain_keys = {normalize_tag(t) for t in plain.split(",") if t.strip()}
    kept = []
    for tag in (t.strip() for t in masked.split(",")):
        if not tag:
            continue
        key = normalize_tag(tag)
        if key.endswith("background") and key not in plain_keys:
            continue
        kept.append(tag)
    return ", ".join(kept)


def run_autotag_captions(
    *,
    resized_dir: Path,
    source_dir: Path,
    tag_batch: Callable[[Sequence[Image.Image]], Sequence[str]],
    options: AutotagOptions | None = None,
    path_pattern: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    apply: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
    master_dir: Path | None = None,
    exclude_tags: Sequence[str] = (),
    mask_dir: Path | None = None,
) -> tuple[list[AutotagProposal], AutotagStats]:
    """Walk the resized tree, tag, and (with ``apply``) write the revised caption.

    Tagging runs on the *resized* image (the pixel data training sees) and the
    caption lands beside it under ``resized_dir``. ``master_dir`` (the overlay)
    and ``source_dir`` are the master, read-only here: the fallback half of
    ``resolve_caption``, so "what this image already says" means the revised
    caption when there is one and the master otherwise. ``tag_batch`` is normally
    ``AnimaTagger.predict_caption_batch`` bound to ``min_confidence``: it takes
    images and answers one caption each, in order.

    The pass is in two halves because the forward is batched. The first is
    caption resolution alone — no pixels — and it is what says which images the
    tagger has to see; the second walks every image in tree order, decides it,
    and pulls its tags from a buffer filled ``batch_size`` candidates at a time.
    So the reports, the skip reasons and the progress line are what a
    one-image-at-a-time pass produced, and the GPU sees whole batches.

    ``exclude_tags`` is :func:`drop_excluded`'s pattern list. ``mask_dir`` (the
    mask tree, mirroring ``resized_dir``) turns on mask-aware tagging: an image
    with a mask that ignores anything is tagged twice in its batch, painted and
    as it is, and :func:`drop_blank_backgrounds` merges the two.
    """
    options = options or AutotagOptions()
    stats = AutotagStats()
    rows: list[AutotagProposal] = []
    size = max(1, int(batch_size))

    # Half one, no pixels: what each image already says. Revised first, master
    # as the read-only fallback — `missing` means no caption anywhere, and
    # `merge` merges into the text that actually speaks for the image, clauses
    # included. Captionless images are walked too: they are `missing`'s work.
    walked: list[CaptionItem] = list(
        iter_captions(
            resized_dir,
            source_dir,
            path_pattern,
            stats,
            master_dir=master_dir,
            uncaptioned=True,
        )
    )

    # The candidates, in tree order: exactly the images the loop below will ask
    # the tagger about, which is what lets it fill a whole batch ahead.
    queue = [w for w in walked if not (w.caption and options.mode == "missing")]
    tagged_by_path: dict[Path, tuple[str, bool]] = {}
    taken = 0

    def fill(chunk: list[CaptionItem]) -> None:
        images = [_open_rgb(w.image_path) for w in chunk]
        painted: list[tuple[int, Image.Image]] = []
        if mask_dir is not None:
            for i, (w, image) in enumerate(zip(chunk, images, strict=True)):
                rel_image = w.image_path.relative_to(resized_dir)
                blank = painted_out(image, find_mask(mask_dir, rel_image))
                if blank is not None:
                    painted.append((i, blank))
        texts = [str(t).strip() for t in tag_batch([*images, *(b for _, b in painted)])]
        by_index = {i: texts[len(chunk) + k] for k, (i, _) in enumerate(painted)}
        for i, w in enumerate(chunk):
            if i in by_index:
                tagged_by_path[w.image_path] = (
                    drop_blank_backgrounds(by_index[i], texts[i]),
                    True,
                )
            else:
                tagged_by_path[w.image_path] = (texts[i], False)

    for index, item in enumerate(walked, 1):
        image_path, rel, caption_path, existing = (
            item.image_path,
            item.rel,
            item.dst_caption,
            item.caption,
        )
        if progress is not None:
            progress(index, len(walked), rel.as_posix())

        if existing and options.mode == "missing":
            stats.skip("has-caption")
            continue
        stats.candidates += 1

        if image_path not in tagged_by_path:  # the buffer ran out: fill it
            chunk = queue[taken : taken + size]
            taken += len(chunk)
            fill(chunk)
        tagged, masked = tagged_by_path.pop(image_path)
        tagged, excluded = drop_excluded(tagged, exclude_tags)
        stats.masked += masked

        proposal = AutotagProposal(
            image=image_path.relative_to(resized_dir).as_posix(),
            caption_path=rel.as_posix(),
            existing=existing,
            target_before=item.target_before,
            excluded=excluded,
            masked=masked,
        )
        if not tagged:
            proposal.status = "skip:no-tags"
            stats.skip("no-tags")
            rows.append(proposal)
            continue

        if options.mode == "merge" and existing:
            proposal.proposed, proposal.added = merge_tags(existing, tagged)
        else:
            proposal.proposed = tagged
            proposal.added = tuple(t.strip() for t in tagged.split(",") if t.strip())

        # Against what already **speaks** for the image, which is the master
        # when there is no revised caption yet: a proposal equal to it says
        # nothing new, and a revised copy of a master is a second spelling of
        # one caption for a hand-edit of the master to go stale behind. Export
        # reads the same ladder (`export_workspace._caption_source`), so the
        # image publishes that text either way.
        if proposal.proposed == proposal.existing:
            proposal.status = "skip:unchanged"
            stats.skip("unchanged")
            rows.append(proposal)
            continue

        rows.append(proposal)
        stats.proposed += 1
        if apply:
            # No trailing newline, so the replay of this report writes the
            # same bytes.
            write_caption(
                caption_path,
                proposal.proposed,
                drop_variants=True,
                history_by=REPLAY_SHAPES["autotag"].history_by,
            )
            stats.written += 1

    return rows, stats


def build_tag_batch(
    ckpt_dir: str | Path | None = None,
    device: str | None = None,
    min_confidence: float = 0.0,
) -> tuple[Callable[[Sequence[Image.Image]], list[str]], Mapping[str, object]]:
    """Load the Anima Tagger and return ``(tag_batch, info)``.

    Torch imports live inside this function so ``run_autotag_captions`` stays
    torch-free. The checkpoint is auto-fetched when absent, ``device=None``
    resolves to the GPU when there is one, and the tagger is the per-process
    one (:func:`~anime_tools.stages._models.load_anima_tagger`), so a position
    pass in the same interpreter reuses it.
    """
    from anime_tools.stages._models import load_anima_tagger

    tagger, resolved = load_anima_tagger(ckpt_dir, device)

    def tag_batch(images: Sequence[Image.Image]) -> list[str]:
        return tagger.predict_caption_batch(
            images, min_confidence=float(min_confidence)
        )

    # ``tagger.device`` is a ``torch.device``; the report is JSON.
    return tag_batch, {
        "tagger_dir": str(resolved),
        "device": str(tagger.device),
    }


def run_autotag(req: AutotagRequest):
    """Tag the resized tree and propose (with ``apply``, write) the revised
    caption. Returns ``(rows, stats)`` from the tagging pass.

    No ``--from_report``: the pass is one forward per image, so re-running it is
    cheaper than the machinery to skip it. The report it leaves is still what
    the GUI's Undo reads (``contract.REPLAY_SHAPES["autotag"]``).
    """
    resized_dir = resolve_path(req.dst)
    source_dir = resolve_path(req.src)
    master_dir = resolve_path(req.master)
    report_dir = resolve_path(req.report_dir)
    mask_dir = resolve_path(req.mask_dir) if req.masked else None
    if mask_dir is not None and not mask_dir.is_dir():
        print(f"--masked: no mask tree at {mask_dir}; every image is tagged as it is")

    tag_batch, info = build_tag_batch(
        req.tagger_dir, device=req.device, min_confidence=req.min_confidence
    )
    rows, stats = run_autotag_captions(
        resized_dir=resized_dir,
        source_dir=source_dir,
        master_dir=master_dir,
        tag_batch=tag_batch,
        options=AutotagOptions(mode=req.mode, min_confidence=req.min_confidence),
        path_pattern=req.path_pattern,
        batch_size=req.batch_size,
        apply=req.apply,
        progress=make_progress(50, first=True),
        exclude_tags=req.exclude_tags,
        mask_dir=mask_dir,
    )

    report_path = write_stage_report(
        report_dir,
        {
            "mode": req.mode,
            "min_confidence": req.min_confidence,
            "batch_size": req.batch_size,
            "exclude_tags": list(req.exclude_tags),
            "mask_dir": str(mask_dir) if mask_dir is not None else None,
            **stage_report_header(
                src=source_dir,
                dst=resized_dir,
                path_pattern=req.path_pattern,
                apply=req.apply,
            ),
            **dict(info),
            "stats": {
                "seen": stats.seen,
                "candidates": stats.candidates,
                "proposed": stats.proposed,
                "written": stats.written,
                "masked": stats.masked,
                "skipped": dict(stats.skipped),
            },
            "rows": [asdict(r) for r in rows],
        },
    )

    print(
        f"\nseen={stats.seen} candidates={stats.candidates} "
        f"proposed={stats.proposed} written={stats.written}"
        + (f" masked={stats.masked}" if mask_dir is not None else "")
    )
    for reason, count in stats.skipped.most_common():
        print(f"  skip:{reason} {count}")
    print(f"report: {report_path}")
    print_dry_run_footer(req.apply, TE_NOTE if stats.written else None)
    return rows, stats
