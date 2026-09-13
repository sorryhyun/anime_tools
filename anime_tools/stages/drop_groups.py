"""Dropping whole tag groups from the revised captions.

:func:`run_drop_groups` is the stage runner over
:class:`~anime_tools.stages.requests.DropGroupRequest`;
``cli/drop_group_captions.py`` is the shell over it. The cut itself is
:func:`anime_tools.captions.correction.drop_caption_groups` — the drop Correct
makes, without the reorder Correct makes around it, which is why this is a stage
of its own rather than a Correct knob: a run here changes nothing but the tags
it names.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from anime_tools._env import resolve_path
from anime_tools.captions.correction import TagKnowledgeBase, drop_caption_groups
from anime_tools.captions.tag_drop_groups import should_drop_tag
from anime_tools.captions.taxonomy import normalize_tag
from anime_tools.contract import REPLAY_SHAPES

from ._caption_io import read_caption, write_caption
from ._progress import make_progress
from ._report import print_dry_run_footer, stage_report_header, write_stage_report
from ._walk_captions import iter_captions
from .captions import TE_NOTE, resolve_tag_csv
from .resize import resized_tree

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from anime_tools.stages.requests import DropGroupRequest


@dataclass
class DropGroupStats:
    seen: int = 0
    written: int = 0
    unchanged: int = 0
    no_caption: int = 0
    """Images with neither a revised nor a master caption."""
    from_master: int = 0
    """Revised captions this run created out of a master."""
    by_group: Counter[str] = field(default_factory=Counter)
    """Tags dropped, per selector — the first one that took each tag."""

    def skip(self, reason: str) -> None:
        if reason == "no-caption":
            self.no_caption += 1


@dataclass
class DropGroupProposal:
    """One image's before/after, written or not. Correct's row shape, plus the
    tags that went: ``target_before`` is the write target's own text (empty
    when the run creates it), ``existing`` what spoke for the image."""

    image: str = ""
    caption_path: str = ""
    existing: str = ""
    target_before: str = ""
    proposed: str = ""
    dropped: list[str] = field(default_factory=list)
    status: str = "ok"


def _selector_of(tag: str, kb: TagKnowledgeBase, selectors: tuple[str, ...]) -> str:
    key = normalize_tag(tag)
    return next((s for s in selectors if should_drop_tag(key, kb, (s,))), selectors[0])


def drop_tag_groups(
    source_dir: Path,
    resized_dir: Path,
    kb: TagKnowledgeBase,
    selectors: Iterable[str],
    *,
    keep: Iterable[str] = (),
    recursive: bool = True,
    path_pattern: str | None = None,
    apply: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[DropGroupProposal], DropGroupStats]:
    """Cut ``selectors`` out of every caption under the resized tree.

    Revised first, master as the fallback, like every caption stage. An image
    that loses nothing is ``skip:unchanged`` and writes nothing — not even a
    revised copy of the master it read, which would say nothing new. A write
    pushes the replaced text onto the history sidecar and drops the variants
    sidecar, whose v0 is the caption before the cut.
    """
    selectors = tuple(selectors)
    keep = tuple(keep)
    history_by = REPLAY_SHAPES["drop_groups"].history_by
    stats = DropGroupStats()
    rows: list[DropGroupProposal] = []
    for image_path, rel, dst, raw, caption_path in iter_captions(
        resized_dir, source_dir, path_pattern, stats, progress, recursive=recursive
    ):
        cut = drop_caption_groups(raw, kb, selectors, keep=keep)
        row = DropGroupProposal(
            image=str(image_path.relative_to(resized_dir)),
            caption_path=str(rel),
            existing=raw,
            target_before=read_caption(dst) if dst.exists() else "",
            proposed=cut.text,
            dropped=list(cut.dropped_tags),
        )
        rows.append(row)
        if not cut.dropped_tags:
            row.status = "skip:unchanged"
            stats.unchanged += 1
            continue
        if caption_path != dst:
            stats.from_master += 1
        stats.by_group.update(_selector_of(t, kb, selectors) for t in cut.dropped_tags)
        if apply:
            write_caption(dst, cut.text, drop_variants=True, history_by=history_by)
        stats.written += 1
    return rows, stats


def run_drop_groups(req: DropGroupRequest):
    """Drop the request's tag groups from the revised captions. Returns
    ``(rows, stats)``.

    No ``--from_report``: the pass is pure text, so re-running it is cheaper
    than the machinery to skip it. The report it leaves is what the GUI's Undo
    replays backwards (``contract.REPLAY_SHAPES["drop_groups"]``).
    """
    from anime_tools.captions.correction import load_tag_knowledge_base

    selectors = req.selectors
    if not selectors:
        # Here rather than in the request: an empty request is still the
        # default one every form and test builds.
        raise ValueError(
            "Nothing to drop: pick at least one --groups slug or --category_paths prefix."
        )
    src = resolve_path(req.src)
    dst = resized_tree(req.dst)
    kb = load_tag_knowledge_base(resolve_tag_csv(req.tag_csv))

    rows, stats = drop_tag_groups(
        src,
        dst,
        kb,
        selectors,
        keep=req.keep_tags,
        recursive=req.recursive,
        path_pattern=req.path_pattern or "*",
        apply=req.apply,
        progress=make_progress(50),
    )

    report_path = write_stage_report(
        resolve_path(req.report_dir),
        {
            **stage_report_header(
                src=src, dst=dst, path_pattern=req.path_pattern, apply=req.apply
            ),
            "selectors": list(selectors),
            "keep_tags": list(req.keep_tags),
            "stats": {
                "seen": stats.seen,
                "written": stats.written,
                "unchanged": stats.unchanged,
                "from_master": stats.from_master,
                "no_caption": stats.no_caption,
                "by_group": dict(stats.by_group.most_common()),
            },
            "rows": [asdict(r) for r in rows],
        },
    )

    verb = "written" if req.apply else "would write"
    print(
        f"Dropped tag groups ({', '.join(selectors)}): "
        f"{stats.written} {verb}, {stats.unchanged} unchanged, "
        f"{stats.from_master} mirrored from the master, "
        f"{stats.no_caption} without a caption ({stats.seen} resized images)"
    )
    for selector, n in stats.by_group.most_common():
        print(f"  {selector}: {n} tag{'' if n == 1 else 's'}")
    print(f"report: {report_path}")
    print_dry_run_footer(req.apply, TE_NOTE if stats.written else None)
    return rows, stats
