"""Position-aware caption rewrite — detect subjects, bind tags to sides.

Detect the ``girl`` instances in a multi-subject image, order them, tag each
mask-blanked crop, and rewrite the caption as
``<flat tag bag>. On the left, akita neru, yellow eyes. On the right, ...``.
An attributable tag *moves* out of the flat bag into its clause, so each
attribute is asserted exactly once; ``rewrite=False`` keeps the additive v1
behaviour. Reversible via :func:`flatten_captions`.

The library half takes its two models as injected callables
(``detect_fn``/``tag_fn``), staying import-free of SAM3/the tagger;
:func:`run_position`, the stage runner, is what loads them — and the multiview
audit phase that runs first, over the captions this sweep rejects
(:func:`~anime_tools.stages.multiview_audit.run_audit_phase`).

Per-rule evidence and the knob table live in ``docs/position_captions.md``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from anime_tools._env import resolve_path
from anime_tools._json import write_json
from anime_tools.captions.caption_layout import (
    caption_boy_count,
    caption_panel_ceiling,
    caption_subject_count,
    is_candidate,
    is_repeated_subject_layout,
)
from anime_tools.captions.clause_rewrite import MovedTag, RemovalPlan, plan_bag_removals
from anime_tools.captions.clause_vocabulary import (
    ClauseGroups,
    ClauseVocabulary,
    load_clause_groups,
    load_clause_vocabulary,
)
from anime_tools.captions.position_clauses import (
    PositionClause,
    assign_positions,
    compose_caption,
    flatten_caption,
    has_clauses,
    ordered_indices,
    parse_caption,
)
from anime_tools.captions.taxonomy import normalize_tag
from anime_tools.contract import REPLAY_SHAPES

# Re-exported: the knob dataclass is a leaf so the GUI's schema build can read the
# defaults without importing this module (:mod:`stages._options`).
from anime_tools.stages._options import PositionCaptionOptions
from anime_tools.stages.instance_detection import (
    Detection,
    box_area,
    box_containment,
    box_iou,
    crop_instance,
    dedupe_detections,
    drop_small_boxes,
    mask_box_fill,
    mask_containment,
    merge_part_detections,
)

from ._analysis import ANALYSIS_SUBDIR, clear_analysis, write_analysis
from ._caption_io import read_caption, write_caption
from ._progress import make_progress
from ._report import print_dry_run_footer, stage_report_header, write_stage_report
from ._walk_captions import iter_captions

if TYPE_CHECKING:
    from anime_tools.stages.requests import PositionRequest

# Convenience re-exports: canonical homes are the modules imported above, but
# consumers reach for them on this module.
__all__ = [
    "ClauseGroups",
    "ClauseVocabulary",
    "Detection",
    "ImageProposal",
    "InstanceProposal",
    "MovedTag",
    "PositionCaptionOptions",
    "PositionCaptionStats",
    "RemovalPlan",
    "box_area",
    "box_containment",
    "box_iou",
    "caption_boy_count",
    "caption_panel_ceiling",
    "caption_subject_count",
    "crop_instance",
    "dedupe_detections",
    "detect_subjects",
    "drop_small_boxes",
    "flatten_captions",
    "is_candidate",
    "is_repeated_subject_layout",
    "load_clause_groups",
    "load_clause_vocabulary",
    "mask_box_fill",
    "mask_containment",
    "merge_part_detections",
    "plan_bag_removals",
    "propose_for_image",
    "run_position",
    "run_position_captions",
    "summarize",
]


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


@dataclass
class InstanceProposal:
    position: str
    box: list[int]
    score: float
    tags: list[str]
    crop: str | None = None
    source: str = "subject"
    # How many of ``tags`` the flat bag did not already contain.
    novel: int = 0


@dataclass
class ImageProposal:
    image: str
    caption_path: str
    status: str
    detected: int = 0
    expected: int | None = None
    original: str = ""
    """What spoke for the image and the clauses were composed from — the revised
    caption, or the master when there is no revised one yet."""
    target_before: str = ""
    """What the *write target* holds right now (``""`` when it does not exist
    yet). The replay's drift baseline, which is not ``original``: a rewrite of a
    master writes a revised caption that was never there, and the undo of that
    write is a delete."""
    proposed: str | None = None
    instances: list[InstanceProposal] = field(default_factory=list)
    # Boxes as detected, recorded even when a gate rejects the image (reviewer
    # evidence); ``instances`` only populates once every gate passes.
    detections: list[dict] = field(default_factory=list)
    tokens: int | None = None
    # Which bag tags the clauses took, and which reached a clause but stayed
    # flat (tag -> the rule that pinned it). Both empty under ``rewrite=False``.
    moved: list[dict] = field(default_factory=list)
    pinned: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "proposed"


@dataclass
class PositionCaptionStats:
    seen: int = 0
    candidates: int = 0
    proposed: int = 0
    written: int = 0
    rewritten: int = 0
    moved_tags: int = 0
    # Clause tags in total, and how many were novel (not in the caption).
    # ``clause_tags - novel_tags`` is reuse.
    clause_tags: int = 0
    novel_tags: int = 0
    pinned_tags: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    # Captions the multiview audit phase handed over, and how many of those the
    # sweep could not turn into clauses and wrote for the tag alone.
    promoted: int = 0
    promoted_written: int = 0

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def pin(self, reason: str) -> None:
        self.pinned_tags[reason] = self.pinned_tags.get(reason, 0) + 1


def detect_subjects(
    image: Image.Image,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    options: PositionCaptionOptions,
    expected: int | None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
) -> list[Detection]:
    """Detect + dedupe, with two escalations when the count falls short.

    Both fire only when detection undershoots the expected count: a lower score
    threshold, which recovers an extreme close-up, then body-part prompts
    (``part_detect_fn``), which recover a headless panel the subject prompt
    can't see at any threshold.

    GOTCHA: target is ``expected or min_instances``, NOT ``expected`` alone — a
    ``multiple views`` sheet reports ``expected=None`` on purpose (count tags
    characters, not views); gating on truthiness would skip the retry for that
    whole population.
    """

    def run(threshold: float) -> list[Detection]:
        dets = dedupe_detections(
            detect_fn(image, threshold),
            options.iou_threshold,
            options.containment_threshold,
            options.dedupe_fill_ratio,
            options.mask_containment_threshold,
        )
        return drop_small_boxes(dets, image.size, options.min_area_frac)

    dets = run(options.score_threshold)
    target = expected or options.min_instances
    if len(dets) < target and options.retry_score_threshold < options.score_threshold:
        retry = run(options.retry_score_threshold)
        if len(retry) > len(dets):
            dets = retry

    if len(dets) >= target or part_detect_fn is None or not options.part_prompts:
        return dets

    parts: list[Detection] = []
    for prompt in options.part_prompts:
        parts.extend(part_detect_fn(image, prompt, options.part_score_threshold))
    parts = drop_small_boxes(parts, image.size, options.min_area_frac)
    merged = merge_part_detections(
        dets,
        parts,
        iou_threshold=options.iou_threshold,
        containment_threshold=options.part_containment_threshold,
    )
    # Top up to the target, no further: a part prompt can fragment into more
    # boxes than there are real panels.
    return merged[: max(target, len(dets))]


def _relax_bag_keeps(
    kept_sets: list[dict[str, float]],
    score_sets: list[dict[str, float]],
    predictions: list[Mapping[str, object]],
    flat_bag: frozenset[str],
    options: PositionCaptionOptions,
) -> None:
    """Admit sub-threshold flat-bag tags into each crop's kept set, in place.

    Needs the per-tag thresholds ``AnimaTagger.predict`` attaches; a no-op per
    crop when a stub ``tag_fn`` omits them. See ``bag_relax``.
    """
    relax = options.bag_relax
    word_relax = options.bag_word_relax
    if relax >= 1.0 and word_relax >= 1.0:
        return
    for kept, scores, pred in zip(kept_sets, score_sets, predictions):
        thresholds = pred.get("thresholds") or {}
        for tag in flat_bag:
            if tag in kept or tag not in scores or tag not in thresholds:
                continue
            floor = thresholds[tag] * relax * word_relax ** (len(tag.split()) - 1)
            floor = max(floor, options.bag_relax_min_score)
            if scores[tag] >= floor:
                kept[tag] = float(scores[tag])


def propose_for_image(
    image: Image.Image,
    caption: str,
    *,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions,
    crop_sink: Callable[[int, str, Image.Image], str] | None = None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
    mask_sink: Callable[[list[Detection]], None] | None = None,
) -> ImageProposal:
    """Build the clause proposal for one image. Never writes any caption.

    ``mask_sink`` sees the detections each time their order is settled: as
    detected (``proposal.detections``' order, which is all a skip has), then
    reading-ordered once the gates pass (``proposal.instances``' order). The
    last call is the one that matches the returned proposal."""
    # Parsed once and asked five times: every layout question below takes the
    # parse (``caption_layout``), so the caption string is never re-split.
    parsed = parse_caption(caption)
    flat_bag = parsed.tag_keys
    expected = caption_subject_count(parsed)

    proposal = ImageProposal(
        image="",
        caption_path="",
        status="proposed",
        expected=expected,
        original=caption,
    )

    dets = detect_subjects(image, detect_fn, options, expected, part_detect_fn)
    if mask_sink is not None:
        mask_sink(dets)
    proposal.detected = len(dets)
    proposal.detections = [
        {
            "box": [int(v) for v in d.box],
            "score": round(float(d.score), 3),
            "source": d.source,
        }
        for d in dets
    ]
    if len(dets) < options.min_instances:
        proposal.status = "skip:too-few-instances"
        return proposal
    if len(dets) > options.max_instances:
        proposal.status = "skip:too-many-instances"
        return proposal
    # Detection and the caption's count must agree, else we'd write clauses we
    # can't ground. "Agree" is the range girls..girls+boys, because the ``girl``
    # prompt picks up males inconsistently.
    if options.strict_count and expected:
        boys = caption_boy_count(parsed)
        upper = None if boys is None else expected + boys
        if len(dets) < expected or (upper is not None and len(dets) > upper):
            proposal.status = "skip:count-mismatch"
            return proposal
    # A layout tag waives the check above (``expected`` is None by design); an
    # ``Nkoma`` tag restores a generous ceiling so a subject detected twice
    # still has a backstop.
    if options.strict_count and not expected:
        ceiling = caption_panel_ceiling(parsed)
        if ceiling is not None and len(dets) > ceiling:
            proposal.status = "skip:count-mismatch"
            return proposal

    order = ordered_indices([d.box for d in dets], image.size, row_tol=options.row_tol)
    dets = [dets[i] for i in order]
    if mask_sink is not None:
        mask_sink(dets)
    positions = assign_positions(
        [d.box for d in dets], image.size, row_tol=options.row_tol
    )

    # GOTCHA: mask-blanking is a subject-crop fix. On a part box the mask IS the
    # part, so blanking would delete the very content (torn jeans, pantyhose)
    # the part pass exists to recover. Part crops take the plain padded bbox.
    crops = [
        crop_instance(
            image,
            d,
            pad=options.pad,
            blank=options.blank_crops and d.source == "subject",
        )
        for d in dets
    ]
    predictions = [tag_fn(crop) for crop in crops]
    kept_sets = [dict(p.get("kept") or {}) for p in predictions]
    score_sets = [dict(p.get("scores") or {}) for p in predictions]
    _relax_bag_keeps(kept_sets, score_sets, predictions, flat_bag, options)
    # A tag only *this* crop keeps is attributable to it; one every crop keeps
    # discriminates nothing and stays in the flat bag.
    counts: dict[str, int] = {}
    for kept in kept_sets:
        for tag in kept:
            counts[tag] = counts.get(tag, 0) + 1
    attributable = frozenset(t for t, n in counts.items() if n == 1)
    shared = frozenset(t for t, n in counts.items() if n == len(kept_sets))
    view_invariant = options.multi_view_gate and is_repeated_subject_layout(parsed)

    for i, (det, kept, pred) in enumerate(zip(dets, kept_sets, predictions)):
        tags = vocabulary.select(
            kept,
            dict(pred.get("groups") or {}),
            flat_bag=flat_bag,
            attributable=attributable,
            shared=shared,
            max_tags=options.max_clause_tags,
            name_confidence=options.name_confidence,
            allow_unlisted_names=options.allow_unlisted_names,
            discriminative_only=options.discriminative_only,
            allow_identity=det.source == "subject",
            bag_gated_identity=options.bag_gated_identity,
            view_invariant=view_invariant,
            bind_framing=options.bind_framing,
            bind_view_anatomy=options.bind_view_anatomy,
            max_novel_tags=options.max_novel_tags,
        )
        crop_name = crop_sink(i, positions[i], crops[i]) if crop_sink else None
        proposal.instances.append(
            InstanceProposal(
                position=positions[i],
                box=[int(v) for v in det.box],
                score=round(float(det.score), 3),
                tags=tags,
                crop=crop_name,
                source=det.source,
                # ``flat_bag`` is ``parsed.tag_keys``, so the probe must use the
                # same key or an underscored tag never matches.
                novel=sum(1 for t in tags if normalize_tag(t) not in flat_bag),
            )
        )

    clauses = [
        PositionClause(position=inst.position, tags=tuple(inst.tags))
        for inst in proposal.instances
        if inst.tags
    ]
    if len(clauses) < options.min_instances:
        # Every crop tagged identically — the subjects are genuinely
        # indistinguishable to the tagger, so there is nothing to bind.
        proposal.status = "skip:no-discriminative-tags"
        return proposal

    flat = list(parsed.flat_tags)
    if options.rewrite:
        plan = plan_bag_removals(
            parsed.flat_tags,
            [inst.tags for inst in proposal.instances],
            [inst.position for inst in proposal.instances],
            kept_sets,
            score_sets,
            vocabulary=vocabulary,
            margin=options.attribution_margin,
        )
        proposal.pinned = dict(plan.blocked)
        taken = {normalize_tag(m.tag) for m in plan.moved}
        remaining = [t for t in flat if normalize_tag(t) not in taken]
        # The rewrite removes text, so guard against emptying the bag.
        if remaining:
            flat = remaining
            proposal.moved = [
                {"tag": m.tag, "position": m.position, "margin": m.margin}
                for m in plan.moved
            ]

    proposal.proposed = compose_caption(flat, clauses)
    return proposal


# ---------------------------------------------------------------------------
# Review artifacts
# ---------------------------------------------------------------------------


def _crop_sink(crops_dir: Path, rel: Path) -> Callable[[int, str, Image.Image], str]:
    """Save each crop under ``crops_dir`` mirroring the dataset layout — the
    dry-run review artifact, the exact pixels the tagger saw."""
    target = crops_dir / rel.parent

    def sink(index: int, position: str, crop: Image.Image) -> str:
        target.mkdir(parents=True, exist_ok=True)
        name = f"{rel.stem}_{index}_{position.replace(' ', '-')}.png"
        crop.save(target / name)
        return str((target / name).relative_to(crops_dir))

    return sink


def _save_skip_overlay(
    crops_dir: Path, rel: Path, image: Image.Image, proposal: ImageProposal
) -> None:
    """Draw the detected boxes over a skipped image, under ``_skipped/``.

    A skip produces no crops, so the overlay is the reviewer's only evidence.
    """
    from PIL import ImageDraw

    target = crops_dir / "_skipped" / rel.parent
    target.mkdir(parents=True, exist_ok=True)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for i, det in enumerate(proposal.detections):
        box = det["box"]
        draw.rectangle(box, outline=(255, 0, 0), width=4)
        draw.text(
            (box[0] + 6, box[1] + 6), f"{i}:{det['score']:.2f}", fill=(255, 255, 0)
        )
    status = proposal.status.removeprefix("skip:")
    canvas.save(target / f"{rel.stem}_{status}.png")


# ---------------------------------------------------------------------------
# Dataset passes
# ---------------------------------------------------------------------------


def run_position_captions(
    *,
    resized_dir: Path,
    source_dir: Path,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions | None = None,
    path_pattern: str | None = None,
    apply: bool = False,
    crops_dir: Path | None = None,
    token_count_fn: Callable[[str], int] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
    promoted: Mapping[str, str] | None = None,
    analysis_dir: Path | None = None,
) -> tuple[list[ImageProposal], PositionCaptionStats]:
    """Walk the resized tree, propose clauses, and (with ``apply``) write them.

    ``promoted`` is the multiview audit phase's verdict, ``{caption_path:
    caption}`` (:func:`~anime_tools.stages.multiview_audit.promotions`): the
    caption it names REPLACES the walked one before
    :func:`~anime_tools.captions.caption_layout.is_candidate` sees it, so an
    image the audit just tagged ``multiple views`` is swept here instead of
    being rejected as ``single-subject`` — and ``is_repeated_subject_layout``
    reads the same tag, arming the view-invariant gate. The promoted text is
    what a clause proposal composes from, so under ``apply`` the tag persists
    with the clauses; a promoted image whose proposal fails is written with the
    tag alone rather than losing it.

    GOTCHA: the caption master (``source_dir``) is NEVER written — the rewrite
    lands at ``resized_dir/<rel>``, what the TE step encodes, and the master is
    only the read fallback. That holds for a promoted caption too: the audit's
    tag lands in the revised tree, the one every later stage actually reads.
    The stale ``{stem}.variants.txt`` sidecar, which wins over ``{stem}.txt`` at
    encode time, is dropped alongside the write. The write replaces rather than
    appends, so ``apply`` defaults off.

    ``analysis_dir`` keeps each swept image's proposal and instance masks
    (:mod:`~anime_tools.stages._analysis`) — the GUI's analysis badge; an image
    the sweep walks but does not propose for loses the pair it had.
    """
    options = options or PositionCaptionOptions()
    stats = PositionCaptionStats()
    rows: list[ImageProposal] = []
    promoted = promoted or {}

    walked = list(iter_captions(resized_dir, source_dir, path_pattern, stats))
    for index, (image_path, rel, dst_caption, caption) in enumerate(walked, 1):
        if progress is not None:
            progress(index, len(walked), str(rel))
        promotion = promoted.get(rel.as_posix())
        if promotion is not None and promotion != caption:
            caption = promotion
            stats.promoted += 1
        image_rel = image_path.relative_to(resized_dir).as_posix()
        ok, reason = is_candidate(caption)
        if not ok:
            clear_analysis(analysis_dir, image_rel)
            stats.skip(reason)
            # A promotion the sweep still cannot use (the gate admitted it, but
            # the tag did not make it a candidate) is written on its own so the
            # audit's verdict is not silently dropped.
            if promotion is not None and apply:
                write_caption(
                    dst_caption, caption, drop_variants=True, history_by="audit"
                )
                stats.promoted_written += 1
            continue
        stats.candidates += 1

        crop_sink = _crop_sink(crops_dir, rel) if crops_dir is not None else None
        # What the analysis label map is drawn from: the order propose_for_image
        # settled last, which is the order its proposal's rows are in.
        labelled: list[Detection] = []

        def hold_labels(dets: list[Detection], _held=labelled) -> None:
            _held[:] = dets

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        proposal = propose_for_image(
            image,
            caption,
            detect_fn=detect_fn,
            tag_fn=tag_fn,
            vocabulary=vocabulary,
            options=options,
            crop_sink=crop_sink,
            part_detect_fn=part_detect_fn,
            mask_sink=hold_labels if analysis_dir is not None else None,
        )
        proposal.image = str(image_path.relative_to(resized_dir))
        proposal.caption_path = str(rel)
        proposal.target_before = (
            read_caption(dst_caption) if dst_caption.exists() else ""
        )
        rows.append(proposal)
        if analysis_dir is not None:
            record = asdict(proposal)
            record["labels"] = "instances" if proposal.instances else "detections"
            write_analysis(analysis_dir, image_rel, record, labelled, image.size)

        if not proposal.ok:
            stats.skip(proposal.status.removeprefix("skip:"))
            if crops_dir is not None:
                _save_skip_overlay(crops_dir, rel, image, proposal)
            # No clauses to write, but the audit's tag is a fact about the
            # picture and outlives a failed proposal.
            if promotion is not None and apply:
                write_caption(
                    dst_caption, caption, drop_variants=True, history_by="audit"
                )
                stats.promoted_written += 1
            continue
        stats.proposed += 1
        stats.clause_tags += sum(len(i.tags) for i in proposal.instances)
        stats.novel_tags += sum(i.novel for i in proposal.instances)
        if proposal.moved:
            stats.rewritten += 1
            stats.moved_tags += len(proposal.moved)
        for reason in proposal.pinned.values():
            stats.pin(reason)
        if token_count_fn is not None and proposal.proposed:
            proposal.tokens = token_count_fn(proposal.proposed)
        if apply:
            write_caption(
                dst_caption,
                proposal.proposed,
                drop_variants=True,
                history_by="position",
            )
            stats.written += 1

    return rows, stats


def flatten_captions(
    *,
    resized_dir: Path,
    source_dir: Path,
    path_pattern: str | None = None,
    apply: bool = False,
) -> tuple[list[dict], PositionCaptionStats]:
    """Undo a rewrite: merge every caption's clauses back into its flat bag.

    Text-only (no SAM3, no tagger, no pixels), writing the same revised caption
    as :func:`run_position_captions`.

    GOTCHA: hand-written clauses are flattened too — the pass can't tell them
    from generated ones. Hence the dry-run default.
    """
    stats = PositionCaptionStats()
    rows: list[dict] = []
    for _, rel, dst_caption, original in iter_captions(
        resized_dir, source_dir, path_pattern, stats
    ):
        if not has_clauses(original):
            stats.skip("no-clauses")
            continue
        stats.candidates += 1
        flattened = flatten_caption(original)
        if flattened == original:
            stats.skip("unchanged")
            continue
        stats.proposed += 1
        rows.append(
            {
                "caption_path": str(rel),
                "original": original,
                # Same distinction the sweep records: ``original`` may be the
                # master, and flattening it creates a revised caption.
                "target_before": (
                    read_caption(dst_caption) if dst_caption.exists() else ""
                ),
                "proposed": flattened,
            }
        )
        if apply:
            write_caption(
                dst_caption,
                flattened,
                drop_variants=True,
                history_by="flatten",
            )
            stats.written += 1
    return rows, stats


# ---------------------------------------------------------------------------
# The stage runner
# ---------------------------------------------------------------------------

TE_NOTE = (
    "\nWritten to the resized captions (the master is untouched). Run "
    "`make preprocess-te` now to regenerate the variant sidecars and "
    "re-encode."
)


def summarize(
    rows: list[ImageProposal],
    stats: PositionCaptionStats,
    options: PositionCaptionOptions,
) -> dict[str, object]:
    """What the run itself says, for ``report.json``'s ``summary``.

    The counts, how much of the flat bag the clauses took, which rules pinned a
    tag flat, and what the part prompts recovered. The runner adds the roots it
    walked and the knobs it was given — those are the request's to report, not
    the sweep's.
    """
    return {
        "seen": stats.seen,
        "candidates": stats.candidates,
        "proposed": stats.proposed,
        "written": stats.written,
        # How much of the flat bag the clauses took. Zero under --no_rewrite.
        "rewritten": stats.rewritten,
        "moved_tags": stats.moved_tags,
        "clause_tags": stats.clause_tags,
        "novel_tags": stats.novel_tags,
        "reuse_ratio": (
            round(1.0 - stats.novel_tags / stats.clause_tags, 3)
            if stats.clause_tags
            else None
        ),
        "pinned_tags": dict(sorted(stats.pinned_tags.items(), key=lambda kv: -kv[1])),
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
        "part_prompts": list(options.part_prompts),
        # Images with at least one bound instance from a part prompt.
        "part_recovered": sum(
            1 for r in rows if any(i.source != "subject" for i in r.instances)
        ),
        "max_tokens": max(
            (r.tokens for r in rows if r.tokens is not None), default=None
        ),
    }


def _run_flatten(req: PositionRequest, src: Path, dst: Path, report_dir: Path):
    """The inverse pass — text only, so it short-circuits before any model load."""
    rows, stats = flatten_captions(
        resized_dir=dst, source_dir=src, path_pattern=req.path_pattern, apply=req.apply
    )
    summary = {
        "mode": "flatten",
        "applied": bool(req.apply),
        "seen": stats.seen,
        "with_clauses": stats.candidates,
        "flattened": stats.proposed,
        "written": stats.written,
        "skipped": dict(sorted(stats.skipped.items(), key=lambda kv: -kv[1])),
    }
    # Its own name, not ``report.json``: replaying a flatten would write the
    # clauses back.
    write_json(report_dir / "flatten_report.json", {"summary": summary, "images": rows})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_dir / 'flatten_report.json'}")
    print_dry_run_footer(req.apply, TE_NOTE)
    return rows, stats


def run_position(req: PositionRequest):
    """Detect → order → crop → tag → compose over the resized tree, or the
    ``flatten`` / ``from_report`` text-only passes. Returns ``(rows, stats)``."""
    from anime_tools.stages.replay import run_replay_cli

    src = resolve_path(req.src)
    dst = resolve_path(req.dst)
    report_dir = resolve_path(req.report_dir)

    if req.flatten:
        return _run_flatten(req, src, dst, report_dir)
    if req.from_report:
        # ``drop_variants`` mirrors the stage's own write: a stale
        # ``{stem}.variants.txt`` outranks ``{stem}.txt`` at encode time.
        rows, stats, _ = run_replay_cli(
            req,
            spec=REPLAY_SHAPES["position"],
            src=src,
            dst=dst,
            report_dir=report_dir,
            after_write_note=TE_NOTE,
        )
        return rows, stats

    from anime_tools.masking._prompts import prompt_embed_sha256, resolve_prompt_embed
    from anime_tools.stages._models import load_tagger
    from anime_tools.stages.detector import build_detect_fn

    # Deferred, and only here: the audit phase's module imports this one.
    from anime_tools.stages.multiview_audit import run_audit_phase

    # Both stay resident: the pipeline is per-image (detect -> crop -> tag), not
    # two dataset-wide passes.
    detect_fn, part_detect_fn, sam_model, sam_processor = build_detect_fn(
        req.detection, device=req.device
    )
    tagger, vocabulary, _ckpt_dir = load_tagger(req)

    token_count_fn = None
    if req.qwen3:
        from anime_tools.captions.tokenizers import load_qwen3_tokenizer_from_dir

        tokenizer = load_qwen3_tokenizer_from_dir(req.qwen3)

        def token_count_fn(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    # Phase 1: the audit, over the captions phase 2 rejects as single-subject.
    # BEFORE the sweep, not after: `multiple views` is what promotes an image out
    # of that rejection and what arms the view-invariant gate, so a tag written
    # afterwards would need a second position run to do any work.
    _audit_rows, audit_stats, promoted = run_audit_phase(
        req,
        src=src,
        dst=dst,
        report_dir=report_dir,
        detect_fn=detect_fn,
        part_detect_fn=part_detect_fn,
        tagger=tagger,
        vocabulary=vocabulary,
    )

    options = req.options()
    rows, stats = run_position_captions(
        resized_dir=dst,
        source_dir=src,
        detect_fn=detect_fn,
        part_detect_fn=part_detect_fn,
        tag_fn=tagger.predict,
        vocabulary=vocabulary,
        options=options,
        path_pattern=req.path_pattern,
        apply=req.apply,
        crops_dir=(report_dir / "crops") if req.crops else None,
        token_count_fn=token_count_fn,
        progress=make_progress(200),
        promoted=promoted,
        analysis_dir=report_dir / ANALYSIS_SUBDIR,
    )
    del sam_processor, sam_model

    over_budget = [
        r for r in rows if r.tokens is not None and r.tokens > req.max_tokens
    ]
    embed_path = resolve_prompt_embed(req.detection.prompt_embed)
    summary = {
        **stage_report_header(
            src=src, dst=dst, path_pattern=req.path_pattern, apply=req.apply
        ),
        "rewrite": bool(req.rewrite),
        # A soft prompt is a file: two runs only compare when the sha matches.
        "prompt": req.detection.prompt,
        "prompt_embed": str(embed_path) if embed_path else None,
        "prompt_embed_sha256": prompt_embed_sha256(embed_path),
        "attribution_margin": req.attribution_margin,
        "max_novel_tags": req.max_novel_tags,
        **summarize(rows, stats, options),
        # The audit phase, when it ran. ``promoted`` is what phase 1 handed to
        # phase 2; ``promoted_written`` is the tail phase 2 could not turn into
        # clauses and wrote for the tag alone.
        "multiview_audit": (
            {
                "mode": req.multiview_audit,
                "audited": audit_stats.audited,
                "findings": audit_stats.findings,
                "verdicts": dict(
                    sorted(audit_stats.verdicts.items(), key=lambda kv: -kv[1])
                ),
                "promoted": stats.promoted,
                "promoted_written": stats.promoted_written,
                "report": str(report_dir / "audit" / "audit_report.json"),
            }
            if audit_stats is not None
            else None
        ),
        "over_token_budget": [r.image for r in over_budget],
    }
    report_path = write_stage_report(
        report_dir, {"summary": summary, "images": [asdict(r) for r in rows]}
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nreport: {report_path}")
    if over_budget:
        print(
            f"WARNING: {len(over_budget)} caption(s) exceed {req.max_tokens} tokens — "
            "the tail truncates silently at TE-cache time."
        )
    if audit_stats is not None:
        print(
            f"multiview audit: {audit_stats.findings} finding(s) over "
            f"{audit_stats.audited} single-subject caption(s); "
            f"{stats.promoted} promoted into this run's sweep. "
            f"sheets: {report_dir / 'audit' / 'sheets'}"
        )
        if req.audits and not req.promotes and audit_stats.findings:
            print(
                "  --multiview_audit=report tags nothing. Re-run with "
                "`--multiview_audit apply` to feed the findings into the sweep."
            )
    print_dry_run_footer(req.apply, TE_NOTE)
    if req.apply and req.rewrite and stats.moved_tags:
        print(
            f"{stats.moved_tags} tag(s) moved out of the flat bag across "
            f"{stats.rewritten} caption(s). To back that out: "
            '`make caption-position ARGS="--flatten --apply"`.'
        )
    return rows, stats
