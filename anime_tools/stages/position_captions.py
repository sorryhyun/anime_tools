"""Position-aware caption rewrite — detect subjects, bind tags to sides.

Detect the ``girl`` instances in a multi-subject image, order them, tag each
mask-blanked crop, and rewrite the caption as
``<flat tag bag>. On the left, akita neru, yellow eyes. On the right, ...``.
An attributable tag *moves* out of the flat bag into its clause, so each
attribute is asserted exactly once; ``rewrite=False`` keeps the additive v1
behaviour. Reversible via :func:`flatten_captions`.

Takes its two models as injected callables (``detect_fn``/``tag_fn``), staying
import-free of SAM3/the tagger; ``stages/cli/position_captions.py`` owns
argparse + model loading.

Per-rule evidence and the knob table live in
``docs/position_captions.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

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

from ._caption_io import write_caption
from ._walk_captions import iter_captions

# Convenience re-exports: canonical homes are the modules imported above, but
# every consumer reaches for them on this module. Listed here so they read as
# the public surface rather than dead imports.
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
    "run_position_captions",
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

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def pin(self, reason: str) -> None:
        self.pinned_tags[reason] = self.pinned_tags.get(reason, 0) + 1


@dataclass(frozen=True)
class PositionCaptionOptions:
    """Knobs for one pass. Defaults are the shipped recipe."""

    prompt: str = "girl"
    score_threshold: float = 0.5
    retry_score_threshold: float = 0.35
    # Body-part fallback: extra SAM3 prompts run only when the subject prompt
    # undershoots. Off by default (empty tuple) — see ``merge_part_detections``.
    part_prompts: tuple[str, ...] = ()
    part_score_threshold: float = 0.5
    part_containment_threshold: float = 0.7
    iou_threshold: float = 0.65
    # Off by default — see ``box_containment``.
    containment_threshold: float = 1.01
    # On by default, unlike its box counterpart — see ``mask_containment``. Two
    # boxes nest identically whether the inner one is a fragment or a second
    # girl in front of the first; their masks do not.
    mask_containment_threshold: float = 0.8
    # Mask-quality tie-break inside an NMS-matched pair — see
    # ``dedupe_detections``; 0 disables (score-only survivor).
    dedupe_fill_ratio: float = 2.0
    min_area_frac: float = 0.005
    pad: float = 0.06
    blank_crops: bool = True
    row_tol: float = 0.25
    max_clause_tags: int = 8
    # How many tags a clause may introduce that the caption never contained;
    # the rest fills from the flat bag first, since only a bag tag can *move*.
    max_novel_tags: int = 1
    name_confidence: float = 0.5
    allow_unlisted_names: bool = False
    min_instances: int = 2
    max_instances: int = 8
    strict_count: bool = True
    discriminative_only: bool = True
    bag_gated_identity: bool = True
    # On a repeated-subject layout (``multiple views`` / comic panels), keep the
    # character's own traits — and her name — out of every clause: they belong
    # to the girl, not to a view of her.
    multi_view_gate: bool = True
    # Let a clause say which *view* it describes (`close-up`, `full body`).
    bind_framing: bool = True
    # Let a view layout's clause carry the anatomy visible in that panel.
    bind_view_anatomy: bool = True
    # Bag-tag keep relaxation (1.0 = off): a bag tag can only MOVE into a
    # clause, never be invented, so the crop tagger only has to *localize* it
    # and its per-tag F1 threshold may be relaxed for that population — which
    # recovers pose tags whose scores collapse once mask-blanking removes the
    # scene context. Applied before the attributable/shared census, so a rival
    # crop's borderline score also BLOCKS a move the strict kept sets allowed.
    bag_relax: float = 0.35
    # Extra relaxation per word beyond the first (compounds with ``bag_relax``):
    # a more specific tag is less likely to clear on noise. 1.0 = off.
    bag_word_relax: float = 0.85
    # Raw-score floor under the relaxation, which can otherwise drag a 2-word
    # tag to ~0.16× of its threshold. Only the relax path is floored. 0 = off.
    bag_relax_min_score: float = 0.3
    # Move an attributable tag out of the flat bag into its clause. False is the
    # additive v1 behaviour (bag untouched), kept for the training A/B.
    rewrite: bool = True
    # How far the winning crop must clear every other, relative to its own
    # probability (``1 - rival/winner``), before a tag may leave the bag. Gates
    # only the removal — a tag that fails still enters its clause, degrading to
    # v1 for it. 0.0 = trust the tagger's thresholds alone.
    attribution_margin: float = 0.25


def detect_subjects(
    image: Image.Image,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    options: PositionCaptionOptions,
    expected: int | None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
) -> list[Detection]:
    """Detect + dedupe, with two escalations when the count falls short.

    Both fire only when detection undershoots the expected count (on a resolved
    image they'd only add duplicates): a lower score threshold, which recovers
    an extreme close-up, then body-part prompts (``part_detect_fn``), which
    recover a headless panel the subject prompt can't see at any threshold.

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
    # Top up to the target, no further — a part prompt is a looser concept than
    # ``girl`` and can fragment into more boxes than there are real panels.
    return merged[: max(target, len(dets))]


def _relax_bag_keeps(
    kept_sets: list[dict[str, float]],
    score_sets: list[dict[str, float]],
    predictions: list[Mapping[str, object]],
    flat_bag: frozenset[str],
    options: PositionCaptionOptions,
) -> None:
    """Admit sub-threshold flat-bag tags into each crop's kept set, in place.

    See ``bag_relax`` on :class:`PositionCaptionOptions` for why. Needs the
    per-tag thresholds ``AnimaTagger.predict`` attaches; a no-op per crop when a
    stub ``tag_fn`` omits them.
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
) -> ImageProposal:
    """Build the clause proposal for one image. Never writes any caption."""
    parsed = parse_caption(caption)
    flat_bag = parsed.tag_keys
    expected = caption_subject_count(caption)

    proposal = ImageProposal(
        image="",
        caption_path="",
        status="proposed",
        expected=expected,
        original=caption,
    )

    dets = detect_subjects(image, detect_fn, options, expected, part_detect_fn)
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
    # can't ground. "Agree" is a range because the ``girl`` prompt picks up
    # males inconsistently: girls..girls+boys are all consistent.
    if options.strict_count and expected:
        boys = caption_boy_count(caption)
        upper = None if boys is None else expected + boys
        if len(dets) < expected or (upper is not None and len(dets) > upper):
            proposal.status = "skip:count-mismatch"
            return proposal
    # A layout tag waives the check above (``expected`` is None by design); an
    # ``Nkoma`` tag restores a generous ceiling so a subject detected twice
    # still has a backstop.
    if options.strict_count and not expected:
        ceiling = caption_panel_ceiling(caption)
        if ceiling is not None and len(dets) > ceiling:
            proposal.status = "skip:count-mismatch"
            return proposal

    order = ordered_indices([d.box for d in dets], image.size, row_tol=options.row_tol)
    dets = [dets[i] for i in order]
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
    view_invariant = options.multi_view_gate and is_repeated_subject_layout(caption)

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
                # ``flat_bag`` is ``parsed.tag_keys``, so the probe has to be the
                # same key — a ``lower()`` one never matched an underscored tag.
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
        # The rewrite removes text, so an emptied bag is asserted against
        # rather than assumed impossible.
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
    """Save each crop under ``crops_dir`` mirroring the dataset layout.

    The dry-run review artifact: a proposed clause next to the exact pixels the
    tagger saw is the only way to tell a detection miss from a tagging miss.
    """
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

    A skip produces no crops, so the overlay is the reviewer's only evidence for
    the rows they must adjudicate: over-detection, missing subject, or wrong
    caption count?
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
) -> tuple[list[ImageProposal], PositionCaptionStats]:
    """Walk the resized tree, propose clauses, and (with ``apply``) write them.

    GOTCHA: the caption master (``source_dir``) is NEVER written — the rewrite
    lands at ``resized_dir/<rel>``, what the TE step encodes, and the master is
    only the read fallback for an unmirrored image. The stale
    ``{stem}.variants.txt`` sidecar, which wins over ``{stem}.txt`` at encode
    time, is dropped alongside the write, which is a rewrite rather than an
    append — recoverable (:func:`flatten_captions`) but not a no-op, hence
    ``apply`` defaults off.
    """
    options = options or PositionCaptionOptions()
    stats = PositionCaptionStats()
    rows: list[ImageProposal] = []

    walked = list(iter_captions(resized_dir, source_dir, path_pattern, stats))
    for index, (image_path, rel, dst_caption, caption) in enumerate(walked, 1):
        if progress is not None:
            progress(index, len(walked), str(rel))
        ok, reason = is_candidate(caption)
        if not ok:
            stats.skip(reason)
            continue
        stats.candidates += 1

        crop_sink = _crop_sink(crops_dir, rel) if crops_dir is not None else None
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
        )
        proposal.image = str(image_path.relative_to(resized_dir))
        proposal.caption_path = str(rel)
        rows.append(proposal)

        if not proposal.ok:
            stats.skip(proposal.status.removeprefix("skip:"))
            if crops_dir is not None:
                _save_skip_overlay(crops_dir, rel, image, proposal)
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

    Text-only (no SAM3, no tagger, no pixels) since the rewrite moves tags
    rather than deleting them. Same revised caption as
    :func:`run_position_captions`.

    GOTCHA: hand-written clauses are flattened too — the pass can't tell them
    from generated ones — which is a real loss of curation if they only ever
    existed here. Hence the dry-run default.
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
