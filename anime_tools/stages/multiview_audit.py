"""Find images that draw one character several times but are captioned ``1girl``.

Audits the complement of the clause pipeline: every image ``caption-position``
throws away as ``single-subject``. Two boxes where the caption claims one girl
is the finding; three signals then argue about what it means, and a verdict
needs two of them before ``--apply`` writes it: identity agreement across the
crops, the whole-image ``multiple views`` head, and the people-count head
insisting on ``1girl``.

Detection forces its escalation target to ``min_instances`` (2) rather than the
caption's count, or the caption's ``expected=1`` would stop the search at the
first box — on precisely the image we are trying to catch.

Read-only apart from :func:`apply_findings`. See ``docs/multiview_audit.md``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from anime_tools.captions.caption_layout import (
    caption_boy_count,
    is_candidate,
)
from anime_tools.captions.clause_vocabulary import ClauseVocabulary
from anime_tools.captions.position_clauses import (
    assign_positions,
    compose_caption,
    flat_tag_set,
    ordered_indices,
    parse_caption,
)
from anime_tools.captions.taxonomy import count_of, exact_count, normalize_tag

# Re-exported: the verdict names and the two witness floors are request defaults,
# so they live in the leaf the schema build reads (:mod:`stages._options`).
from anime_tools.stages._options import (
    DEFAULT_IDENTITY_CONFIDENCE,
    DEFAULT_MULTIVIEW_PROB,
    EXTRA_CHARACTER,
    MULTIPLE_VIEWS,
)
from anime_tools.stages.instance_detection import Detection, crop_instance

from ._analysis import clear_analysis, write_analysis
from ._caption_io import read_caption
from ._walk_captions import iter_captions
from .position_captions import PositionCaptionOptions, detect_subjects
from .replay import apply_one, undo_one

# The audit population, named by `is_candidate`'s own reason string.
AUDIT_SKIP_REASON = "single-subject"

UNSURE = "unsure"
# Not a finding: the caption's own girls+boys counts already cover every box.
COUNT_EXPLAINED = "count-explained"

# How much identity evidence must agree before two boxes are called the same
# character: "all comparable groups but one", so a single misread eye colour on
# a mask-blanked crop doesn't split a genuine view pair.
_SAME_CHARACTER_AGREEMENT = 0.66
# ...and how little forces the opposite call. Between the two lies `unsure`,
# which the report keeps but never proposes an edit for.
_DIFFERENT_CHARACTER_AGREEMENT = 0.34
# Groups needed before the agreement ratio is trusted: two crops agreeing only
# on `brown hair` says nothing.
_MIN_COMPARABLE_GROUPS = 2
# Which detector produced the finding.
SOURCE_DETECTION = "detection"
SOURCE_TAGGER = "tagger-only"


@dataclass
class CropIdentity:
    """What the tagger read off one crop, reduced to the identity trio + name.

    GOTCHA: ``groups`` holds only the values that cleared
    ``DEFAULT_IDENTITY_CONFIDENCE`` **on the raw probability**. The group heads
    are an argmax over a softmax — they name a hair colour for a headless crop
    as confidently as for a portrait, and they land in ``kept`` by that argmax,
    so filtering on ``kept`` membership does nothing.
    """

    box: tuple[float, float, float, float]
    score: float
    position: str = ""
    source: str = "subject"
    # Cleared the detector's own confidence floor, rather than being recovered
    # by the retry escalation. Only a reliable box's identity gets a vote.
    reliable: bool = True
    groups: dict[str, str] = field(default_factory=dict)
    # What the ungated argmax said, and how sure it was.
    raw_groups: dict[str, str] = field(default_factory=dict)
    group_scores: dict[str, float] = field(default_factory=dict)
    name: str | None = None


@dataclass
class MultiviewFinding:
    """One audited image that looks under-counted."""

    image: str
    caption_path: str
    instances: int
    girls: int | None
    boys: int | None
    # Whether the geometry or the whole-image tagger raised this one.
    source: str = SOURCE_DETECTION
    # Whole-image corroboration, independent of the geometry.
    tagger_multiple_views: float | None = None
    people_count: str | None = None
    identity_agreement: float | None = None
    comparable_groups: int = 0
    crops: list[CropIdentity] = field(default_factory=list)
    verdict: str = UNSURE
    witnesses: list[str] = field(default_factory=list)
    confidence: str = "weak"
    suggested_tag: str | None = None
    caption: str = ""
    """The caption audited — the revised one, or the master when there is no
    revised caption yet."""
    target_before: str = ""
    """What the write target (the revised caption) holds right now, ``""`` when
    it does not exist. The drift baseline for :func:`apply_findings` and for the
    replay, which is not ``caption``: tagging a master writes a revised caption
    that was never there, and the undo of that write is a delete."""
    proposed: str = ""
    # Contact sheet for this row, relative to the sheets dir; "" when off.
    sheet: str = ""


@dataclass
class MultiviewAuditStats:
    seen: int = 0
    audited: int = 0
    findings: int = 0
    skipped: Counter = field(default_factory=Counter)
    verdicts: Counter = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1


def is_audit_target(caption: str) -> tuple[bool, str]:
    """Is this one of the captions ``caption-position`` never looks at?

    True exactly when :func:`is_candidate` rejects it as ``single-subject``. A
    girls-count of 0 stays in — two subjects on a ``1boy`` caption is a caption
    bug too.
    """
    ok, reason = is_candidate(caption)
    if ok:
        return False, f"handled-by-position-captions:{reason}"
    if reason != AUDIT_SKIP_REASON:
        return False, reason
    return True, AUDIT_SKIP_REASON


def _girls_count(caption: str) -> int | None:
    """The exact ``Ngirls`` the caption claims, or ``None`` if it claims none.

    Unlike :func:`~anime_tools.captions.taxonomy.count_of`, "unknown" (a bare
    ``multiple girls``) and "absent" collapse to one ``None``: both mean there
    is no number to compare against.
    """
    return count_of(flat_tag_set(caption), "girl") or None


def identity_agreement(
    crops: Sequence[CropIdentity],
    identity_groups: Collection[str],
) -> tuple[float | None, int]:
    """Fraction of identity groups on which every crop agrees, and how many voted.

    ``identity_groups`` is the caller's own vocabulary
    (``vocabulary.clause_groups.identity``), passed in rather than re-read from
    :func:`default_clause_groups`: a run loaded from a custom
    ``configs/clause_vocabulary.yaml`` must score on the groups it declared, and
    ``audit_image`` already collects each crop's values against exactly this set.

    Two things are excluded from the vote:

    * A crop the detector was **not confident about** (``reliable=False``). Its
      mask comes out shredded, and mask-blanking hands the tagger a near-white
      canvas it reads a hair colour off *at 0.99*, which the probability gate
      cannot catch. Detection score separates these; mask fill does not
      (``docs/position_captions.md``).
    * Any group not *every* surviving crop resolved — a headless panel reports
      no eye colour, and scoring that as disagreement would call every close-up
      a second character.
    """
    usable = [c for c in crops if c.reliable]
    if len(usable) < 2:
        return None, 0
    agree = 0
    comparable = 0
    for group in sorted(identity_groups):
        values = [c.groups.get(group) for c in usable]
        if any(v is None for v in values):
            continue
        comparable += 1
        if len(set(values)) == 1:
            agree += 1
    if not comparable:
        return None, 0
    return agree / comparable, comparable


def _verdict(
    crops: Sequence[CropIdentity],
    multiview_prob: float | None,
    multiview_threshold: float,
    identity_groups: Collection[str],
) -> tuple[str, float | None, int]:
    """Same character in every box, or a character the caption never counted?

    Identity first, as the only signal that distinguishes the two; the
    whole-image ``multiple views`` head is the fallback, and its absence leaves
    ``unsure``.
    """
    agreement, comparable = identity_agreement(crops, identity_groups)
    names = {c.name for c in crops if c.name and c.reliable}
    # Decisive on its own: two girls can share brown hair, not a character name.
    if len(names) > 1:
        return EXTRA_CHARACTER, agreement, comparable
    if agreement is not None and comparable >= _MIN_COMPARABLE_GROUPS:
        if agreement >= _SAME_CHARACTER_AGREEMENT:
            return MULTIPLE_VIEWS, agreement, comparable
        if agreement <= _DIFFERENT_CHARACTER_AGREEMENT:
            return EXTRA_CHARACTER, agreement, comparable
        return UNSURE, agreement, comparable
    if (multiview_prob or 0.0) >= multiview_threshold:
        return MULTIPLE_VIEWS, agreement, comparable
    return UNSURE, agreement, comparable


def suggest_tag(
    verdict: str,
    instances: int,
    girls: int | None,
    *,
    suggest_counts: bool = False,
) -> str | None:
    """The tag the caption is missing, or ``None`` when we won't guess.

    ``multiple views`` is count-agnostic and always safe to propose. The
    extra-character case is **report-only by default**: detection counts
    *subjects*, not girls, so an ``Ngirls`` proposal is wrong whenever the extra
    body is a boy. Opt in with ``suggest_counts``.
    """
    if verdict == MULTIPLE_VIEWS:
        return MULTIPLE_VIEWS
    if (
        suggest_counts
        and verdict == EXTRA_CHARACTER
        and girls is not None
        and instances > girls
    ):
        return f"{instances}girls"
    return None


def propose_caption(caption: str, tag: str) -> str:
    """Append ``tag`` to the end of the flat bag.

    End of the *bag*, not of the text: position clauses have to stay trailing or
    the grammar stops parsing them, so the splice goes through
    :func:`compose_caption`. An ``Ngirls`` suggestion instead *replaces* the
    stale count in place, since two girls-counts contradict.

    "Already there?" is :func:`normalize_tag` — the suggestion is spelled the
    tagger's way and the bag may be spelled the master's.
    """
    parsed = parse_caption(caption)
    flat = [t for t in parsed.flat_tags if t.strip()]
    keys = [normalize_tag(t) for t in flat]
    if normalize_tag(tag) in keys:
        return caption.strip()
    count_at = next(
        (i for i, t in enumerate(keys) if exact_count(t, "girl") is not None),
        None,
    )
    if exact_count(tag, "girl") is not None and count_at is not None:
        flat[count_at] = tag
    else:
        flat.append(tag)
    return compose_caption(flat, parsed.clauses)


def audit_image(
    image: Image.Image,
    caption: str,
    *,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
    crop_sink: Callable[[int, str, Image.Image], str] | None = None,
    multiview_threshold: float = DEFAULT_MULTIVIEW_PROB,
    identity_confidence: float = DEFAULT_IDENTITY_CONFIDENCE,
    suggest_counts: bool = False,
    mask_sink: Callable[[list[Detection]], None] | None = None,
) -> MultiviewFinding:
    """Detect, and when more than one subject lands, ask the tagger who they are.

    An image with nothing to report comes back ``UNSURE`` with ``instances < 2``;
    the caller filters. ``mask_sink`` sees the reading-ordered detections, the
    order ``crops`` is in.
    """
    girls = _girls_count(caption)
    finding = MultiviewFinding(
        image="",
        caption_path="",
        instances=0,
        girls=girls,
        boys=caption_boy_count(caption),
        caption=caption.strip(),
    )

    # Unconditional, before the geometry decides anything: the only thing that
    # can flag a sheet SAM merged into a single box.
    whole = tag_fn(image)
    scores = whole.get("scores") or {}
    finding.tagger_multiple_views = float(scores.get(MULTIPLE_VIEWS, 0.0))
    people = whole.get("people_count")
    finding.people_count = str(people) if people is not None else None

    # expected=None, NOT caption_subject_count(): a caption claiming one girl
    # would satisfy the target on the first box and suppress both escalations.
    dets = detect_subjects(image, detect_fn, options, None, part_detect_fn)
    order = ordered_indices([d.box for d in dets], image.size, row_tol=options.row_tol)
    dets = [dets[i] for i in order]
    finding.instances = len(dets)
    if mask_sink is not None:
        mask_sink(dets)

    # A caption whose counts already cover every detected body is fine:
    # `1girl, 1boy` lands here only because the *girls*-count is one, and the
    # `girl` prompt does not exclude males. An unknown boy count is unbounded.
    counted = None if finding.boys is None else (girls or 0) + finding.boys
    explained = counted is None or counted >= len(dets)

    if explained or len(dets) < max(2, options.min_instances):
        # The tagger alone can still raise the image, but it is a single
        # witness by construction, so it never rises above `weak`.
        if finding.tagger_multiple_views >= multiview_threshold:
            finding.source = SOURCE_TAGGER
            finding.verdict = MULTIPLE_VIEWS
            finding.witnesses = ["tagger-multiple-views"]
            finding.suggested_tag = MULTIPLE_VIEWS
            finding.proposed = propose_caption(caption, MULTIPLE_VIEWS)
        elif explained:
            finding.verdict = COUNT_EXPLAINED
        return finding

    positions = assign_positions(
        [d.box for d in dets], image.size, row_tol=options.row_tol
    )
    for index, det in enumerate(dets[: options.max_instances]):
        # Part boxes take the plain padded bbox: on a part detection the mask IS
        # the part, so blanking would delete the very pixels it recovered.
        crop = crop_instance(
            image,
            det,
            pad=options.pad,
            blank=options.blank_crops and det.source == "subject",
        )
        if crop_sink is not None:
            crop_sink(index, positions[index] or "crop", crop)
        out = tag_fn(crop)
        raw = {
            g: v
            for g, v in (out.get("groups") or {}).items()
            if g in vocabulary.clause_groups.identity and v
        }
        kept = out.get("kept") or {}
        crop_scores = out.get("scores") or {}
        name = max(
            (t for t in kept if t in vocabulary.characters),
            key=lambda t: kept[t],
            default=None,
        )
        if name is not None and kept[name] < options.name_confidence:
            name = None
        finding.crops.append(
            CropIdentity(
                box=det.box,
                score=det.score,
                position=positions[index],
                source=det.source,
                reliable=det.score >= options.score_threshold,
                # Gated on the raw probability, not `kept` membership.
                groups={
                    g: v
                    for g, v in raw.items()
                    if float(crop_scores.get(v, 0.0)) >= identity_confidence
                },
                raw_groups=raw,
                group_scores={
                    g: round(float(crop_scores.get(v, 0.0)), 4) for g, v in raw.items()
                },
                name=name,
            )
        )

    verdict, agreement, comparable = _verdict(
        finding.crops,
        finding.tagger_multiple_views,
        multiview_threshold,
        vocabulary.clause_groups.identity,
    )
    finding.verdict = verdict
    finding.identity_agreement = agreement
    finding.comparable_groups = comparable
    finding.suggested_tag = suggest_tag(
        verdict, finding.instances, girls, suggest_counts=suggest_counts
    )
    if finding.suggested_tag:
        finding.proposed = propose_caption(caption, finding.suggested_tag)

    # Independent witnesses to "one girl drawn twice": `strong` means at least
    # two of the three agreed, which is the tier `--apply` writes.
    witnesses: list[str] = []
    if comparable >= _MIN_COMPARABLE_GROUPS and (agreement or 0.0) >= (
        _SAME_CHARACTER_AGREEMENT
    ):
        witnesses.append("identity-agreement")
    if (finding.tagger_multiple_views or 0.0) >= multiview_threshold:
        witnesses.append("tagger-multiple-views")
    if finding.people_count == "1girl":
        witnesses.append("people-count-1girl")
    if verdict == EXTRA_CHARACTER:
        # Different question: an extra character is made credible by the
        # identity evidence splitting, not by the view head firing.
        witnesses = []
        if len({c.name for c in finding.crops if c.name and c.reliable}) > 1:
            witnesses.append("distinct-names")
        if comparable >= _MIN_COMPARABLE_GROUPS and (agreement or 0.0) <= (
            _DIFFERENT_CHARACTER_AGREEMENT
        ):
            witnesses.append("identity-disagreement")
        if finding.people_count not in (None, "1girl", "no_people"):
            witnesses.append(f"people-count-{finding.people_count}")
    finding.witnesses = witnesses
    finding.confidence = "strong" if len(witnesses) >= 2 else "weak"
    return finding


def run_multiview_audit(
    *,
    resized_dir: Path,
    source_dir: Path,
    detect_fn: Callable[[Image.Image, float], list[Detection]],
    tag_fn: Callable[[Image.Image], Mapping[str, object]],
    vocabulary: ClauseVocabulary,
    options: PositionCaptionOptions | None = None,
    path_pattern: str | None = None,
    crops_dir: Path | None = None,
    sheets_dir: Path | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    part_detect_fn: Callable[[Image.Image, str, float], list[Detection]] | None = None,
    multiview_threshold: float = DEFAULT_MULTIVIEW_PROB,
    identity_confidence: float = DEFAULT_IDENTITY_CONFIDENCE,
    suggest_counts: bool = False,
    analysis_dir: Path | None = None,
) -> tuple[list[MultiviewFinding], MultiviewAuditStats]:
    """Walk the resized tree and report every under-counted caption.

    ``caption_path`` is reported relative so the caller can decide which tree to
    edit. No caption is written; ``analysis_dir`` keeps each finding and its
    instance masks (:mod:`~anime_tools.stages._analysis`), and clears the pair
    of an image audited without one.
    """
    options = options or PositionCaptionOptions()
    stats = MultiviewAuditStats()
    rows: list[MultiviewFinding] = []

    for image_path, rel, dst_caption, caption in iter_captions(
        resized_dir, source_dir, path_pattern, stats, progress
    ):
        image_rel = image_path.relative_to(resized_dir).as_posix()
        ok, reason = is_audit_target(caption)
        if not ok:
            clear_analysis(analysis_dir, image_rel)
            stats.skip(reason)
            continue
        stats.audited += 1

        # The sheet needs the exact crops the tagger read, teed into memory as
        # produced: a plain bbox re-crop shows different pixels than were scored.
        held: list[Image.Image] = []
        save_crop = None
        if crops_dir is not None:
            from .position_captions import _crop_sink

            save_crop = _crop_sink(crops_dir, rel)

        # ``held``/``save_crop`` are rebuilt per image, so they are bound as
        # defaults rather than closed over the loop variables.
        def crop_sink(
            i: int,
            pos: str,
            crop: Image.Image,
            _held: list[Image.Image] = held,
            _save=save_crop,
        ) -> str:
            if sheets_dir is not None:
                _held.append(crop.copy())
            return _save(i, pos, crop) if _save is not None else ""

        labelled: list[Detection] = []

        def hold_labels(dets: list[Detection], _held=labelled) -> None:
            _held[:] = dets

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        finding = audit_image(
            image,
            caption,
            detect_fn=detect_fn,
            tag_fn=tag_fn,
            vocabulary=vocabulary,
            options=options,
            part_detect_fn=part_detect_fn,
            crop_sink=crop_sink,
            multiview_threshold=multiview_threshold,
            identity_confidence=identity_confidence,
            suggest_counts=suggest_counts,
            mask_sink=hold_labels if analysis_dir is not None else None,
        )
        # Only the tagger can raise an image the geometry had nothing to say
        # about: too few boxes, or a caption that already counts them all.
        if finding.source != SOURCE_TAGGER:
            if finding.verdict == COUNT_EXPLAINED:
                clear_analysis(analysis_dir, image_rel)
                stats.skip(COUNT_EXPLAINED)
                continue
            if finding.instances < max(2, options.min_instances):
                clear_analysis(analysis_dir, image_rel)
                stats.skip("single-instance")
                continue
        # Both rels go into audit_report.json and come back as dict keys and
        # `resized_dir / …` joins, so they are spelled posix like every other
        # rel the package persists (see `exclude.rel_key`). `str()` on a
        # Windows path would key the report on backslashes, and the sweep
        # that looks a promotion up by rel would miss every row.
        finding.image = image_path.relative_to(resized_dir).as_posix()
        finding.caption_path = rel.as_posix()
        finding.target_before = (
            read_caption(dst_caption) if dst_caption.exists() else ""
        )
        stats.findings += 1
        stats.verdicts[finding.verdict] += 1
        rows.append(finding)

        if sheets_dir is not None:
            from .multiview_sheet import render_contact_sheet, sheet_path

            target = sheet_path(sheets_dir, finding)
            target.parent.mkdir(parents=True, exist_ok=True)
            render_contact_sheet(image, finding, held).save(target)
            finding.sheet = str(target.relative_to(sheets_dir))
        if analysis_dir is not None:
            record = asdict(finding)
            record["labels"] = "crops"
            write_analysis(analysis_dir, image_rel, record, labelled, image.size)

    return rows, stats


def admitted(
    findings: Iterable[MultiviewFinding],
    *,
    verdicts: Sequence[str] = (MULTIPLE_VIEWS,),
    confidences: Sequence[str] = ("strong",),
) -> list[MultiviewFinding]:
    """The findings a write — or a promotion — is allowed to act on.

    One predicate for both callers: :func:`apply_findings`, which writes them to
    disk, and :func:`promotions`, which hands them to the clause sweep in
    memory, so the two cannot disagree about what is actionable. An empty
    ``proposed`` still passes here — :func:`~anime_tools.stages.replay.apply_one`
    reports that as ``no-proposal``, which the report counts.
    """
    return [
        f for f in findings if f.verdict in verdicts and f.confidence in confidences
    ]


def promotions(
    findings: Iterable[MultiviewFinding],
    *,
    verdicts: Sequence[str] = (MULTIPLE_VIEWS,),
    confidences: Sequence[str] = ("strong",),
) -> dict[str, str]:
    """``{caption_path: proposed}`` for every admitted finding — the audit's
    verdict as the clause sweep reads it, with no file in between.

    This is what makes audit-then-position one run rather than two: the sweep
    consults this map before :func:`~anime_tools.captions.caption_layout.is_candidate`,
    so an image the audit just called ``multiple views`` is promoted out of the
    ``single-subject`` rejection and arms the view-invariant gate in the same
    pass. Nothing here touches disk, so a dry run reports exactly the plan an
    ``--apply`` would carry out.
    """
    return {
        f.caption_path: f.proposed
        for f in admitted(findings, verdicts=verdicts, confidences=confidences)
        # An empty proposal would promote the image to an empty caption; the
        # write path lets it through only because apply_one names it and counts
        # it, and there is nothing here to count it into.
        if f.proposed
    }


def apply_findings(
    findings: Iterable[MultiviewFinding],
    *,
    resized_dir: Path,
    verdicts: Sequence[str] = (MULTIPLE_VIEWS,),
    confidences: Sequence[str] = ("strong",),
) -> tuple[list[tuple[str, str, str]], Counter]:
    """Write proposed captions into the **revised** tree.

    Where every other stage writes, and the caption master is never touched: it
    is hand-written, and what the audit proposes is a machine verdict off a few
    crops. It is also the only tree the write reaches — ``resolve_caption`` is
    revised-first, so a caption written to the master would be read past by the
    clause sweep, the correction pass and the TE step alike.

    The drift baseline is ``target_before`` — what that revised caption holds —
    not the ``caption`` audited, which is the master for an image nobody has
    revised yet. Those are the images the write *creates* a caption for, and
    against ``caption`` every one of them read as ``missing-caption``.

    Returns ``(written, skipped)``: the ``(rel, before, after)`` triples written
    and a count per :func:`~anime_tools.stages.replay.apply_one` status for the
    gated rows that were not, so a caption edited since the audit is ``drifted``
    and left alone. The replaced text goes onto the ``.history.txt`` sidecar.
    """
    written: list[tuple[str, str, str]] = []
    skipped: Counter = Counter()
    for finding in admitted(findings, verdicts=verdicts, confidences=confidences):
        status = apply_one(
            resized_dir / finding.caption_path,
            finding.target_before,
            finding.proposed,
            apply=True,
            drop_variants=True,
            history_by="audit",
        )
        if status == "written":
            written.append(
                (finding.caption_path, finding.target_before.strip(), finding.proposed)
            )
        else:
            skipped[status] += 1
    return written, skipped


def curated_proposal(row: dict) -> tuple[str, str] | None:
    """(tag, proposed caption) for a hand-accepted report row.

    An accepted ``unsure`` becomes ``multiple views``; an accepted
    ``extra-character`` becomes ``{instances}girls``, replacing the stale count.
    ``None`` when no edit applies.
    """
    caption = row["caption"]
    if row["verdict"] == EXTRA_CHARACTER:
        girls, instances = row.get("girls"), row.get("instances") or 0
        if girls is None or instances <= girls:
            return None
        tag = f"{instances}girls"
    else:
        tag = MULTIPLE_VIEWS
    proposed = propose_caption(caption, tag)
    return None if proposed == caption.strip() else (tag, proposed)


# :func:`~anime_tools.stages.replay.undo_one`'s statuses, named from the revert
# direction. ``removed`` keeps its own name: the caption the apply created is
# gone, not restored to an earlier text.
_REVERT_STATUS = {
    "written": "reverted",
    "would-write": "would-revert",
    "already-applied": "already-reverted",
}


def apply_curated(
    rows: Iterable[dict],
    accepted: set[str],
    *,
    resized_dir: Path,
    apply: bool,
) -> tuple[list[dict], list[str]]:
    """Apply a reviewer-curated accept list of report rows to the revised captions.

    Returns ``(manifest, unmatched)``: one entry per accepted row with the
    verbatim before/after and any accepted image with no finding. Same drift
    guard as :func:`apply_findings`, and the same baseline: ``target_before``
    when the report records one, and the audited ``caption`` for a report
    written before it did.
    """
    by_image = {r["image"]: r for r in rows if r.get("verdict")}
    unmatched = sorted(accepted - set(by_image))
    manifest: list[dict] = []
    for image in sorted(accepted & set(by_image)):
        row = by_image[image]
        before = row["target_before"] if "target_before" in row else row["caption"]
        entry = {
            "image": image,
            "caption_path": row["caption_path"],
            "verdict": row["verdict"],
            "confidence": row["confidence"],
            "tag": None,
            "before": before,
            "after": None,
            "status": "pending",
        }
        manifest.append(entry)
        derived = curated_proposal(row)
        if derived is None:
            entry["status"] = "no-edit"
            continue
        entry["tag"], entry["after"] = derived
        entry["status"] = apply_one(
            resized_dir / row["caption_path"],
            before,
            entry["after"],
            apply=apply,
            drop_variants=True,
            history_by="audit",
        )
    return manifest, unmatched


def revert_curated(
    manifest: Iterable[dict],
    *,
    resized_dir: Path,
    apply: bool,
) -> list[dict]:
    """Undo :func:`apply_curated` from its manifest, restoring ``before`` only
    where the revised caption still holds exactly ``after``.

    An entry whose ``before`` is empty is one the apply *created* a revised
    caption for; :func:`~anime_tools.stages.replay.undo_one` deletes that file
    rather than filling it with the master's text."""
    results: list[dict] = []
    for entry in manifest:
        outcome = {"image": entry["image"], "status": "skipped"}
        results.append(outcome)
        # Content-based, not status-based: the current==after check below is the
        # real guard, and it stays valid however many times apply ran.
        if not entry.get("after"):
            outcome["status"] = f"no-edit ({entry.get('status')})"
            continue
        status = undo_one(
            resized_dir / entry["caption_path"],
            entry["after"],
            entry["before"],
            apply=apply,
            drop_variants=True,
            history_by="audit",
        )
        outcome["status"] = _REVERT_STATUS.get(status, status)
    return results
