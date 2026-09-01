"""Find images that draw one character several times but are captioned ``1girl``.

Audits the complement of the clause pipeline: every image ``caption-position``
throws away as ``single-subject`` (girls-count <= 1 and no layout tag), which is
exactly the population a mis-tagged view sheet hides in. Two boxes where the
caption claims one girl is the finding; three signals then argue about what it
means, and a verdict needs two of them before ``--apply`` will write it:
identity agreement across the crops (the only one that separates "one girl drawn
twice" from "a character the caption never counted", but silent on a headless
close-up), the whole-image ``multiple views`` head, and the people-count head
insisting on ``1girl`` while the geometry sees several bodies.

Detection forces its escalation target to ``min_instances`` (2) rather than the
caption's count, so the low-threshold retry and the body-part fallback actually
fire; gating on the caption's ``expected=1`` would stop the search at the first
box, on precisely the image we are trying to catch.

Read-only apart from :func:`apply_findings`. Full rationale in
``docs/multiview_audit.md``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from anime_tools.captions.caption_layout import (
    caption_boy_count,
    is_candidate,
)
from anime_tools.captions.clause_vocabulary import (
    ClauseVocabulary,
    default_clause_groups,
)
from anime_tools.captions.position_clauses import (
    assign_positions,
    compose_caption,
    flat_tag_set,
    ordered_indices,
    parse_caption,
)
from anime_tools.captions.taxonomy import count_of, exact_count, normalize_tag
from anime_tools.stages.instance_detection import Detection, crop_instance

from ._walk_captions import iter_captions
from .position_captions import PositionCaptionOptions, detect_subjects
from .replay import apply_one

# The audit population, named by `is_candidate`'s own reason string, so the two
# cannot drift apart.
AUDIT_SKIP_REASON = "single-subject"

MULTIPLE_VIEWS = "multiple views"
EXTRA_CHARACTER = "extra-character"
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
# Groups needed before the agreement ratio is trusted at all. One is a coin
# flip: two crops agreeing only on `brown hair` says nothing.
_MIN_COMPARABLE_GROUPS = 2
# P(multiple views) from the whole-image tagger at which it counts as a witness.
DEFAULT_MULTIVIEW_PROB = 0.5
# Probability an identity-group winner needs before it is believed. Measured
# rather than picked: a legible face scores 0.978-1.000 on hair/eye colour,
# while a headless panel's invented values land at 0.54-0.63. Anything in that
# gap is the head guessing from an empty crop.
DEFAULT_IDENTITY_CONFIDENCE = 0.9

# Which detector produced the finding.
SOURCE_DETECTION = "detection"
SOURCE_TAGGER = "tagger-only"


@dataclass
class CropIdentity:
    """What the tagger read off one crop, reduced to the identity trio + name.

    GOTCHA: ``groups`` holds only the values that cleared
    ``DEFAULT_IDENTITY_CONFIDENCE`` **on the raw probability**. The group heads
    are an argmax over a softmax, so they name a hair colour for a headless
    crop as confidently as for a portrait, and they land in ``kept`` by that
    argmax rather than by a sigmoid threshold — so filtering on ``kept``
    membership does nothing at all.
    """

    box: tuple[float, float, float, float]
    score: float
    position: str = ""
    source: str = "subject"
    # Cleared the detector's own confidence floor, rather than being recovered
    # by the retry escalation. Only a reliable box's identity gets a vote.
    reliable: bool = True
    groups: dict[str, str] = field(default_factory=dict)
    # What the ungated argmax said, and how sure it was — so a reviewer can see
    # whether a missing identity is a shy head or a genuinely faceless crop.
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
    girls-count of 0 stays in: the ``girl`` prompt finding two subjects on
    scenery or a ``1boy`` caption is just as much a caption bug.
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
) -> tuple[float | None, int]:
    """Fraction of identity groups on which every crop agrees, and how many voted.

    Two things are excluded from the vote:

    * A crop the detector was **not confident about** (``reliable=False``, i.e.
      recovered only by the retry escalation). Its mask comes out shredded, and
      mask-blanking then hands the tagger a near-white canvas it reads a hair
      colour off *at 0.99* — which the probability gate cannot catch. Detection
      score separates these; mask fill measurably does not
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
    for group in sorted(default_clause_groups().identity):
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
) -> tuple[str, float | None, int]:
    """Same character in every box, or a character the caption never counted?

    Identity first, as the only signal that distinguishes the two. When the
    crops can't supply it, the whole-image ``multiple views`` head is the
    fallback, and its absence leaves ``unsure`` rather than a guess.
    """
    agreement, comparable = identity_agreement(crops)
    names = {c.name for c in crops if c.name and c.reliable}
    # A name disagreement is decisive on its own — two *named* characters is the
    # one signal the identity trio can't fake (two girls can share brown hair).
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

    ``multiple views`` is count-agnostic, so it needs no arithmetic and is
    always safe to propose. The extra-character case is **report-only by default**: detection counts
    *subjects*, not girls (the ``girl`` prompt does not reliably exclude males),
    so an ``Ngirls`` proposal is wrong whenever the extra body is a boy. Opt in
    with ``suggest_counts``.
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

    "End of the bag", not end of the *text*: position clauses have to stay
    trailing or the grammar stops parsing them, so the splice goes through
    :func:`compose_caption` rather than a string concat.

    A minimal splice, not a re-correction — the ordering pass buckets tags
    properly. An ``Ngirls`` suggestion is the exception: it *replaces* the stale
    count in place, since two girls-counts in one bag contradict each other.

    "Already there?" is :func:`normalize_tag`, the shared key — the suggestion
    is spelled the tagger's way and the bag may be spelled the master's.
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
) -> MultiviewFinding:
    """Detect, and when more than one subject lands, ask the tagger who they are.

    An image with nothing to report comes back ``UNSURE`` with ``instances < 2``;
    the caller filters.
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

    # Unconditional, before the geometry decides anything: one extra forward,
    # and the only thing that can flag a sheet SAM merged into a single box.
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

    # A caption whose counts already cover every detected body has nothing wrong
    # with it: `1girl, 1boy` lands in this audit only because the *girls*-count
    # is one, and the `girl` prompt does not exclude males. Same range the clause
    # pipeline's count gate uses; an unknown boy count (None) is unbounded.
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
                # Confidence-gated on the raw probability — see the CropIdentity
                # docstring for why `kept` membership is not a gate here.
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
        finding.crops, finding.tagger_multiple_views, multiview_threshold
    )
    finding.verdict = verdict
    finding.identity_agreement = agreement
    finding.comparable_groups = comparable
    finding.suggested_tag = suggest_tag(
        verdict, finding.instances, girls, suggest_counts=suggest_counts
    )
    if finding.suggested_tag:
        finding.proposed = propose_caption(caption, finding.suggested_tag)

    # Independent witnesses to "one girl drawn twice". Detection alone cannot
    # tell two views from two girls, so `strong` means at least two of the three
    # agreed — the tier `--apply` writes.
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
        # Different question, different witnesses: an extra character is made
        # credible by the identity evidence splitting, not the view head firing.
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
) -> tuple[list[MultiviewFinding], MultiviewAuditStats]:
    """Walk the resized tree and report every under-counted caption.

    ``caption_path`` is reported relative so the caller can decide which tree to
    edit. Nothing here writes.
    """
    options = options or PositionCaptionOptions()
    stats = MultiviewAuditStats()
    rows: list[MultiviewFinding] = []

    for image_path, rel, _dst_caption, caption in iter_captions(
        resized_dir, source_dir, path_pattern, stats, progress
    ):
        ok, reason = is_audit_target(caption)
        if not ok:
            stats.skip(reason)
            continue
        stats.audited += 1

        # The sheet needs the exact crops the tagger read, so they are teed into
        # memory as produced (one image's worth) rather than re-cropped from the
        # boxes: a plain bbox re-crop shows different pixels than were scored.
        held: list[Image.Image] = []
        save_crop = None
        if crops_dir is not None:
            from .position_captions import _crop_sink

            save_crop = _crop_sink(crops_dir, rel)

        # ``held``/``save_crop`` are rebuilt per image and the sink is consumed
        # inside this iteration; bound as defaults so that holds by
        # construction rather than by reading the call order.
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
        )
        # Only the tagger can raise an image the geometry had nothing to say
        # about: too few boxes, or a caption that already counts them all.
        if finding.source != SOURCE_TAGGER:
            if finding.verdict == COUNT_EXPLAINED:
                stats.skip(COUNT_EXPLAINED)
                continue
            if finding.instances < max(2, options.min_instances):
                stats.skip("single-instance")
                continue
        finding.image = str(image_path.relative_to(resized_dir))
        finding.caption_path = str(rel)
        stats.findings += 1
        stats.verdicts[finding.verdict] += 1
        rows.append(finding)

        if sheets_dir is not None:
            from .multiview_sheet import render_contact_sheet, sheet_path

            target = sheet_path(sheets_dir, finding)
            target.parent.mkdir(parents=True, exist_ok=True)
            render_contact_sheet(image, finding, held).save(target)
            finding.sheet = str(target.relative_to(sheets_dir))

    return rows, stats


def apply_findings(
    findings: Iterable[MultiviewFinding],
    *,
    source_dir: Path,
    verdicts: Sequence[str] = (MULTIPLE_VIEWS,),
    confidences: Sequence[str] = ("strong",),
) -> tuple[list[tuple[str, str, str]], Counter]:
    """Write proposed captions into the **master** tree.

    Returns ``(written, skipped)``: the ``(rel, before, after)`` triples written
    and a count per :func:`~anime_tools.stages.replay.apply_one` status for the
    gated rows that were not, so a master edited since the audit is ``drifted``
    and left alone.

    The master is the right target here, unlike the clause rewrite: a missing
    ``multiple views`` is a fact about the picture, not a derived re-phrasing,
    and every later stage reads down from it.

    GOTCHA: ``image_dataset/`` is gitignored, so this is not git-recoverable —
    the report's ``caption`` field is the only verbatim before-text.
    """
    written: list[tuple[str, str, str]] = []
    skipped: Counter = Counter()
    for finding in findings:
        if finding.verdict not in verdicts or finding.confidence not in confidences:
            continue
        status = apply_one(
            source_dir / finding.caption_path,
            finding.caption,
            finding.proposed,
            apply=True,
            newline=True,
        )
        if status == "written":
            written.append(
                (finding.caption_path, finding.caption.strip(), finding.proposed)
            )
        else:
            skipped[status] += 1
    return written, skipped


def curated_proposal(row: dict) -> tuple[str, str] | None:
    """(tag, proposed caption) for a hand-accepted report row.

    An accepted ``unsure`` means the reviewer judged the sheet a view layout; an
    accepted ``extra-character`` means the body count is real
    (``{instances}girls`` replacing the stale count). ``None`` when no edit
    applies.
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


# :func:`apply_one`'s vocabulary read from the revert direction.
_REVERT_STATUS = {
    "written": "reverted",
    "would-write": "would-revert",
    "already-applied": "already-reverted",
}


def apply_curated(
    rows: Iterable[dict],
    accepted: set[str],
    *,
    source_dir: Path,
    apply: bool,
) -> tuple[list[dict], list[str]]:
    """Apply a reviewer-curated accept list of report rows to the caption master.

    Returns ``(manifest, unmatched)``: one entry per accepted row with the
    verbatim before/after (the only undo — ``image_dataset/`` is gitignored) and
    any accepted image with no finding. Same drift guard as
    :func:`apply_findings`.
    """
    by_image = {r["image"]: r for r in rows if r.get("verdict")}
    unmatched = sorted(accepted - set(by_image))
    manifest: list[dict] = []
    for image in sorted(accepted & set(by_image)):
        row = by_image[image]
        entry = {
            "image": image,
            "caption_path": row["caption_path"],
            "verdict": row["verdict"],
            "confidence": row["confidence"],
            "tag": None,
            "before": row["caption"],
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
            source_dir / row["caption_path"],
            row["caption"],
            entry["after"],
            apply=apply,
            newline=True,
        )
    return manifest, unmatched


def revert_curated(
    manifest: Iterable[dict],
    *,
    source_dir: Path,
    apply: bool,
) -> list[dict]:
    """Undo :func:`apply_curated` from its manifest, restoring ``before`` only
    where the master still holds exactly ``after``."""
    results: list[dict] = []
    for entry in manifest:
        outcome = {"image": entry["image"], "status": "skipped"}
        results.append(outcome)
        # Content-based, not status-based: the current==after check below is
        # the real guard, and it stays valid however many times apply ran.
        if not entry.get("after"):
            outcome["status"] = f"no-edit ({entry.get('status')})"
            continue
        # A revert is an apply with the two texts swapped; only the names of the
        # success-adjacent outcomes differ.
        status = apply_one(
            source_dir / entry["caption_path"],
            entry["after"],
            entry["before"],
            apply=apply,
            newline=True,
        )
        outcome["status"] = _REVERT_STATUS.get(status, status)
    return results
