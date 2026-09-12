"""The caption stages as request objects — the surface the CLIs, the GUI and the
trainer share.

Torch-free: run one through its stage's runner (``run_autotag(req)`` and
friends, re-exported from :mod:`anime_tools.stages` and named by
``registry.py``, which import the models), or hand ``to_argv()`` to a subprocess.
Every field is a flag of the matching ``stages/cli`` parser — the parser is
generated from these classes (:meth:`Request.parser`), so the help, the
default and the argument group are written here, once. Flags are spelled with
underscores (``--path_pattern``), the caption stages' canonical form, and take
the hyphenated spelling as an alias.

The SAM3 detection flags are one nested :class:`DetectionRequest`, and the
audit's verdict gate one nested :class:`MultiviewRequest`, so the position stage
and the multiview audit cannot declare either twice.

Every default and help string comes from a leaf — :mod:`anime_tools.stages._options`
and :mod:`anime_tools.masking._prompts` — never from the stage module that uses it:
``registry.py`` is import-light on purpose, and the GUI resolves each request class
to build its form, so reaching into a stage for one number would cost that build
numpy, PIL and yaml. ``tests/test_registry_requests.py`` measures it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import ClassVar

from anime_tools import workspace as WS
from anime_tools._device import DEVICE_HELP
from anime_tools._request import READ, WRITE, Request, arg
from anime_tools.buckets import ALLOWED_TARGET_RES
from anime_tools.captions.tag_drop_groups import drop_group_names
from anime_tools.contract import AUTOTAG_MODES
from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT, DEFAULT_SUBJECT_PROMPT_EMBED
from anime_tools.masking._prompts import (
    CHECKPOINT_HELP,
    PROMPT_EMBED_HELP,
    SUBJECT_PROMPT,
    prompt_list,
)
from anime_tools.stages._options import (
    CROP_ANCHORS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CROP_ANCHOR,
    DEFAULT_IDENTITY_CONFIDENCE,
    DEFAULT_MIN_PIXELS,
    DEFAULT_MULTIVIEW_PROB,
    EXTRA_CHARACTER,
    MULTIPLE_VIEWS,
    PositionCaptionOptions,
)
from anime_tools.tagger.dbv4_meta import DEFAULT_TAGGER_DIR

__all__ = [
    "MULTIVIEW_MODES",
    "POSITION_ONLY_FLAGS",
    "AuditRequest",
    "AutotagRequest",
    "CorrectRequest",
    "DetectionRequest",
    "ExportRequest",
    "MultiviewRequest",
    "OcrRequest",
    "PositionRequest",
    "ResizeRequest",
]

DEFAULT_MAX_TOKENS = 512
"""Both tokenizers pad to this; a caption past it truncates silently."""

DEFAULT_EXPORT_INDEX = f"{WS.REPORTS}/caption_index.json"

PATTERN_HELP = "fnmatch glob (| to OR-combine) on the path relative to {root}"

POSITION_ONLY_FLAGS = ("blank_crops", "min_instances", "strict_count")
"""Detection-group knobs only the position stage declares: the audit pins
``min_instances`` and takes the ``PositionCaptionOptions`` default for the other
two."""

DETECTION = "detection"
"""The argument group both SAM3 stages run their detector under."""

MULTIVIEW = "multiview audit"
"""The argument group the audit's verdict gate sits in — declared once in
:class:`MultiviewRequest` and embedded by both SAM3 stages."""

MULTIVIEW_MODES = ("off", "report", "apply")
"""``PositionRequest.multiview_audit``: what the audit phase is allowed to do."""

OCR_DRAWER = "Combine OCR"
"""Export's drawer: the text-clause combine and the tree it reads."""

DETECTOR = "Detector"
"""The OCR stage's group for the text-block detector's knobs."""

VL_READER = "VL reader"
"""The OCR stage's group for the manga VL reader and the optional text-mask
components it reads."""


def _csv(default: tuple[str, ...], **meta) -> tuple[str, ...]:
    """A comma-separated flag held as a tuple."""
    return field(default=default, metadata={READ: prompt_list, WRITE: ",".join, **meta})


def _off(default: bool, flag: str, *, help: str = "", group: str = "") -> bool:
    """A ``store_false`` switch: ``flag`` is the one spelling that flips it."""
    return arg(default, off=flag, help=help, group=group)


def _report_dir(default: str) -> str:
    return arg(default, help=f"Where report.json lands (default: {default})")


@dataclass(frozen=True, kw_only=True)
class StageRequest(Request):
    FLAG_SEP: ClassVar[str] = "_"


@dataclass(frozen=True, kw_only=True)
class DatasetRequest(StageRequest):
    """The three dataset roots every caption stage walks."""

    src: str = arg(
        WS.SOURCE_ROOT,
        help="Caption master dir — hand-written, and the read-only fallback for "
        "every stage. Only Export writes it, publishing a revised master back",
    )
    dst: str = arg(
        WS.RESIZED,
        help="Resized images — what every stage opens, and where the revised caption "
        "is written",
    )
    path_pattern: str = arg("*", help=PATTERN_HELP.format(root="--dst"))


@dataclass(frozen=True, kw_only=True)
class ApplyRequest(DatasetRequest):
    """Dry-run by default; ``apply`` writes.

    The report every run leaves is what the GUI's Undo reads back
    (``anime_tools.contract.REPLAY_SHAPES``), whether or not the stage can
    replay one.
    """

    apply: bool = arg(
        False, help="Write what the run proposes (default: dry run, report only)"
    )
    report_dir: str
    """Where ``report.json`` lands; distinct per stage so one stage cannot read
    another's report."""


@dataclass(frozen=True, kw_only=True)
class ReplayRequest(ApplyRequest):
    """Adds ``from_report``: apply a dry run's proposals without a model.

    Only for a stage whose model pass is worth not repeating — position clauses
    (SAM3 detect + crop + tag) and the multiview audit (one pass re-applied at
    several verdict/confidence tiers). Autotag is one forward pass per image and
    deliberately has no replay: re-run it.
    """

    from_report: str | None = arg(
        None,
        help="Replay a previous dry run's report.json instead of re-running the "
        "model: writes exactly what it proposed and loads no model. Skips any "
        "caption that changed since. Emits apply_report.json beside the report it "
        "reads, never over it",
    )


@dataclass(frozen=True, kw_only=True)
class TaggerRequest(StageRequest):
    """The Anima Tagger flags."""

    tagger_dir: str | None = arg(
        None, help=f"Anima Tagger checkpoint dir (default: {DEFAULT_TAGGER_DIR})"
    )
    device: str | None = arg(None, help=DEVICE_HELP)
    """``None`` resolves at model-load time, never earlier, so the torch-free
    replay path never pays for the probe."""


@dataclass(frozen=True, kw_only=True)
class DetectionRequest(StageRequest):
    """The SAM3 detector both SAM3 stages run: subject prompt, thresholds, the
    body-part fallback, and the dedupe geometry.

    Everything here but ``checkpoint`` and ``prompt_embed`` is a field of
    :class:`PositionCaptionOptions`, which is how :meth:`PositionRequest.options`
    builds the detection half.
    """

    GROUP: ClassVar[str] = DETECTION

    prompt: str = arg(SUBJECT_PROMPT, help="SAM3 text prompt for a subject")
    prompt_embed: str = arg(DEFAULT_SUBJECT_PROMPT_EMBED, help=PROMPT_EMBED_HELP)
    """The learned soft prompt standing in for ``prompt``; ``none`` for text."""
    checkpoint: str = arg(DEFAULT_SAM3_CHECKPOINT, help=CHECKPOINT_HELP)
    score_threshold: float = arg(
        0.5,
        help="Subject confidence floor. Raising it trades recall for fewer "
        "proposals; the audit is precision-sensitive since every hit is read by hand",
    )
    retry_score_threshold: float = arg(
        0.35,
        help="Retry threshold when detection undershoots the expected count. This "
        "is SAM3's own confidence floor, not a post-filter — see build_detect_fn",
    )
    part_prompts: tuple[str, ...] = _csv(
        (),
        help="Comma-separated body-part prompts, tried only when the subject prompt "
        "undershoots — recovers headless close-up panels (a hip / backside crop next "
        "to one full body) that 'girl' cannot see at any threshold. Off by default; "
        'try "buttocks,hips,thighs"',
    )
    part_score_threshold: float = arg(
        0.5,
        help="Confidence floor for a body-part box (kept separate from the subject "
        "threshold — part prompts are the looser concept)",
    )
    part_containment_threshold: float = arg(
        0.7,
        help="Drop a part box this nested inside an already-kept box. Unlike "
        "--containment_threshold this is safe to leave on: a part inside a subject "
        "is that subject's own body, never a second subject",
    )
    iou_threshold: float = arg(
        0.65, help="NMS IoU above which two boxes are one subject"
    )
    containment_threshold: float = arg(
        1.01,
        help="Suppress a box this nested inside a kept one (intersection over the "
        "smaller box). Off by default (>1.0 disables): a real second subject is as "
        "nested as a group box — enabling it cost 32 real subjects to save 12 group "
        "boxes",
    )
    mask_containment_threshold: float = arg(
        0.8,
        help="Suppress a detection whose MASK is this nested inside a kept one. On "
        "by default, unlike its box counterpart: a second girl in front of the first "
        "nests identically by box but her mask is disjoint. >1.0 disables (the "
        "pre-2026-08-19 behaviour)",
    )
    dedupe_fill_ratio: float = arg(
        2.0,
        help="Mask-quality tie-break inside an NMS-matched pair; 0 = off (score-only "
        "survivor). See docs/multiview_audit.md.",
    )
    min_area_frac: float = arg(
        0.005,
        help="Drop detections smaller than this fraction of the image — an inset (a "
        "character on a phone screen) is not a bindable subject",
    )
    pad: float = arg(0.06, help="bbox padding fraction")
    row_tol: float = arg(
        0.25,
        help="Minimum fractional overlap (of the narrower box extent) for two "
        "subjects to share a row — and a column, on magazine layouts where a "
        "full-height subject bridges a stack of panels",
    )
    max_instances: int = arg(8, help="Most subjects one image may bind")

    @property
    def floor(self) -> float:
        """The lowest confidence any pass asks for — the processor's own floor."""
        return min(
            self.score_threshold, self.retry_score_threshold, self.part_score_threshold
        )


VERDICT = "verdict"


@dataclass(frozen=True, kw_only=True)
class MultiviewRequest(StageRequest):
    """The multiview audit's verdict gate: what counts as a witness, what the
    review surface is, and which findings a write is allowed to touch.

    Nested rather than declared per stage, because the audit runs from two
    places — its own stage, and the position stage's ``--multiview_audit``
    phase — and a drifting threshold between the two would make the same
    corpus give two answers.
    """

    GROUP: ClassVar[str] = MULTIVIEW

    MIN_INSTANCES: ClassVar[int] = 2
    """Pinned rather than exposed: two subjects is what the audit is for."""

    multiview_threshold: float = arg(
        DEFAULT_MULTIVIEW_PROB,
        help="Whole-image P(multiple views) at which the tagger counts as a witness "
        "— and, on its own, raises an image detection saw as one box",
        group=VERDICT,
    )
    identity_confidence: float = arg(
        DEFAULT_IDENTITY_CONFIDENCE,
        help="Probability an identity-group winner needs before the verdict "
        "believes it. The group heads are softmax argmaxes, so they name a hair "
        "colour for a headless crop too — lowering this lets those back in",
        group=VERDICT,
    )
    suggest_counts: bool = arg(
        False,
        help=f"Also propose an 'Ngirls' fix for a '{EXTRA_CHARACTER}' verdict. Off "
        "because the 'girl' prompt does not exclude males — check the people-count "
        "head in the report before trusting any of these",
        group=VERDICT,
    )
    apply_verdicts: tuple[str, ...] = _csv(
        (MULTIPLE_VIEWS,),
        help=f"Comma-separated verdicts a write may touch ('{MULTIPLE_VIEWS}', "
        f"'{EXTRA_CHARACTER}')",
        group=VERDICT,
    )
    apply_confidence: tuple[str, ...] = _csv(
        ("strong",),
        help="Comma-separated confidence tiers a write may touch (strong, weak). A "
        "weak finding has only the geometry behind it — review the crops first",
        group=VERDICT,
    )
    sheets: bool = _off(
        True,
        "--no_sheets",
        help="Skip the per-finding contact sheets. They are the review surface — "
        "boxed original + the crops the tagger saw + the proposed edit, one PNG per "
        "finding under <report_dir>/sheets/, named verdict-first",
        group=MULTIVIEW,
    )


def _options(*sources) -> PositionCaptionOptions:
    """A :class:`PositionCaptionOptions` whose every field is read off the first
    of ``sources`` that has it. A field none has is a missing flag."""
    kw = {}
    for f in fields(PositionCaptionOptions):
        for source in sources:
            if hasattr(source, f.name):
                kw[f.name] = getattr(source, f.name)
                break
        else:
            raise AttributeError(f"no request field for option {f.name!r}")
    return PositionCaptionOptions(**kw)


def audit_options(
    detection: DetectionRequest, name_confidence: float
) -> PositionCaptionOptions:
    """The options the multiview audit phase detects under, wherever it runs from.

    The caller's detector verbatim, ``min_instances`` pinned to
    :data:`MultiviewRequest.MIN_INSTANCES`, every other knob at its default.
    Two subjects rather than the caption's count is deliberate: passing
    ``expected=1`` would satisfy the target on the first box and suppress both
    the low-threshold retry and the body-part fallback, on the exact population
    the audit exists to search (``docs/multiview_audit.md``).
    """
    return _options(
        detection,
        _Pinned(
            min_instances=MultiviewRequest.MIN_INSTANCES,
            name_confidence=name_confidence,
        ),
        PositionCaptionOptions(),
    )


@dataclass(frozen=True, kw_only=True)
class AutotagRequest(TaggerRequest, ApplyRequest):
    """Auto-tag the dataset with the Anima Tagger and write the revised caption.

    Walks the resized tree and proposes a caption per image, in one of three
    ``--mode``s: ``missing`` (images no caption speaks for, the default), ``merge``
    (append novel tags, keeping clauses) or ``overwrite`` (replace the caption
    outright). The caption it reads is the revised one, falling back to the master;
    the caption it writes is always the revised one, and what that replaced is kept
    as a ``{stem}.history.txt`` version.

    Dry-run by default; ``--apply`` writes. The TE caches go stale but still look
    current, so follow a real apply with ``make preprocess-te``.
    """

    mode: str = arg(
        "missing",
        choices=AUTOTAG_MODES,
        help="missing: only images no caption speaks for (default). merge: append "
        "novel tags to every caption, keeping its position clauses. overwrite: "
        "replace the caption outright (the replaced text is kept as a history "
        "version)",
    )
    min_confidence: float = arg(
        0.0,
        help="Extra probability floor on top of the tagger's per-tag F1 thresholds "
        "(0-1). 0 leaves its own decisions untouched",
    )
    batch_size: int = arg(DEFAULT_BATCH_SIZE, help="Images per tagger forward pass")
    report_dir: str = _report_dir(f"{WS.REPORTS}/autotag")

    def __post_init__(self) -> None:
        if self.mode not in AUTOTAG_MODES:
            raise ValueError(f"--mode must be one of {list(AUTOTAG_MODES)}")


CLAUSES = "clause composition"


@dataclass(frozen=True, kw_only=True)
class PositionRequest(TaggerRequest, ReplayRequest):
    """Rewrite multi-subject captions into position clauses (SAM3 + Anima Tagger).

    Detect -> order -> crop+blank -> tag -> compose over the resized dataset, moving
    each attributable tag out of the flat bag into its clause (``--flatten`` is the
    inverse pass). Clauses land on the revised caption under ``--dst``, never on the
    caption master. See ``docs/position_captions.md``.

    Dry-run by default; ``--apply`` writes and drops stale ``.variants.txt``
    sidecars, so follow it with a TE re-encode.
    """

    report_dir: str = _report_dir(f"{WS.REPORTS}/position")
    crops: bool = arg(
        False, help="Also export the mask-blanked crops next to the report (review aid)"
    )
    flatten: bool = arg(
        False,
        help="Inverse pass: merge every caption's clauses back into its flat bag and "
        "drop the clauses. Text-only (no SAM3, no tagger) — this is how an --apply "
        "run is backed out, and how the clause-free control corpus for a training "
        "A/B is built. Flattens hand-written clauses too.",
    )
    detection: DetectionRequest = field(default_factory=DetectionRequest)
    multiview: MultiviewRequest = field(default_factory=MultiviewRequest)
    multiview_audit: str = arg(
        "off",
        choices=MULTIVIEW_MODES,
        help="Run the multiview audit as this stage's FIRST phase, over the "
        "captions this one skips as 'single-subject'. off: never (default). "
        "report: audit, write findings + contact sheets, tag nothing — this "
        "stage then sweeps only its own population. apply: also promote every "
        "finding the --apply_verdicts/--apply_confidence gate admits, so a "
        "newly-tagged sheet falls into THIS run's sweep. The order is the whole "
        "point: 'multiple views' is what moves an image from is_candidate's "
        "'single-subject' rejection to 'multiple-views', and what arms the "
        "view-invariant gate — audit after the sweep and the tag arrives too "
        "late to do either. See docs/multiview_audit.md.",
        group=MULTIVIEW,
    )
    blank_crops: bool = _off(
        True,
        "--no_blank_crops",
        help="Skip mask-blanking (probe B: this is what caused the hair-color misses)",
        group=DETECTION,
    )
    min_instances: int = arg(
        2, help="Fewest subjects an image needs before it gets clauses", group=DETECTION
    )
    strict_count: bool = _off(
        True,
        "--no_strict_count",
        help="Propose clauses even when detection disagrees with the girls-count",
        group=DETECTION,
    )
    max_clause_tags: int = arg(8, help="Most tags one clause may carry", group=CLAUSES)
    max_novel_tags: int = arg(
        1,
        help="How many tags a clause may introduce that the caption never contained. "
        "The rest of the clause is filled from the flat bag first. Only a bag tag can "
        "MOVE — a novel one is a pure addition the curated caption never made. 0 = "
        "never invent, --max_clause_tags = the old bag-blind behaviour (46% novel; on "
        "ama_mitsuki, 1 vs 8 cut novel tags 515 to 115 and the caption 40% shorter, "
        "with the moved set unchanged)",
        group=CLAUSES,
    )
    name_confidence: float = arg(
        0.5,
        help="Confidence floor for putting a character name in a clause",
        group=CLAUSES,
    )
    allow_unlisted_names: bool = arg(
        False,
        help="Allow a clause name the flat caption never mentions (off: probe B "
        "scored names 4/7, so an unlisted one is most likely a crop artifact)",
        group=CLAUSES,
    )
    discriminative_only: bool = _off(
        True,
        "--keep_shared_tags",
        help="Keep tags every crop agrees on in every clause. Off by default: on a "
        "multiple-views sheet all views share the character, hair and eyes, so "
        "repeating them binds nothing and crowds out the outfit that differs (they "
        "stay in the flat bag either way — v1 never removes anything).",
        group=CLAUSES,
    )
    bag_gated_identity: bool = _off(
        True,
        "--ungated_identity",
        help="Let a clause carry a hair/eye color the flat caption never listed. "
        "Gated by default: the caption is the curated ground truth, the crop tagger "
        "guesses one for every crop including headless ones, and discriminative-only "
        "then promotes the guess precisely because it disagrees — 520 of 1600 "
        "identity clause tags in the first full-corpus dry run contradicted the "
        "caption",
        group=CLAUSES,
    )
    multi_view_gate: bool = _off(
        True,
        "--bind_view_traits",
        help="On a repeated-subject layout (`multiple views`, comic panels), let a "
        "clause carry the character's name and traits (hair, eyes, body, anatomy). "
        "Gated by default: every view or panel is the SAME girl, so those belong to "
        "her, not to a view — 45% of the multiple-views clause tags in the first "
        "full-corpus dry run were view-invariant, and the ones that survived "
        "shared-tag suppression did so precisely because a crop disagreed",
        group=CLAUSES,
    )
    bind_view_anatomy: bool = _off(
        True,
        "--gate_view_anatomy",
        help="On a repeated-subject layout, keep anatomy (`ass`, `thighs`, "
        "`body_parts`) out of every clause — the pre-2026-08-19 behaviour, when "
        "`body_parts` sat in the view-invariant set. Bound by default: unlike hair "
        "color, what anatomy is *visible* is a fact about the panel, so on a sheet of "
        "one girl from the front and the same girl from behind it is the tag that "
        "separates them",
        group=CLAUSES,
    )
    bind_framing: bool = _off(
        True,
        "--no_framing",
        help="Keep `framing` out of every clause (the pre-2026-08-19 behaviour). On "
        "by default: it is the only group that says a view is a headless close-up "
        "rather than a whole figure, which on a `multiple views` sheet of one full "
        "body plus a hip/backside panel is the single thing that tells the clauses "
        "apart. Off restores the A side for an A/B.",
        group=CLAUSES,
    )
    rewrite: bool = _off(
        True,
        "--no_rewrite",
        help="Additive v1: append the clauses but leave the flat bag untouched, so "
        "every bound attribute is asserted twice. Default is the v2 rewrite, which "
        "moves an attributable tag out of the bag into its clause. Kept for the "
        "training A/B arm",
        group=CLAUSES,
    )
    bag_relax: float = arg(
        0.35,
        help="Multiplier on the tagger's per-tag keep threshold for tags the flat "
        "bag already contains (they can only MOVE into a clause, never be invented, "
        "so the curated caption corroborates them — the crop only attributes). 1.0 "
        "= off, the pre-2026-08-19 behaviour. Applied to every crop before the "
        "attributable/shared census, so a rival crop's borderline score also blocks "
        "a move the strict kept sets would have granted. Motivating case: 5828184's "
        "`black panties` scored 0.498 against a 0.800 threshold on the lying crop "
        "and stayed unbound; the 0.35 default is what recovers pose tags off "
        "mask-blanked crops",
        group=CLAUSES,
    )
    bag_word_relax: float = arg(
        0.85,
        help="Extra threshold multiplier per word beyond the first, compounding "
        "with --bag_relax (`black panties` is more specific than `panties`, so a "
        "sub-threshold hit on it is less likely noise). 1.0 = off",
        group=CLAUSES,
    )
    bag_relax_min_score: float = arg(
        0.3,
        help="Absolute score floor under the bag relaxation: a relaxed admission "
        "still needs at least this raw probability, however low bag_relax × "
        "bag_word_relax drags the per-tag threshold. Blocks near-noise fires "
        "(measured: `white gloves` bound to a crop with no hands in frame at a ~0.16 "
        "relaxed floor) while keeping the genuine recoveries (`black panties` at "
        "0.498). Only the relax path is floored. 0.0 = off, the pre-floor behaviour",
        group=CLAUSES,
    )
    attribution_margin: float = arg(
        0.25,
        help="How far the winning crop's probability must clear every other crop's, "
        "RELATIVE to its own (1 - rival/winner), before a tag may LEAVE the flat bag "
        "(the clause carries it either way). Applies on top of the hard rule that no "
        "other crop kept the tag; 0.0 trusts the tagger's per-tag thresholds alone. "
        "Guards the one thing v2 can get wrong that v1 cannot: removing an attribute "
        "the other subjects also have",
        group=CLAUSES,
    )
    qwen3: str | None = arg(
        None,
        help="Qwen3 tokenizer directory — enables the token-budget column in the report",
        group=CLAUSES,
    )
    max_tokens: int = arg(
        DEFAULT_MAX_TOKENS,
        help="Token budget the report measures against",
        group=CLAUSES,
    )

    def __post_init__(self) -> None:
        if self.flatten and self.from_report:
            raise ValueError(
                "--flatten and --from_report are mutually exclusive: the flatten "
                "pass is already text-only, so there is no model pass to skip."
            )
        if self.multiview_audit not in MULTIVIEW_MODES:
            raise ValueError(
                f"--multiview_audit must be one of {list(MULTIVIEW_MODES)}"
            )
        # Both text-only passes load no model, and the audit phase is a SAM3 +
        # tagger sweep — silently skipping it would be the worse failure.
        if self.audits and (self.flatten or self.from_report):
            raise ValueError(
                "--multiview_audit needs the model pass: --flatten and "
                "--from_report are text-only. Run the audit in its own stage "
                "(`python -m anime_tools.stages.cli.audit_multiview`) instead."
            )

    @property
    def audits(self) -> bool:
        """Whether the audit phase runs at all."""
        return self.multiview_audit != "off"

    @property
    def promotes(self) -> bool:
        """Whether a finding the gate admits is fed to the clause sweep."""
        return self.multiview_audit == "apply"

    def options(self) -> PositionCaptionOptions:
        """The options one pass runs under: the detection block plus the
        clause-composition knobs."""
        return _options(self, self.detection)

    def audit_options(self) -> PositionCaptionOptions:
        """The options the audit PHASE detects under — this stage's detector with
        ``min_instances`` pinned, exactly what the audit stage would use."""
        return audit_options(self.detection, self.name_confidence)


@dataclass(frozen=True, kw_only=True)
class AuditRequest(TaggerRequest, ReplayRequest):
    """Audit `1girl` captions for images that are really several views of one girl.

    Sweeps the images the position stage skips as ``single-subject`` and reports
    every one where the ``girl`` prompt finds two or more subjects. See
    ``docs/multiview_audit.md``. The position stage runs this as its own first
    phase under ``--multiview_audit``, which is the usual way in.

    Dry-run by default; ``--apply`` writes the missing tag into the revised
    caption under ``--dst``, never the master, so follow it with a TE re-encode.
    """

    report_dir: str = _report_dir(f"{WS.REPORTS}/multiview_audit")
    crops: bool = arg(
        False, help="Export the per-instance crops next to the report (review aid)"
    )
    detection: DetectionRequest = field(default_factory=DetectionRequest)
    multiview: MultiviewRequest = field(default_factory=MultiviewRequest)
    name_confidence: float = arg(
        0.5,
        help="Confidence floor for naming the character in a finding",
        group=DETECTION,
    )

    def options(self) -> PositionCaptionOptions:
        """The position stage's detector verbatim, with ``min_instances`` pinned
        and every other non-detection knob at its default."""
        return audit_options(self.detection, self.name_confidence)


@dataclass(frozen=True)
class _Pinned:
    min_instances: int
    name_confidence: float


@dataclass(frozen=True, kw_only=True)
class CorrectRequest(ApplyRequest):
    """Write corrected captions next to resized preprocessing images.

    Corrects the revised caption in place, reading the master only for an image
    that has no revised caption yet, with optional variant sidecars. Dry-run by
    default like its siblings, and the ``report.json`` every run leaves is what
    the GUI's Undo replays backwards (``contract.REPLAY_SHAPES["correct"]``).
    """

    report_dir: str = _report_dir(f"{WS.REPORTS}/correct")
    tag_csv: str | None = arg(
        None, help="danbooru_tags_classified.csv path (default: models/ lookup)"
    )
    recursive: bool = arg(False, help="Walk subfolders")
    caption_insert_no_artist: bool = arg(
        False, help="Insert @no-artist at the artist slot when no artist marker exists"
    )
    caption_trigger_word: str = arg(
        "", help="Trigger tag to move into the caption order"
    )
    caption_trigger_at_front: bool = arg(
        False,
        help="Place caption_trigger_word at the very front instead of artist slot",
    )
    caption_drop_groups: str = arg(
        "",
        help="Comma-separated tag groups to strip from every mirrored caption (the "
        "master is never edited). Slugs: "
        + ", ".join(drop_group_names())
        + "; anything else is a literal taxonomy-path prefix from "
        "danbooru_tags_classified.csv (e.g. '효과/연출 > 조명').",
    )
    no_correct: bool = arg(
        False,
        help="Skip bucket-reordering — mirror the raw source caption verbatim as v0 "
        "(the variant-only path: shuffle sidecars without reordering).",
    )
    caption_shuffle_variants: int = arg(
        0,
        help="Number of caption variants to materialize as {stem}.variants.txt "
        "sidecars (0 = none). v0 is the corrected caption; v1..v{N-1} are "
        "smart-shuffled. The TE step encodes these verbatim.",
    )
    caption_tag_dropout_rate: float = arg(
        0.0,
        help="Per-tag dropout probability for v1..v{N-1} (ignored without variants).",
    )
    caption_tag_randomize_rate: float = arg(
        0.0,
        help="Per-tag identity-randomize probability — emits an r-family alongside "
        "the v-family. Needs --qwen3 + --t5_tokenizer_path to build the dual-single "
        "erasure pool. Ignored without >=2 variants.",
    )
    qwen3: str | None = arg(
        None,
        help="Qwen3 tokenizer directory (tokenizer-only load; required for randomize).",
    )
    t5_tokenizer_path: str | None = arg(
        None, help="T5 tokenizer directory (spiece.model + tokenizer.json)."
    )

    @property
    def randomizes(self) -> bool:
        """Whether the identity-randomized r-family is built, which needs both
        tokenizers."""
        return (
            self.caption_tag_randomize_rate > 0.0 and self.caption_shuffle_variants >= 2
        )

    def __post_init__(self) -> None:
        if self.randomizes and not (self.qwen3 and self.t5_tokenizer_path):
            raise ValueError(
                "--caption_tag_randomize_rate > 0 requires --qwen3 and "
                "--t5_tokenizer_path (tokenizer directories; `make` resolves them)."
            )


@dataclass(frozen=True, kw_only=True)
class OcrRequest(StageRequest):
    """Read the text in each image and record what it says.

    Walks the resized tree, boxes every text block with the AnimeText detector
    (``anime_tools.ocr.animetext``, YOLO12 on torch; 54 MB fetched on first
    use, GPL-3.0 weights — never bundled), reads every box with the manga VL reader
    (``anime_tools.ocr.sfx``, PaddleOCR-VL-1.6 fine-tuned on hand-lettered SFX; torch,
    ~2.8 GB fetched on first use) and writes ``{stem}.ocr.txt`` into the OCR tree,
    mirroring the resized layout. No caption is read or written, and no TE
    re-encode is needed afterwards.

    One detector for balloon lines *and* the SFX drawn onto the artwork; a box's
    read is the line, so a read the decode guard rejects drops its box. With
    ``--mask_dir`` the text mask's components no box covers are read too, as lines
    of their own (``det`` ``0.000``). Every line carries the detector's box
    confidence (``det``) and the reader's mean token confidence (``score``).

    Dry-run by default: a dry run emits ``report.json`` carrying every line it would
    have written, and ``--apply`` writes the sidecars and nothing else.
    """

    dst: str = arg(WS.RESIZED, help="Resized images")
    ocr_dir: str = arg(
        WS.OCR,
        help=f"Where the {{stem}}.ocr.txt sidecars land, mirroring --dst "
        f"(default: {WS.OCR})",
    )
    path_pattern: str = arg("*", help=PATTERN_HELP.format(root="--dst"))
    min_chars: int = arg(
        2,
        help="Drop a read shorter than this many non-space characters — a lone "
        "glyph is a misread screentone far more often than it is a word, while "
        "two is already an SFX (ドン, ゴゴ)",
    )
    skip_en: bool = _off(
        True,
        "--keep_en",
        help="Keep ASCII-only lines. Dropped by default: on a scanned comic they are "
        "the page number, the URL and the romaji sfx, never the dialogue",
    )
    strip_symbols: bool = _off(
        True,
        "--keep_symbols",
        help="Keep pictographs (♡ ★ ♪, emoji — Unicode's other-symbol class) in "
        "the text. Dropped by default: a heart is decoration on a line, not "
        "something it says; punctuation (… ! ? ~ ー) always stays",
    )
    det_conf: float = arg(
        0.25,
        help="Keep a detected box scored at least this (0-1). The model card's F1 "
        "threshold is 0.426; 0.25 boxes ~15%% more, nearly all real text on a "
        "manga page",
        group=DETECTOR,
    )
    min_det: float = arg(
        0.6,
        help="Write a line only if the detector scored its box at least this "
        "(0-1). --det_conf decides which boxes are read at all; a mask component "
        "has no box score and is not held to this",
        group=DETECTOR,
    )
    min_box_px: int = arg(
        12,
        help="Ignore a detected box whose longest side is under this many pixels — "
        "screentone and hatching, not text",
        group=DETECTOR,
    )
    max_boxes: int = arg(
        64,
        help="Read at most this many boxes per image, largest first, so one "
        "misread texture cannot cost a thousand crops",
        group=DETECTOR,
    )
    vl_batch_size: int = arg(16, help="Crops per VL forward pass", group=VL_READER)
    min_score: float = arg(
        0.6,
        help="Write a line only if the reader's mean token confidence in it is at "
        "least this (0-1); a read with no confidence is not held to this",
        group=VL_READER,
    )
    mask_dir: str | None = arg(
        None,
        help="Optional: the text-mask tree (`make mask`'s {stem}_mask.png, "
        f"mirroring --dst; usually {WS.MASKS}). Its text components no box covers "
        "are read too and become lines of their own with score 0.000 — lettering "
        "the detector missed",
        group=VL_READER,
    )
    comp_min_side: int = arg(
        32,
        help="With --mask_dir: ignore a mask component whose shorter side is under "
        "this many pixels — screentone, not lettering",
        group=VL_READER,
    )
    comp_max: int = arg(
        16,
        help="With --mask_dir: read at most this many uncovered components per image, "
        "largest first",
        group=VL_READER,
    )
    apply: bool = arg(
        False, help="Write the sidecars (default: dry run). Touches no caption"
    )
    report_dir: str = _report_dir(f"{WS.REPORTS}/ocr")
    device: str | None = arg(None, help=DEVICE_HELP)

    def __post_init__(self) -> None:
        for name in ("det_conf", "min_det", "min_score"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"--{name} must be within 0-1")
        if self.min_chars < 0:
            raise ValueError("--min_chars must be 0 or more")


def _res(value) -> tuple[int, ...] | None:
    return None if value is None else tuple(int(v) for v in value)


def _margins(value) -> tuple[float, float, float, float] | None:
    return None if value is None else tuple(float(v) for v in value)


def _rel_paths(value) -> tuple[str, ...]:
    return tuple(str(v) for v in (value or ()))


@dataclass(frozen=True, kw_only=True)
class ResizeRequest(DatasetRequest):
    """Resize the caption master into the bucket-resolution tree every stage reads.

    Every other stage walks ``--dst``, so an image that exists only under
    ``--src`` is invisible to them. Each image lands in the ``--target_res``
    tier that resizes it the least, keeping its native aspect inside that tier's
    token band; the geometry matches the trainer's ``make preprocess-resize``, so
    whichever side runs first, the other one skips.

    Always writes; there is no dry run, and an image already at its target bucket is
    skipped without a re-decode. The glob matches against ``--src``.
    """

    path_pattern: str = arg("*", help=PATTERN_HELP.format(root="--src"))
    target_res: tuple[int, ...] | None = arg(
        None,
        read=_res,
        write=list,
        nargs="+",
        type=int,
        metavar="EDGE",
        help="Free-fit tiers (allowed: "
        + " ".join(str(e) for e in ALLOWED_TARGET_RES)
        + "). Each image lands in the tier that resizes it the least. Default "
        "(unset) = a single 1024 tier. Must match the trainer's configured "
        "target_res or both sides keep re-resizing each other.",
    )
    min_pixels: int = arg(
        DEFAULT_MIN_PIXELS,
        help=f"Skip images below this pixel count (default: {DEFAULT_MIN_PIXELS:,} = "
        "0.5MP; 0 disables). Smaller images would be upscaled to fill a tier.",
    )
    recursive: bool = arg(
        True, help="Walk subfolders, mirroring the layout under --dst (default: on)"
    )
    overwrite: bool = arg(
        False, help="Re-resize even images already at their target bucket"
    )
    workers: int = arg(4, help="Parallel worker processes (default: 4; 1 runs inline)")
    resize_crop_anchor: str = arg(
        DEFAULT_CROP_ANCHOR,
        choices=tuple(CROP_ANCHORS),
        help="Anchor for the residual cover-crop (default: center)",
    )
    resize_crop_margins: tuple[float, float, float, float] | None = arg(
        None,
        read=_margins,
        write=list,
        nargs=4,
        type=float,
        metavar=("TOP", "RIGHT", "BOTTOM", "LEFT"),
        help="Percent margins cropped from the source before resize (default: 0s)",
    )
    freefit_max_ratio: float = arg(
        4.0,
        help="Aspect-ratio clamp (default 4.0 = 1:4 / 4:1). Beyond-clamp images "
        "cover-crop to the limit; also keeps the token band solvable.",
    )
    skip: tuple[str, ...] = arg(
        (),
        read=_rel_paths,
        write=list,
        nargs="+",
        metavar="REL",
        help="Images to leave out, as paths relative to --src (forward slashes, "
        "exact — no glob), applied after --path_pattern. How the trainer GUI's "
        "per-image curation decisions (skip / move) reach the resize pass; a "
        "skipped image is written nowhere and is invisible to every later stage.",
    )
    excluded_dir: str = arg(
        WS.EXCLUDED,
        help="The exclusion ledger's tree (default: "
        f"{WS.EXCLUDED}). Every rel it lists is added to --skip, which is what "
        "makes an exclusion stick: this is the one stage that could put an "
        "excluded image back in the resized tree every later stage walks.",
    )
    report_dir: str = _report_dir(f"{WS.REPORTS}/resize")

    def __post_init__(self) -> None:
        if self.target_res:
            bad = [e for e in self.target_res if e not in ALLOWED_TARGET_RES]
            if bad:
                raise ValueError(
                    f"--target_res {bad} not in allowed tiers {list(ALLOWED_TARGET_RES)}"
                )
        if self.resize_crop_anchor not in CROP_ANCHORS:
            raise ValueError(
                f"--resize_crop_anchor must be one of {list(CROP_ANCHORS)}"
            )
        if self.resize_crop_margins is not None and len(self.resize_crop_margins) != 4:
            raise ValueError("--resize_crop_margins takes TOP RIGHT BOTTOM LEFT")


@dataclass(frozen=True, kw_only=True)
class ExportRequest(StageRequest):
    """Publish the workspace to the paths the trainer reads.

    The one operation in the package that writes outside ``workspace/``: resized
    images, masks and captions are copied under ``--out``, and whatever curation
    excluded is copied beside them under ``<out>/_excluded/``.

    Takes no ``--path_pattern``, which is why it is not a
    :class:`DatasetRequest`: an export is the workspace or it is a partial
    dataset the trainer reads as the whole one. Narrowing it is what the GUI's
    per-image Run would have meant, and one image published over a stale tree is
    never what "Export" was asked for.

    Dry-run by default. ``--apply`` copies for real, re-deciding every row against
    the destination, so a file edited since the dry run is reported rather than
    clobbered. Taking an export back is the GUI's Undo, not a flag here.
    """

    src: str = arg(
        WS.SOURCE_ROOT,
        help="Caption master dir — where a revised master publishes back to",
    )
    dst: str = arg(WS.RESIZED, help="Resized tree to publish (the workspace's)")
    masks: str = arg(WS.MASKS, help=f"Workspace mask dir (default: {WS.MASKS})")
    master: str = arg(
        WS.DEFAULT_ROOTS["master"],
        help="Revised-master overlay to publish over --src (default: "
        f"{WS.DEFAULT_ROOTS['master']})",
    )
    index: str = arg(
        DEFAULT_EXPORT_INDEX,
        help=f"caption_index.json to publish (default: {DEFAULT_EXPORT_INDEX})",
    )
    out: str = arg(
        WS.EXPORT_ROOT,
        help="Export root: resized/, masks/ and captions/ land under it (default: "
        f"{WS.EXPORT_ROOT}). The tree the trainer reads.",
    )
    excluded_dir: str = arg(
        WS.EXCLUDED,
        help=f"The excluded tree (default: {WS.EXCLUDED}), republished as it is "
        "under <out>/_excluded/ — beside the trainer's tree rather than in it, "
        "so an image curation took out is still there and is never trained on. "
        "A workspace where nothing is excluded publishes nothing extra.",
    )
    combine_ocr: bool = arg(
        False,
        gate="combine_ocr",
        group=OCR_DRAWER,
        help="Attach each image's OCR'd lines to the caption it publishes, as "
        "trailing clauses on the revised caption and on every line of its "
        '.variants.txt: speech as `Japanese text reads as "…", "…"`, sound '
        'effects (one per sound) as `Japanese SFX reads as "…"`; each line is '
        "said once — a repeat of the same speech verbatim, or of the same "
        "sound, is dropped. The workspace "
        "copies stay as they are; an image with no {stem}.ocr.txt publishes "
        "unchanged. Off by default: the trainer then encodes the caption as "
        "curated",
    )
    ocr_min_det: float = arg(
        0.5,
        gate="combine_ocr",
        group=OCR_DRAWER,
        help="Attach only the lines whose detector box confidence (the sidecar's "
        "det column) is at least this. The sidecar keeps every line for "
        "inspection; below 0.5 a box is mostly a nested fragment of its "
        "neighbour or a texture read as kana. 0 attaches everything",
    )
    ocr_min_glyph: float = arg(
        16.0,
        gate="combine_ocr",
        group=OCR_DRAWER,
        help="Attach only the lines whose glyphs are at least this many pixels "
        "across in the resized image — the box's area shared out between the "
        "characters read from it, sqrt(w*h/len). Below 16 the box is too small "
        "to hold what was read (a shop sign read as four kana in 39x22), or it "
        "is a watermark, credit line or narration strip too fine to train on. "
        "0 attaches everything",
    )
    ocr_dir: str = arg(
        WS.OCR,
        gate="combine_ocr",
        help=f"The OCR tree the sidecars are read from, mirroring --dst "
        f"(default: {WS.OCR}) — where the OCR stage wrote them",
    )
    apply: bool = arg(
        False, help="Copy for real (default: list what would be copied and stop)"
    )
    report_dir: str = _report_dir(f"{WS.REPORTS}/export")
