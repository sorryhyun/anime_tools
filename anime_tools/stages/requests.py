"""The caption stages as request objects — the surface the CLIs, the GUI and the
trainer share (``docs/api_first_plan.md``).

Torch-free: run one through :mod:`anime_tools.stages.run` (``run_autotag(req)``
and friends, which import the models), or hand ``to_argv()`` to a subprocess.
Every field is an argparse ``dest`` of the matching ``stages/cli`` parser, and
the defaults are the CLI's. Flags are spelled with underscores here
(``--path_pattern``), the caption stages' canonical form.

The SAM3 detection flags are one nested :class:`DetectionRequest`, so the
position stage and the multiview audit cannot declare the same detector twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import ClassVar

from anime_tools import workspace as WS
from anime_tools._request import OFF, READ, WRITE, Request
from anime_tools.buckets import ALLOWED_TARGET_RES
from anime_tools.contract import AUTOTAG_MODES
from anime_tools.downloads import DEFAULT_SAM3_CHECKPOINT, DEFAULT_SUBJECT_PROMPT_EMBED
from anime_tools.masking._sam3 import SUBJECT_PROMPT, prompt_list
from anime_tools.stages.multiview_audit import (
    DEFAULT_IDENTITY_CONFIDENCE,
    DEFAULT_MULTIVIEW_PROB,
    MULTIPLE_VIEWS,
)
from anime_tools.stages.position_captions import PositionCaptionOptions
from anime_tools.stages.resize import (
    CROP_ANCHORS,
    DEFAULT_CROP_ANCHOR,
    DEFAULT_MIN_PIXELS,
)

__all__ = [
    "AuditRequest",
    "AutotagRequest",
    "CorrectRequest",
    "DetectionRequest",
    "ExportRequest",
    "OcrRequest",
    "PositionRequest",
    "ResizeRequest",
]

DEFAULT_MAX_TOKENS = 512
"""Both tokenizers pad to this; a caption past it truncates silently."""

DEFAULT_EXPORT_INDEX = f"{WS.REPORTS}/caption_index.json"


def _csv(default: tuple[str, ...], **meta) -> tuple[str, ...]:
    """A comma-separated flag held as a tuple."""
    return field(default=default, metadata={READ: prompt_list, WRITE: ",".join, **meta})


def _off(default: bool, flag: str) -> bool:
    """A ``store_false`` switch: ``flag`` is the one spelling that flips it."""
    return field(default=default, metadata={OFF: flag})


@dataclass(frozen=True, kw_only=True)
class StageRequest(Request):
    FLAG_SEP: ClassVar[str] = "_"


@dataclass(frozen=True, kw_only=True)
class DatasetRequest(StageRequest):
    """The three dataset roots every caption stage walks (``_args.add_dataset_args``)."""

    src: str = "image_dataset"
    """The caption master — read-only for every stage but the audit."""
    dst: str = WS.RESIZED
    """The resized tree: what every stage opens, and where the revised caption lands."""
    path_pattern: str = "*"
    """fnmatch glob (``|`` to OR-combine) on the path relative to the walked root."""


@dataclass(frozen=True, kw_only=True)
class ReplayRequest(DatasetRequest):
    """Dry-run by default, ``apply`` writes, ``from_report`` replays a dry run."""

    apply: bool = False
    from_report: str | None = None
    """A previous dry run's ``report.json``: write exactly what it proposed and
    load no model."""
    report_dir: str
    """Where ``report.json`` lands; distinct per stage so one stage's replay
    cannot read another's report."""


@dataclass(frozen=True, kw_only=True)
class TaggerRequest(StageRequest):
    """The Anima Tagger flags (``_args.add_model_args``)."""

    tagger_dir: str | None = None
    """Checkpoint dir; ``None`` is the shipped default (``DEFAULT_TAGGER_DIR``)."""
    device: str | None = None
    """``cuda`` / ``cpu``; ``None`` resolves at model-load time, never earlier,
    so the torch-free replay path never pays for the probe."""


@dataclass(frozen=True, kw_only=True)
class DetectionRequest(StageRequest):
    """The SAM3 detector both SAM3 stages run (``_detection.add_detection_args``):
    subject prompt, thresholds, the body-part fallback, and the dedupe geometry.

    Everything here but ``checkpoint`` and ``prompt_embed`` is a field of
    :class:`PositionCaptionOptions`, which is how :meth:`PositionRequest.options`
    builds the detection half.
    """

    prompt: str = SUBJECT_PROMPT
    prompt_embed: str = DEFAULT_SUBJECT_PROMPT_EMBED
    """The learned soft prompt standing in for ``prompt``; ``none`` for text."""
    checkpoint: str = DEFAULT_SAM3_CHECKPOINT
    """SAM3 weights."""
    score_threshold: float = 0.5
    retry_score_threshold: float = 0.35
    part_prompts: tuple[str, ...] = _csv(())
    """Body-part prompts tried only when the subject prompt undershoots."""
    part_score_threshold: float = 0.5
    part_containment_threshold: float = 0.7
    iou_threshold: float = 0.65
    containment_threshold: float = 1.01
    mask_containment_threshold: float = 0.8
    dedupe_fill_ratio: float = 2.0
    min_area_frac: float = 0.005
    pad: float = 0.06
    row_tol: float = 0.25
    max_instances: int = 8

    @property
    def floor(self) -> float:
        """The lowest confidence any pass asks for — the processor's own floor."""
        return min(
            self.score_threshold, self.retry_score_threshold, self.part_score_threshold
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


@dataclass(frozen=True, kw_only=True)
class AutotagRequest(ReplayRequest, TaggerRequest):
    """Batch Anima Tagger over the resized tree
    (``python -m anime_tools.stages.cli.autotag_captions``)."""

    mode: str = "missing"
    """``missing`` (only images no caption speaks for), ``merge`` (append novel
    tags, keeping clauses) or ``overwrite``."""
    min_confidence: float = 0.0
    """Extra probability floor on top of the tagger's per-tag thresholds; 0 = off."""
    report_dir: str = f"{WS.REPORTS}/autotag"

    def __post_init__(self) -> None:
        if self.mode not in AUTOTAG_MODES:
            raise ValueError(f"--mode must be one of {list(AUTOTAG_MODES)}")


@dataclass(frozen=True, kw_only=True)
class PositionRequest(ReplayRequest, TaggerRequest):
    """Position clauses: SAM3 instances → reading order → mask-blanked crops →
    tagger → clause rewrite (``python -m anime_tools.stages.cli.position_captions``).
    See ``docs/position_captions.md`` for the rules the knobs tune."""

    report_dir: str = f"{WS.REPORTS}/position"
    crops: bool = False
    """Also export the mask-blanked crops next to the report."""
    flatten: bool = False
    """The inverse pass: merge every clause back into the flat bag. Text-only."""
    detection: DetectionRequest = field(default_factory=DetectionRequest)
    blank_crops: bool = _off(True, "--no_blank_crops")
    min_instances: int = 2
    strict_count: bool = _off(True, "--no_strict_count")
    max_clause_tags: int = 8
    max_novel_tags: int = 1
    name_confidence: float = 0.5
    allow_unlisted_names: bool = False
    discriminative_only: bool = _off(True, "--keep_shared_tags")
    bag_gated_identity: bool = _off(True, "--ungated_identity")
    multi_view_gate: bool = _off(True, "--bind_view_traits")
    bind_view_anatomy: bool = _off(True, "--gate_view_anatomy")
    bind_framing: bool = _off(True, "--no_framing")
    rewrite: bool = _off(True, "--no_rewrite")
    bag_relax: float = 0.35
    bag_word_relax: float = 0.85
    bag_relax_min_score: float = 0.3
    attribution_margin: float = 0.25
    qwen3: str | None = None
    """Qwen3 tokenizer dir — enables the token-budget column in the report."""
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        if self.flatten and self.from_report:
            raise ValueError(
                "--flatten and --from_report are mutually exclusive: the flatten "
                "pass is already text-only, so there is no model pass to skip."
            )

    def options(self) -> PositionCaptionOptions:
        """The options one pass runs under: the detection block plus the
        clause-composition knobs."""
        return _options(self, self.detection)


@dataclass(frozen=True, kw_only=True)
class AuditRequest(ReplayRequest, TaggerRequest):
    """Audit ``1girl`` captions for several views of one girl
    (``python -m anime_tools.stages.cli.audit_multiview``). The only stage that
    writes the caption master."""

    report_dir: str = f"{WS.REPORTS}/multiview_audit"
    apply_verdicts: tuple[str, ...] = _csv((MULTIPLE_VIEWS,))
    """Verdicts ``apply`` may write."""
    apply_confidence: tuple[str, ...] = _csv(("strong",))
    """Confidence tiers ``apply`` may write (``strong``, ``weak``)."""
    crops: bool = False
    sheets: bool = _off(True, "--no_sheets")
    """The per-finding contact sheets — the review surface."""
    detection: DetectionRequest = field(default_factory=DetectionRequest)
    name_confidence: float = 0.5
    multiview_threshold: float = DEFAULT_MULTIVIEW_PROB
    identity_confidence: float = DEFAULT_IDENTITY_CONFIDENCE
    suggest_counts: bool = False

    MIN_INSTANCES: ClassVar[int] = 2
    """Pinned rather than exposed: two subjects is what the audit is for."""

    def options(self) -> PositionCaptionOptions:
        """The position stage's detector verbatim, with ``min_instances`` pinned
        and every other non-detection knob at its default."""
        return _options(
            self.detection,
            _Pinned(
                min_instances=self.MIN_INSTANCES, name_confidence=self.name_confidence
            ),
            PositionCaptionOptions(),
        )


@dataclass(frozen=True)
class _Pinned:
    min_instances: int
    name_confidence: float


@dataclass(frozen=True, kw_only=True)
class CorrectRequest(StageRequest):
    """Mirror the master into corrected revised captions, with optional variant
    sidecars (``python -m anime_tools.stages.cli.correct_captions``). Always
    writes: there is no dry run and no report."""

    src: str
    dst: str
    tag_csv: str | None = None
    """``danbooru_tags_classified.csv``; ``None`` looks under ``models/``."""
    path_pattern: str = "*"
    recursive: bool = False
    caption_insert_no_artist: bool = False
    caption_trigger_word: str = ""
    caption_trigger_at_front: bool = False
    caption_drop_groups: str = ""
    """Comma-separated tag groups to strip (``tag_drop_groups.parse_drop_groups``)."""
    no_correct: bool = False
    """Mirror the master verbatim as ``v0`` instead of bucket-reordering."""
    caption_shuffle_variants: int = 0
    caption_tag_dropout_rate: float = 0.0
    caption_tag_randomize_rate: float = 0.0
    qwen3: str | None = None
    t5_tokenizer_path: str | None = None

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
    """PP-OCRv6 over the resized tree, writing ``{stem}.ocr.txt`` sidecars
    (``python -m anime_tools.stages.cli.ocr_captions``). Touches no caption."""

    dst: str = WS.RESIZED
    ocr_dir: str = WS.OCR
    path_pattern: str = "*"
    min_score: float = 0.6
    min_chars: int = 3
    skip_en: bool = _off(True, "--keep_en")
    """Drop ASCII-only lines: on a scanned comic they are page numbers and sfx."""
    join_cjk: bool = _off(True, "--no_join_cjk")
    """Join the columns of one vertical balloon into one line."""
    min_box_px: int = 12
    max_boxes: int = 64
    det_limit_side: int = 1440
    batch_size: int = 8
    apply: bool = False
    report_dir: str = f"{WS.REPORTS}/ocr"
    device: str | None = None


def _res(value) -> tuple[int, ...] | None:
    return None if value is None else tuple(int(v) for v in value)


def _margins(value) -> tuple[float, float, float, float] | None:
    return None if value is None else tuple(float(v) for v in value)


@dataclass(frozen=True, kw_only=True)
class ResizeRequest(DatasetRequest):
    """Resize the master into the bucket-resolution tree every stage reads
    (``python -m anime_tools.stages.cli.resize_images``). Always writes; the glob
    matches against ``src``."""

    target_res: tuple[int, ...] | None = field(
        default=None, metadata={READ: _res, WRITE: list}
    )
    """Free-fit tiers; ``None`` is a single 1024 tier. Must match the trainer's."""
    min_pixels: int = DEFAULT_MIN_PIXELS
    """Skip images below this many pixels; 0 disables."""
    recursive: bool = True
    copy_captions: bool = False
    overwrite: bool = False
    workers: int = 4
    resize_crop_anchor: str = DEFAULT_CROP_ANCHOR
    resize_crop_margins: tuple[float, float, float, float] | None = field(
        default=None, metadata={READ: _margins, WRITE: list}
    )
    """Percent margins ``(top, right, bottom, left)`` cropped before the resize."""
    freefit_max_ratio: float = 4.0
    report_dir: str = f"{WS.REPORTS}/resize"

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
class ExportRequest(DatasetRequest):
    """Publish the workspace to the paths the trainer reads
    (``python -m anime_tools.stages.cli.export_workspace``) — the one stage that
    writes outside ``workspace/``."""

    masks: str = WS.MASKS
    master: str = WS.DEFAULT_ROOTS["master"]
    """The revised-master overlay, published over ``src``."""
    index: str = DEFAULT_EXPORT_INDEX
    out: str = WS.EXPORT_ROOT
    """The export root: ``resized/``, ``masks/`` and ``captions/`` land under it."""
    apply: bool = False
    report_dir: str = f"{WS.REPORTS}/export"
