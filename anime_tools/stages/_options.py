"""Every value a stage request field *defaults to*, in a module that imports
nothing heavier than the stdlib.

``registry.py`` is deliberately import-light, so a caller can list the stages
without importing one — but the GUI resolves each request class to build its
form, and a request declaring ``arg(DEFAULT_MIN_PIXELS)`` used to reach for the
whole stage module to read one number. Measured: that cost the schema build
numpy, PIL and yaml. So the numbers live here and each owning stage re-exports
them, which is also what keeps a default spelled once.

:class:`PositionCaptionOptions` sits here for the same reason: it is a plain
dataclass of floats, and ``DetectionRequest.options()`` builds one field by
field.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CROP_ANCHORS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CROP_ANCHOR",
    "DEFAULT_IDENTITY_CONFIDENCE",
    "DEFAULT_MIN_PIXELS",
    "DEFAULT_MULTIVIEW_PROB",
    "EXTRA_CHARACTER",
    "MULTIPLE_VIEWS",
    "PositionCaptionOptions",
]

# --- resize (``stages/resize.py``) -------------------------------------------

DEFAULT_MIN_PIXELS = 500_000
"""0.5MP. Below this an image cannot fill a 1024 tier without visible upscale."""

DEFAULT_CROP_ANCHOR = "center"
CROP_ANCHORS: dict[str, tuple[float, float]] = {
    "top_left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}

# --- autotag (``stages/autotag.py``) -----------------------------------------

DEFAULT_BATCH_SIZE = 8
"""Images per tagger forward. The backbone is happiest with a batch and the walk
has to decode the next image anyway; eight is small enough that the whole batch's
decoded pixels are nothing next to the weights."""

# --- multiview audit (``stages/multiview_audit.py``) -------------------------

MULTIPLE_VIEWS = "multiple views"
"""The tag that says one character is drawn several times — the verdict the audit
proposes, and the clause pipeline's own signal that a layout is repeated."""
EXTRA_CHARACTER = "extra-character"
"""The other verdict: the second box is a second character, so the count is wrong."""

# P(multiple views) from the whole-image tagger at which it counts as a witness.
DEFAULT_MULTIVIEW_PROB = 0.5
# Probability an identity-group winner needs before it is believed. Measured: a
# legible face scores 0.978-1.000 on hair/eye colour, a headless panel's
# invented values land at 0.54-0.63.
DEFAULT_IDENTITY_CONFIDENCE = 0.9


# --- position clauses (``stages/position_captions.py``) ----------------------


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
    # On by default, unlike its box counterpart — see ``mask_containment``.
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
    # character's own traits and name out of every clause: they belong to the
    # girl, not to a view of her.
    multi_view_gate: bool = True
    # Let a clause say which *view* it describes (`close-up`, `full body`).
    bind_framing: bool = True
    # Let a view layout's clause carry the anatomy visible in that panel.
    bind_view_anatomy: bool = True
    # Bag-tag keep relaxation (1.0 = off): a bag tag can only MOVE into a clause,
    # never be invented, so the crop tagger only has to *localize* it and its
    # per-tag F1 threshold may be relaxed — which recovers pose tags whose scores
    # collapse once mask-blanking removes the scene context. Applied before the
    # attributable/shared census, so a rival crop's borderline score also BLOCKS
    # a move the strict kept sets allowed.
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
    # only the removal — a tag that fails still enters its clause. 0.0 = trust
    # the tagger's thresholds alone.
    attribution_margin: float = 0.25
