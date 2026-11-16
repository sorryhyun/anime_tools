"""The GUI's field bindings: which stage flags are dataset roots, ⚙ Settings
values, report or mask tails, per-run panel choices or folded under Advanced —
and :class:`Bindings`, the values a request's settings fill them with.
Re-exported by :mod:`anime_tools.gui.stages`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anime_tools.gui._stage_field import Field


REPLAY_FIELD = "from_report"
"""``--from_report``: the dest a replay-capable stage exposes."""


SETTINGS_KEY = "stage_defaults"
"""Where :data:`SETTING_FIELDS`' values live in the settings file."""

SETTING_FIELDS: dict[str, str] = {
    # dest → settings key. Stage-independent knobs, set once in ⚙ Settings;
    # hidden from the form and filled by :func:`build_argv`.
    "path_pattern": "path_pattern",
    "tagger_dir": "tagger_dir",
    "checkpoint": "checkpoint",
    "prompt_embed": "prompt_embed",
}

REPORT_SETTING = "report_root"
"""The directory every stage's report lands under, in :data:`SETTINGS_KEY`.

Only the *root* is the setting; each stage keeps its own sub-path
(:func:`report_subpath`), so one stage's ``--from_report`` replay cannot read
another's report. Blank means "beside the ``dst`` root". The one report a stage
*reads* (:data:`REPORT_INPUTS`) is bound the same way.
"""

MASK_SETTING = "mask_root"
"""The directory each mask generator's own tree lands under, in
:data:`SETTINGS_KEY`.

Only the root is the setting; the generator keeps its own tail
(:func:`mask_subpath`) rather than writing into the ``masks`` root, so a second
tree merged in beside it (a hand-painted one, another tool's) cannot overwrite
its ``{stem}_mask.png`` at the same relative path. Blank means beside the
``masks`` root.
"""

MASK_FIELDS: dict[str, str] = {
    # stage id → the dest naming a mask tree under :data:`MASK_SETTING`.
    "masks_sam": "mask_dir",
    # The merge *reads* the generator's tree, so it moves with it; its tail is a
    # list because the flag is.
    "masks_merge": "mask_dirs",
}

REPORT_INPUTS: dict[str, str] = {
    # stage id → the dest naming a report this stage *reads*. Bound to
    # :data:`REPORT_SETTING` like the report a stage writes, but kept out of
    # :attr:`Stage.report`, which is the report this run produces.
    "export": "index",
}

PANEL_FIELDS: dict[str, frozenset[str]] = {
    # stage id → bound dests this stage's own *panel* may override.
    #
    # A bound field is normally hidden and filled from ⚙ Settings. Export is the
    # exception: where it publishes (``out``) and which caption index it publishes
    # (``index``) are per-run choices. They stay bound but stay on the form, each
    # opening on the Settings-derived value (:func:`resolved_schema`); typing over
    # one wins for that run only. Everything Export *reads* stays hidden.
    "export": frozenset({"out", "index"}),
}

SCOPE_FIELD = "path_pattern"
"""The :data:`SETTING_FIELDS` key the GUI narrows to run a stage on one image."""

PREPROCESS_STAGE = "resize"
"""The stage that populates the resized tree, run as a preflight rather than a dock
panel: every stage bound to ``dst`` walks ``workspace/resized/``, so an image only
in the caption master is invisible to it. Idempotent, so the preflight is cheap."""

EXPORT_STAGE = "export"
"""The one stage that writes outside the workspace — so the one a run over the
GUI's drop area (``gui.dataset.TEMP_SPACE``) is refused."""

PREPROCESS_SETTINGS_KEY = "preprocess"
"""Where the resize form's values live in the settings file. It has no panel, so
its knobs are set once in ⚙ Settings."""

BASIC_FIELDS: dict[str, frozenset[str]] = {
    # stage id → the dests that stay on the form when Advanced is off; everything
    # else on that stage folds away behind the form's one toggle. A stage with no
    # row here has no advanced fields at all.
    #
    # The rule for a row is what a run changes its mind about: what to detect, how
    # sure it has to be, how many there are, and what to write.
    "position": frozenset(
        {
            "crops",
            "flatten",
            "multiview_audit",
            "prompt",
            "score_threshold",
            "min_instances",
            "max_instances",
            "rewrite",
        }
    ),
    "audit": frozenset(
        {
            "apply_verdicts",
            "apply_confidence",
            "crops",
            "sheets",
            "prompt",
            "score_threshold",
            "max_instances",
        }
    ),
    "correct": frozenset(
        {
            "no_correct",
            "caption_trigger_word",
            "caption_insert_no_artist",
            "caption_drop_groups",
            "caption_shuffle_variants",
        }
    ),
    "drop_groups": frozenset({"groups", "keep_tags"}),
    "ocr": frozenset(
        {"det_conf", "min_det", "min_score", "min_chars", "skip_en", "strip_symbols"}
    ),
    "groups": frozenset({"sim_min", "min_size"}),
    "masks_sam": frozenset({"masks", "threshold", "dilate", "force"}),
}
"""Which of each stage's own knobs the form shows before Advanced is on."""


AUTO_FIELDS: frozenset[str] = frozenset({"device"})
"""Dests the GUI neither shows nor sends. This process is torch-free, so it
cannot see whether the *child* will find a GPU; the stage resolves ``--device``
itself."""


_CAPTION_ROOTS = {"src": "src", "master": "master", "dst": "dst"}

ROOT_FIELDS: dict[str, dict[str, str]] = {
    # stage id → {dest: dataset root name}, filled from the Settings dialog's
    # dataset roots so no stage form re-asks for them.
    "resize": {"src": "src", "dst": "dst"},
    # The caption stages read the ladder — revised, then the master overlay the
    # caption editor writes, then the hand-written master — so all three roots.
    # ``--masked`` paints out what the merged masks root (the tree Export
    # publishes, so the one the trainer's masked loss reads) ignores.
    "autotag": {**_CAPTION_ROOTS, "mask_dir": "masks"},
    "position": _CAPTION_ROOTS,
    "correct": _CAPTION_ROOTS,
    "drop_groups": _CAPTION_ROOTS,
    "audit": _CAPTION_ROOTS,
    # OCR, grouping and the mask generator read the *resized* tree: one
    # geometry for the whole pipeline. A mask cut from master pixels lands off the
    # subject for a ratio-clamped image. OCR binds no `src` — it reads no caption
    # and its sidecars go to their own tree.
    "ocr": {"dst": "dst"},
    "groups": {"source_dir": "dst"},
    "masks_sam": {"image_dir": "dst"},
    # Only the *merged* output is the masks root; the generator's --mask-dir is
    # an intermediate that has to differ from it, so it stays on the form.
    "masks_merge": {"output_dir": "masks"},
    # Runs the pipeline backwards: reads the workspace trees and writes ``src``
    # (a revised master) and ``out`` (everything else).
    "export": {
        "src": "src",
        "dst": "dst",
        "masks": "masks",
        "master": "master",
        "out": "out",
    },
}


NO_PREFLIGHT: frozenset[str] = frozenset({PREPROCESS_STAGE, EXPORT_STAGE})
"""Stages the resize preflight never runs in front of.

``export`` is bound to ``dst`` but publishes the resized tree rather than
consuming it, so an empty one is a refusal, not a hidden step.
"""

FOLLOW_UPS: dict[str, str] = {
    # The generator writes only its own tree; the merge is what fills the
    # ``masks`` root the sidebar shows and Export publishes, so a run that
    # stopped at the generator left both empty.
    "masks_sam": "masks_merge",
}
"""stage id → the stage that runs right after it in the same job, with that
stage's last saved form (so a hand-painted tree listed on the merge form is
merged in too)."""


def preprocess_for(stage_id: str) -> str | None:
    """The stage that must run before ``stage_id``, or ``None``.

    A stage bound to the ``dst`` root reads the resized tree, so it needs
    :data:`PREPROCESS_STAGE` in front of it, unless :data:`NO_PREFLIGHT` names it.
    """
    if stage_id in NO_PREFLIGHT:
        return None
    if "dst" in ROOT_FIELDS.get(stage_id, {}).values():
        return PREPROCESS_STAGE
    return None


def _tail(default: str) -> str:
    """A CLI default minus its first path component."""
    parts = PurePosixPath(default).parts
    return PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else default


def report_subpath(default: str) -> str:
    """The part of a report default that is *not* a dataset root.

    Every ``--report_dir`` / ``--out`` default is ``<dataset root>/<this stage's
    own path>``; dropping the first component lets one :data:`REPORT_SETTING` move
    every report at once while each stage keeps a directory of its own.
    """
    return _tail(default)


def mask_subpath(default: str | list[str]) -> str | list[str]:
    """The same, for a mask tree under :data:`MASK_SETTING`.

    A list in, a list out: the merge's inputs are one bound field naming both
    generators' tails, so they move together or the merge reads nothing.
    """
    if isinstance(default, list):
        return [_tail(d) for d in default]
    return _tail(default)


def _under(root: str, tail: str | list[str]) -> str | list[str]:
    """Join one tail, or every tail of a list field, onto a Settings root."""
    if isinstance(tail, list):
        return [f"{root}/{t}" for t in tail]
    return f"{root}/{tail}"


@dataclass(frozen=True)
class Bindings:
    """What a request's settings say the bound fields are: the values
    :func:`build_argv` and :func:`resolved_schema` fill a form's hidden fields
    with, one object rather than four keyword arguments threaded through both.

    ``roots`` fills :data:`ROOT_FIELDS`, ``settings`` the :data:`SETTING_FIELDS`
    (a ``path_pattern`` there is how a run is narrowed to one image), and the two
    roots each stage's own :attr:`Field.report` / :attr:`Field.mask` tail is
    joined onto. Built per request by ``RunContext.bindings()``.
    """

    roots: Mapping[str, str] = field(default_factory=dict)
    settings: Mapping[str, str] = field(default_factory=dict)
    report_root: str | None = None
    mask_root: str | None = None

    def value(self, f: Field) -> str | list[str] | None:
        """What Settings says ``f`` is, or ``None`` if it is not bound.

        Read by both :func:`build_argv` and :func:`resolved_schema`, so the argv
        and the form cannot disagree.
        """
        if f.root:
            return self.roots.get(f.root) or None
        if f.setting:
            return self.settings.get(f.setting) or None
        if f.report:
            return f"{self.report_root}/{f.report}" if self.report_root else None
        if f.mask:
            return _under(self.mask_root, f.mask) if self.mask_root else None
        return None


UNBOUND = Bindings()
"""No settings at all: every bound field falls back to the CLI's own default."""
