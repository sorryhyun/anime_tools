"""Stage registry for the web GUI: which CLIs are exposed, and how their
argparse parsers become a JSON form schema + back into an argv.

Torch/FastAPI-free on purpose. Every stage's ``build_parser()`` is imported
lazily, so a stage whose deps are missing is listed as *unavailable* rather than
breaking the server.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

REPLAY_FIELD = "from_report"
"""``--from_report``: the argparse dest a replay-capable stage exposes."""
REPLAY_REPORT_NAME = "apply_report.json"
"""What a replay writes, mirroring ``anime_tools.stages.replay``. Duplicated to
keep this module free of stage imports."""

GATE_ATTR = "gui_gate"
"""The attribute an argument group carries when it is a *drawer*: the dest of the
boolean that switches the whole group on. Set by
``anime_tools.masking._masks.gated_group``, spelled here to avoid importing it."""

CACHE_ENV = "ANIME_TOOLS_CACHE"
CACHE_VERSION = 2
"""Bumped when the on-disk schema cache format changes, so old files miss."""

_PATH_HINTS = (
    "dir",
    "path",
    "src",
    "dst",
    "out",
    "csv",
    "checkpoint",
    "config",
    "report",
    "accept",
    "revert",
    "manifest",
    "qwen3",
    "source",
)

_DIR_HINTS = ("dir", "src", "dst", "root", "source", "tokenizer", "qwen3")
"""Which of those name a *directory*, so the host's chooser opens in the mode that
can return one; everything else with a path in its name is a file."""


SETTINGS_KEY = "stage_defaults"
"""Where :data:`SETTING_FIELDS`' values live in the settings file."""

SETTING_FIELDS: dict[str, str] = {
    # argparse dest → settings key. Stage-independent knobs, set once in
    # ⚙ Settings; hidden from the form and filled by :func:`build_argv`.
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

Only the root is the setting; each generator keeps its own tail
(:func:`mask_subpath`). Both spell the flag ``--mask-dir``, so a shared value
would have them write ``{stem}_mask.png`` over each other at the same relative
path and leave ``merge_masks`` one tree to union instead of two. Blank means
beside the ``masks`` root.
"""

MASK_FIELDS: dict[str, str] = {
    # stage id → the dest naming a mask tree under :data:`MASK_SETTING`.
    "masks_sam": "mask_dir",
    "masks_mit": "mask_dir",
    # The merge *reads* both generators' trees, so it moves with them; its tail
    # is a list because the flag is.
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
    "ocr": frozenset({"min_score", "min_chars", "skip_en"}),
    "groups": frozenset({"sim_min", "min_size"}),
    "masks_sam": frozenset(
        {"prompts", "focus_prompts", "threshold", "dilate", "force"}
    ),
    # The two detector switches are gates, so they stay whatever this says (see
    # :attr:`Field.advanced`); named anyway so the row is the whole basic form.
    "masks_mit": frozenset(
        {"use_sam", "sam_prompts", "use_mit", "ctd_gate", "dilate", "force"}
    ),
}
"""Which of each stage's own knobs the form shows before Advanced is on."""


AUTO_FIELDS: frozenset[str] = frozenset({"device"})
"""Dests the GUI neither shows nor sends. This process is torch-free, so it
cannot see whether the *child* will find a GPU; the stage resolves ``--device``
itself."""


ROOT_FIELDS: dict[str, dict[str, str]] = {
    # stage id → {argparse dest: dataset root name}, filled from the Settings
    # dialog's dataset roots so no stage form re-asks for them.
    "resize": {"src": "src", "dst": "dst"},
    "autotag": {"src": "src", "dst": "dst"},
    "position": {"src": "src", "dst": "dst"},
    "correct": {"src": "src", "dst": "dst"},
    "audit": {"src": "src", "dst": "dst"},
    # OCR, grouping and the two mask generators read the *resized* tree: one
    # geometry for the whole pipeline. A mask cut from master pixels lands off the
    # subject for a ratio-clamped image. OCR binds no `src` — it reads no caption
    # and its sidecars go to their own tree.
    "ocr": {"dst": "dst"},
    "groups": {"source_dir": "dst"},
    "masks_sam": {"image_dir": "dst"},
    "masks_mit": {"image_dir": "dst"},
    # Only the *merged* output is the masks root; each generator's --mask-dir
    # is an intermediate that has to differ from it, so it stays on the form.
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


@dataclass(frozen=True)
class Stage:
    id: str
    title: str
    module: str
    panel: str
    """Which dock button this stage lives under; several stages share one, and
    the panel picks between them."""
    extra: str
    """Feature area (``tagger`` / ``stages`` / ...); informational only."""
    report: tuple[str, str | None] | None = None
    """``(dest, filename)``: the form field naming the report dir (or file when
    ``filename`` is None), so the GUI can fetch the result after a run."""
    notes: str = ""
    short: str = ""
    """Label for the in-panel picker; defaults to :attr:`title`."""
    hidden: bool = False
    """Keep this stage out of the dock. It still has a schema and an argv, so
    it can run as a preflight and be configured from Settings."""


STAGES: tuple[Stage, ...] = (
    Stage(
        "resize",
        "Resize to buckets",
        "anime_tools.stages.cli.resize_images",
        "Resize",
        "stages",
        report=("report_dir", "report.json"),
        hidden=True,
        notes=(
            "Runs automatically before every stage that reads the resized "
            "tree. These defaults apply to all of them."
        ),
    ),
    Stage(
        "autotag",
        "Autotag captions",
        "anime_tools.stages.cli.autotag_captions",
        "Autotag",
        "tagger",
        report=("report_dir", "report.json"),
        notes="Only `missing` mode is non-destructive.",
    ),
    Stage(
        "position",
        "Position captions",
        "anime_tools.stages.cli.position_captions",
        "Curate",
        "stages",
        report=("report_dir", "report.json"),
        short="Position",
    ),
    Stage(
        "correct",
        "Correct + mirror captions",
        "anime_tools.stages.cli.correct_captions",
        "Curate",
        "tokenizers",
        notes="Writes the revised captions under the resized tree; the master is never edited.",
        short="Correct",
    ),
    Stage(
        "audit",
        "Multiview audit",
        "anime_tools.stages.cli.audit_multiview",
        "Curate",
        "stages",
        report=("report_dir", "report.json"),
        short="Audit",
    ),
    Stage(
        "ocr",
        "OCR text",
        "anime_tools.stages.cli.ocr_captions",
        "OCR",
        "stages",
        report=("report_dir", "report.json"),
        notes=(
            "Writes {stem}.ocr.txt into the OCR tree, mirroring the resized "
            "tree. Reads and writes no caption."
        ),
    ),
    Stage(
        "groups",
        "Build groups",
        "anime_tools.grouping.cli.build_groups",
        "Groups",
        "grouping",
        report=("out", None),
    ),
    Stage(
        "masks_sam",
        "SAM3 subject masks",
        "anime_tools.masking.cli.generate_masks",
        "Masks",
        "masking",
        short="Subject",
    ),
    Stage(
        "masks_mit",
        "Text masks",
        "anime_tools.masking.cli.generate_masks_mit",
        "Masks",
        "masking",
        short="Text",
        notes=(
            "Two detectors, each behind its own switch: SAM3 on a prompt "
            "(balloons) and the UNet++ segmenter (lettering). Their masks are "
            "unioned."
        ),
    ),
    Stage(
        "masks_merge",
        "Merge masks",
        "anime_tools.masking.cli.merge_masks",
        "Masks",
        "masking",
        short="Merge",
    ),
    Stage(
        "export",
        "Export workspace",
        "anime_tools.stages.cli.export_workspace",
        "Export",
        "stages",
        report=("report_dir", "report.json"),
        notes=(
            "The only stage that writes outside the workspace. Run shows what "
            "would publish; Apply copies it to the tree the trainer reads."
        ),
    ),
)


PANELS: tuple[str, ...] = tuple(dict.fromkeys(s.panel for s in STAGES if not s.hidden))
"""The dock's buttons, in registry order. Hidden stages contribute none."""


NO_PREFLIGHT: frozenset[str] = frozenset({PREPROCESS_STAGE, "export"})
"""Stages the resize preflight never runs in front of.

``export`` is bound to ``dst`` but publishes the resized tree rather than
consuming it, so an empty one is a refusal, not a hidden step.
"""


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


BY_ID: dict[str, Stage] = {s.id: s for s in STAGES}


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


@dataclass
class Field:
    dest: str
    kind: str  # bool | int | float | str | enum | list
    flags: list[str] = field(default_factory=list)  # [] → positional
    default: Any = None
    choices: list[Any] | None = None
    help: str = ""
    required: bool = False
    path: bool = False
    path_kind: str = "dir"
    """``dir`` | ``file``: which chooser the ``…`` beside a ``path`` field
    opens. Meaningless unless ``path``."""
    group: str = ""
    negate: str | None = None
    """For BooleanOptionalAction: the ``--no-…`` flag."""
    label: str = ""
    """What the form shows: the flag, or the dest for a ``store_false`` flag so
    a ticked box always means *on*."""
    root: str | None = None
    """Bound to a dataset root: hidden from the form, filled by :func:`build_argv`
    from the Settings roots."""
    setting: str | None = None
    """Bound to a :data:`SETTING_FIELDS` key: hidden, filled from the Settings
    dialog's stage defaults."""
    report: str | None = None
    """This stage's own path under the :data:`REPORT_SETTING` root: hidden, filled
    by :func:`build_argv` as ``<root>/<report>``."""
    mask: str | list[str] | None = None
    """This stage's own tail(s) under the :data:`MASK_SETTING` root, bound and
    hidden like :attr:`report`. A list for the merge, which names both generators'
    trees in one flag."""
    auto: bool = False
    """In :data:`AUTO_FIELDS`: never shown, never sent, always auto-detected."""
    overridable: bool = False
    """In :data:`PANEL_FIELDS`: bound but *shown*, and a value the form sends wins
    for that run. Its :attr:`default` is the bound value once the schema has been
    through :func:`resolved_schema`."""
    advanced: bool = False
    """Not in this stage's :data:`BASIC_FIELDS` row: an ordinary form field folded
    away until the Advanced toggle is on. Never set on a required field or on a
    drawer's own gate."""
    gate: str | None = None
    """The dest of the boolean this field hangs off — a *drawer* (see
    :data:`GATE_ATTR`). The gate carries its own dest here, which is how the form
    tells the checkbox from what it folds away. A shut drawer's fields never reach
    the argv."""


def load_parser(stage: Stage) -> argparse.ArgumentParser:
    return importlib.import_module(stage.module).build_parser()


def _kind(a: argparse.Action) -> str:
    if isinstance(
        a,
        argparse.BooleanOptionalAction
        | argparse._StoreTrueAction
        | argparse._StoreFalseAction,
    ):
        return "bool"
    if a.nargs in ("+", "*") or (isinstance(a.nargs, int) and a.nargs > 1):
        return "list"
    if a.choices:
        return "enum"
    if a.type is int:
        return "int"
    if a.type is float:
        return "float"
    return "str"


def fields_of(parser: argparse.ArgumentParser) -> list[Field]:
    groups: dict[int, str] = {}
    gates: dict[int, str] = {}
    for g in parser._action_groups:
        if g.title not in ("positional arguments", "options", "optional arguments"):
            gate = getattr(g, GATE_ATTR, None)
            for a in g._group_actions:
                groups[id(a)] = g.title or ""
                if gate:
                    gates[id(a)] = gate
    out: list[Field] = []
    for a in parser._actions:
        if isinstance(a, argparse._HelpAction):
            continue
        kind = _kind(a)
        flags = list(a.option_strings)
        negate = None
        if isinstance(a, argparse.BooleanOptionalAction):
            negate = next((f for f in flags if f.startswith("--no-")), None)
            flags = [f for f in flags if not f.startswith("--no-")]
        default = a.default
        if default is argparse.SUPPRESS:
            default = None
        if kind == "bool":
            default = bool(default)
        name = a.dest.replace("_", "-")
        label = (
            a.dest
            if isinstance(a, argparse._StoreFalseAction)
            else (flags[0] if flags else a.dest)
        )
        out.append(
            Field(
                dest=a.dest,
                kind=kind,
                flags=flags,
                default=default,
                choices=list(a.choices) if a.choices else None,
                help=(a.help or "").replace("%(default)s", str(default)),
                # A positional is required unless argparse can do without it:
                # ``nargs="*"`` plus a default is the one shape that can, and
                # clearing such a field falls back to that default.
                required=bool(a.required) or (not flags and a.default is None),
                path=any(h in name for h in _PATH_HINTS),
                path_kind="dir" if any(h in name for h in _DIR_HINTS) else "file",
                group=groups.get(id(a), ""),
                negate=negate,
                label=label,
                gate=gates.get(id(a)),
            )
        )
    return out


def schema(stage: Stage) -> dict[str, Any]:
    """JSON-ready description of one stage, or its unavailability."""
    base = {
        "id": stage.id,
        "title": stage.title,
        "panel": stage.panel,
        "short": stage.short or stage.title,
        "module": stage.module,
        "extra": stage.extra,
        "notes": stage.notes,
        "report": bool(stage.report),
        "hidden": stage.hidden,
        # The preflight this stage gets, so the run bar can name it up front.
        "preprocess": preprocess_for(stage.id),
    }
    try:
        parser = load_parser(stage)
    except ImportError as e:  # extra not installed
        return {**base, "available": False, "error": str(e), "fields": [], "doc": ""}
    fs = fields_of(parser)
    bound = ROOT_FIELDS.get(stage.id, {})
    basic = BASIC_FIELDS.get(stage.id)
    report_dests = {stage.report[0] if stage.report else None} | {
        REPORT_INPUTS.get(stage.id)
    }
    for f in fs:
        f.root = bound.get(f.dest)
        f.setting = SETTING_FIELDS.get(f.dest)
        f.auto = f.dest in AUTO_FIELDS
        if f.dest in report_dests and isinstance(f.default, str):
            f.report = report_subpath(f.default)
        if f.dest == MASK_FIELDS.get(stage.id) and f.default is not None:
            f.mask = mask_subpath(f.default)
        f.overridable = f.dest in PANEL_FIELDS.get(stage.id, frozenset())
        # Only a field the form actually shows can be folded away; a bound or
        # auto one is already hidden.
        f.advanced = bool(
            basic is not None
            and f.dest not in basic
            and (
                f.overridable
                or not (f.root or f.setting or f.report or f.mask or f.auto)
            )
            and not f.required
            and f.gate != f.dest
            and f.dest not in ("apply", REPLAY_FIELD)
        )
        if f.root or f.report or f.mask:
            # A bound field names a path by construction, so it says so rather
            # than waiting for its flag's name to hint at it. That only matters
            # for a field on a form (:data:`PANEL_FIELDS`), where it puts the
            # right ``…`` chooser beside Export's destinations.
            f.path = True
            f.path_kind = (
                "file"
                if not f.root and PurePosixPath(str(f.default or "")).suffix
                else "dir"
            )
    return {
        **base,
        "available": True,
        "doc": parser.description or "",
        "apply": any(f.dest == "apply" for f in fs),
        # Takes a ``--path_pattern``, so one run can be narrowed to one image.
        "scoped": any(f.dest == SCOPE_FIELD for f in fs),
        # Can write a dry run's proposals instead of recomputing them.
        "replay": any(f.dest == REPLAY_FIELD for f in fs),
        "fields": [f.__dict__ for f in fs],
    }


def _under(root: str, tail: str | list[str]) -> str | list[str]:
    """Join one tail, or every tail of a list field, onto a Settings root."""
    if isinstance(tail, list):
        return [f"{root}/{t}" for t in tail]
    return f"{root}/{tail}"


def bound_value(
    f: Field,
    *,
    roots: dict[str, str] | None = None,
    settings: dict[str, str] | None = None,
    report_root: str | None = None,
    mask_root: str | None = None,
) -> str | list[str] | None:
    """What Settings says this field is, or ``None`` if it is not bound.

    Read by both :func:`build_argv` and :func:`resolved_schema`, so the argv and
    the form cannot disagree.
    """
    if f.root:
        return (roots or {}).get(f.root) or None
    if f.setting:
        return (settings or {}).get(f.setting) or None
    if f.report:
        return f"{report_root}/{f.report}" if report_root else None
    if f.mask:
        return _under(mask_root, f.mask) if mask_root else None
    return None


def resolved_schema(
    sc: dict[str, Any],
    *,
    roots: dict[str, str] | None = None,
    settings: dict[str, str] | None = None,
    report_root: str | None = None,
    mask_root: str | None = None,
) -> dict[str, Any]:
    """``sc`` with every :attr:`Field.overridable` default replaced by what
    Settings currently says (:func:`bound_value`).

    The dump is collected once in a child interpreter and cached on the source
    tree, so the bound values a *form* has to show are filled in here, per
    request. Returns a copy, leaving the cached dump alone.
    """
    fields = sc.get("fields") or []
    if not any(f.get("overridable") for f in fields):
        return sc
    out = []
    for fd in fields:
        v = (
            bound_value(
                Field(**fd),
                roots=roots,
                settings=settings,
                report_root=report_root,
                mask_root=mask_root,
            )
            if fd.get("overridable")
            else None
        )
        out.append({**fd, "default": v} if v is not None else fd)
    return {**sc, "fields": out}


def build_argv(
    fields: list[dict[str, Any]],
    values: dict[str, Any],
    *,
    apply: bool = False,
    roots: dict[str, str] | None = None,
    settings: dict[str, str] | None = None,
    report_root: str | None = None,
    mask_root: str | None = None,
) -> list[str]:
    """Turn a ``{dest: value}`` form payload into argv for ``python -m <module>``.

    ``fields`` is the ``schema()["fields"]`` list, so the server never imports the
    stage module. A value equal to the parser default (or empty) is omitted;
    ``apply`` toggles ``--apply`` regardless of what the form sent.

    ``roots``, ``settings``, ``report_root`` and ``mask_root`` fill the bound
    fields (:data:`ROOT_FIELDS`, :data:`SETTING_FIELDS`, :attr:`Field.report`,
    :attr:`Field.mask`), overriding whatever the form sent — each root joined to
    the stage's own tail. Narrowing a run to one image is a ``path_pattern``
    handed in through ``settings``. :data:`AUTO_FIELDS` never reach the argv.
    """
    argv: list[str] = []
    positional: list[str] = []
    fs = [Field(**fd) if isinstance(fd, dict) else fd for fd in fields]
    # A drawer's own checkbox decides whether the rest of it is even a value: the
    # stage ignores the knobs of a detector it is not running.
    gate_on = {
        f.dest: bool(values.get(f.dest, f.default)) for f in fs if f.gate == f.dest
    }
    for f in fs:
        bound = bool(f.root or f.setting or f.report or f.mask)
        if f.auto or f.dest in AUTO_FIELDS:
            continue
        if f.gate and f.gate != f.dest and not gate_on.get(f.gate, True):
            continue
        if f.dest == "apply":
            if apply:
                argv.append(f.flags[0])
            continue
        if bound:
            # Bound fields come from Settings only, so a stale value in a saved
            # form cannot win over the roots or the pattern the user set. The
            # exception is a :data:`PANEL_FIELDS` dest, where a typed value wins
            # and a blank means "whatever Settings says".
            v = (
                bound_value(
                    f,
                    roots=roots,
                    settings=settings,
                    report_root=report_root,
                    mask_root=mask_root,
                )
                or ""
            )
            if f.overridable:
                typed = values.get(f.dest)
                v = typed if str(typed or "").strip() else v
        else:
            v = values.get(f.dest, f.default)
        if f.kind == "bool":
            v = bool(v)
            if v == f.default:
                continue
            if f.negate is not None:
                argv.append(f.flags[0] if v else f.negate)
            else:
                argv.append(f.flags[0])
            continue
        if v is None or v == "" or v == []:
            if f.required:
                raise ValueError(f"{f.flags[0] if f.flags else f.dest} is required")
            continue
        if f.kind == "list":
            items = v if isinstance(v, list) else str(v).split("\n")
            items = [str(x).strip() for x in items if str(x).strip()]
            if not items:
                if f.required:
                    raise ValueError(f"{f.dest} is required")
                continue
            if f.flags:
                argv += [f.flags[0], *items]
            else:
                positional += items
            continue
        if f.kind == "int":
            v = int(v)
        elif f.kind == "float":
            v = float(v)
        # An overridable field is spelled out even when it equals the default,
        # since :func:`resolved_schema` may have *made* the Settings value that
        # default; dropping it would hand the run back to the CLI's own.
        if (
            not f.overridable
            and f.default is not None
            and v == f.default
            and not f.required
        ):
            continue
        if f.flags:
            argv += [f.flags[0], str(v)]
        else:
            positional.append(str(v))
    return argv + positional


def form_values(fields: list[dict[str, Any]], values: dict[str, Any]) -> dict[str, Any]:
    """``values`` minus everything the form does not own.

    The GUI persists the last form per stage; bound and auto-detected dests are
    not the form's to remember. A :data:`PANEL_FIELDS` dest is the exception: it
    is bound *and* the form's, so it is kept.
    """
    drop = {
        f["dest"]
        for f in fields
        if not f.get("overridable")
        and (
            f.get("root")
            or f.get("setting")
            or f.get("report")
            or f.get("auto")
            or f["dest"] in AUTO_FIELDS
        )
    }
    return {k: v for k, v in values.items() if k not in drop}


def report_path(
    stage: Stage,
    fields: list[dict[str, Any]],
    values: dict[str, Any],
    report_root: str | None = None,
) -> str | None:
    """Where this stage's report lands for this run (unresolved).

    The same answer :func:`build_argv` puts on the argv; with no ``report_root``
    this falls back to the CLI's own default. A replay (``--from_report``) writes
    :data:`REPLAY_REPORT_NAME` instead, because the two flags normally name the
    same directory and the stage refuses to clobber the dry run it replays.
    """
    if not stage.report:
        return None
    dest, filename = stage.report
    default = next((f["default"] for f in fields if f["dest"] == dest), None)
    tail = next((f.get("report") for f in fields if f["dest"] == dest), None)
    base = f"{report_root}/{tail}" if report_root and tail else default
    if not base:
        return None
    if filename and values.get(REPLAY_FIELD):
        filename = REPLAY_REPORT_NAME
    return f"{base}/{filename}" if filename else str(base)


def dump_schemas() -> dict[str, dict[str, Any]]:
    """Every stage's schema, keyed by id. Imports every stage CLI module, so
    call it in a child process from anything long-lived."""
    return {s.id: schema(s) for s in STAGES}


def dump_schemas_in_child() -> dict[str, dict[str, Any]]:
    """:func:`dump_schemas` in a fresh interpreter, so the caller stays torch-free."""
    code = (
        "import json, anime_tools.gui.stages as S; print(json.dumps(S.dump_schemas()))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        # json.dumps escapes non-ASCII, but a traceback on stderr does not, and
        # the locale codec (cp949/cp1252 on Windows) would raise on it.
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"stage schema dump failed:\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---- on-disk memo for the child dump ------------------------------------
# The child interpreter is on the GUI's startup path. Stage CLIs defer their heavy
# imports, so the dump is ~0.2s and the memo only keeps startup flat if a stage
# regains a slow import.


def cache_dir() -> Path:
    """Where the GUI keeps derived, throw-away state — outside the curation home.
    ``$ANIME_TOOLS_CACHE`` overrides."""
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "anime_tools" / "gui"


def schema_cache_path() -> Path:
    return cache_dir() / "schemas.json"


def schema_cache_key() -> str:
    """What has to change for a cached dump to be wrong: the installed version, the
    interpreter, and ``(path, mtime_ns, size)`` for every ``.py`` under the package
    — whole-package, because a parser's defaults routinely come from the stage
    module behind its CLI. :func:`importlib.util.find_spec` keys on a stage module
    outside the package without importing it.
    """
    parts = [f"v{CACHE_VERSION}", _distribution_version(), sys.version]
    files = set(Path(__file__).resolve().parent.parent.rglob("*.py"))
    for s in STAGES:
        try:
            spec = importlib.util.find_spec(s.module)
        except (ImportError, ValueError, AttributeError):
            spec = None
        origin = spec.origin if spec is not None else None
        parts.append(f"{s.id}={s.module}@{origin}")
        if origin:
            files.add(Path(origin).resolve())
    for f in sorted(files):
        try:
            st = f.stat()
        except OSError:
            parts.append(f"{f}:gone")
        else:
            parts.append(f"{f}:{st.st_mtime_ns}:{st.st_size}")
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _distribution_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("anime_tools")
    except (PackageNotFoundError, ImportError, ValueError):
        return "unknown"


def _read_schema_cache(path: Path, key: str) -> dict[str, dict[str, Any]] | None:
    """The cached dump if it is still valid. Anything else — missing, truncated,
    stale — is ``None`` ("dump again"), never an error."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(blob, dict) or blob.get("key") != key:
        return None
    schemas = blob.get("schemas")
    if not isinstance(schemas, dict) or not schemas:
        return None
    return schemas


def _write_schema_cache(
    path: Path, key: str, schemas: dict[str, dict[str, Any]]
) -> None:
    """Best effort: a read-only or full cache dir costs a rebuild, not a start."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"key": key, "schemas": schemas}), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def load_schemas(*, cache: bool = True) -> dict[str, dict[str, Any]]:
    """Every stage's schema, from :func:`schema_cache_path` when it is still
    valid and from a fresh child interpreter otherwise."""
    if not cache:
        return dump_schemas_in_child()
    try:
        key, path = schema_cache_key(), schema_cache_path()
    except (OSError, RuntimeError):
        # No usable cache location (``Path.home()`` unresolvable, package dir
        # gone); the cache is an optimisation, so fall through to a fresh dump.
        return dump_schemas_in_child()
    cached = _read_schema_cache(path, key)
    if cached is not None:
        return cached
    schemas = dump_schemas_in_child()
    _write_schema_cache(path, key, schemas)
    return schemas
