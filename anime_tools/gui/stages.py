"""The web GUI over the stage registry: how a stage's request dataclass becomes
a JSON form schema, and how a form payload becomes the request's argv.

The registry itself is :mod:`anime_tools.stages.registry` (re-exported here);
this module adds the GUI's *bindings* — which flags are dataset roots, which
are ⚙ Settings values, which fold under Advanced — and stays torch-free: a
request class is imported to be described (:func:`schema`) or built
(:func:`build_argv`), and a stage whose deps are missing is listed as
*unavailable* rather than breaking the server.
"""

from __future__ import annotations

import inspect
from dataclasses import MISSING, dataclass, field
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any

from anime_tools._request import Arg, args_of
from anime_tools.contract import GATE_ATTR, REPLAY_REPORT_NAME
from anime_tools.stages.registry import BY_ID, PANELS, STAGES, Stage, request_class

__all__ = [
    "AUTO_FIELDS",
    "BASIC_FIELDS",
    "BY_ID",
    "GATE_ATTR",
    "MASK_FIELDS",
    "MASK_SETTING",
    "NO_PREFLIGHT",
    "PANELS",
    "PANEL_FIELDS",
    "PREPROCESS_SETTINGS_KEY",
    "PREPROCESS_STAGE",
    "REPLAY_FIELD",
    "REPLAY_REPORT_NAME",
    "REPORT_INPUTS",
    "REPORT_SETTING",
    "ROOT_FIELDS",
    "SCOPE_FIELD",
    "SETTINGS_KEY",
    "SETTING_FIELDS",
    "STAGES",
    "Field",
    "Stage",
    "bound_value",
    "build_argv",
    "dump_schemas",
    "form_values",
    "load_parser",
    "load_schemas",
    "mask_subpath",
    "preprocess_for",
    "report_path",
    "report_subpath",
    "resolved_schema",
    "schema",
]

REPLAY_FIELD = "from_report"
"""``--from_report``: the dest a replay-capable stage exposes."""
# ``REPLAY_REPORT_NAME`` is what ``--from_report --apply`` writes; read from the
# contract rather than the stage that owns it, so this module imports no stage.

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
    # stage id → {dest: dataset root name}, filled from the Settings dialog's
    # dataset roots so no stage form re-asks for them.
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
    """One form field — an :class:`anime_tools._request.Arg` plus the GUI's
    bindings. Shipped to the browser as a dict (``frontend/src/types.ts``
    mirrors it)."""

    dest: str
    kind: str  # bool | int | float | str | enum | list
    flags: list[str] = field(default_factory=list)  # [] → positional
    default: Any = None
    """The argv spelling of the field's default (a prompt list is its
    comma-separated string), JSON-ready; ``None`` for a required field."""
    choices: list[Any] | None = None
    help: str = ""
    required: bool = False
    path: bool = False
    path_kind: str = "dir"
    """``dir`` | ``file``: which chooser the ``…`` beside a ``path`` field
    opens. Meaningless unless ``path``."""
    group: str = ""
    negate: str | None = None
    """For a bool without a ``store_false`` spelling: the ``--no-…`` flag."""
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
    """The dest of the boolean this field hangs off — a *drawer*. The gate carries
    its own dest here, which is how the form tells the checkbox from what it folds
    away. A shut drawer's fields never reach the argv."""

    @property
    def bound(self) -> bool:
        return bool(self.root or self.setting or self.report or self.mask)


def load_parser(stage: Stage):
    """The stage's CLI parser — generated from the same request the schema is."""
    return stage.request_class().parser()


def _json(value: Any) -> Any:
    if value is MISSING:
        return None
    if isinstance(value, tuple):
        return list(value)
    return value


def field_of(a: Arg) -> Field:
    """An :class:`Arg` as the form sees it, before the stage's bindings."""
    name = a.name.replace("_", "-")
    return Field(
        dest=a.name,
        kind=a.kind,
        flags=list(a.flags),
        default=_json(a.default),
        choices=list(a.choices) if a.choices else None,
        help=a.help,
        required=a.required,
        path=any(h in name for h in _PATH_HINTS),
        path_kind="dir" if any(h in name for h in _DIR_HINTS) else "file",
        group=a.group,
        negate=a.negate,
        label=a.name if a.off else (a.flags[0] if a.flags else a.name),
        gate=a.gate,
    )


def schema(stage: Stage) -> dict[str, Any]:
    """JSON-ready description of one stage, or its unavailability."""
    base = {
        "id": stage.id,
        "title": stage.title,
        "panel": stage.panel,
        "short": stage.short or stage.title,
        "module": stage.module,
        "request": stage.request,
        "extra": stage.extra,
        "notes": stage.notes,
        "report": bool(stage.report),
        "hidden": stage.hidden,
        # The preflight this stage gets, so the run bar can name it up front.
        "preprocess": preprocess_for(stage.id),
    }
    try:
        cls = stage.request_class()
    except ImportError as e:  # extra not installed
        return {**base, "available": False, "error": str(e), "fields": [], "doc": ""}
    fs = [field_of(a) for a in args_of(cls)]
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
            and (f.overridable or not (f.bound or f.auto))
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
        "doc": inspect.getdoc(cls) or "",
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

    The schemas are built once at startup and know nothing about a settings
    file, so the bound values a *form* has to show are filled in here, per
    request. Returns a copy, leaving the stored schema alone.
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


def _blank(v: Any) -> bool:
    return v is None or v == "" or v == []


def _coerce(f: Field, v: Any) -> Any:
    """A form value as the request's parser would have left it on the namespace:
    the field's kind decides the Python type, and a blank falls back to the
    default (or is refused on a required field)."""
    if f.kind == "list" and not _blank(v):
        items = v if isinstance(v, list) else str(v).split("\n")
        v = [str(x).strip() for x in items if str(x).strip()]
    if _blank(v):
        if f.required:
            raise ValueError(f"{f.flags[0] if f.flags else f.dest} is required")
        return f.default
    if f.kind == "bool":
        return bool(v)
    if f.kind == "int":
        return int(v)
    if f.kind == "float":
        return float(v)
    if f.kind == "list":
        return v
    return str(v)


def build_argv(
    sc: dict[str, Any],
    values: dict[str, Any],
    *,
    apply: bool = False,
    roots: dict[str, str] | None = None,
    settings: dict[str, str] | None = None,
    report_root: str | None = None,
    mask_root: str | None = None,
) -> list[str]:
    """Turn a ``{dest: value}`` form payload into argv for ``python -m <module>``.

    ``sc`` is the stage's :func:`schema`. The payload is coerced field by field
    into the namespace the stage's own parser would have produced, read into
    the request (so its ``__post_init__`` validation runs here, before a child
    is spawned — a ``ValueError`` is the form's error), and spelled back out by
    ``Request.to_argv()``: a value at the request default is omitted, and
    ``apply`` toggles ``--apply`` regardless of what the form sent.

    ``roots``, ``settings``, ``report_root`` and ``mask_root`` fill the bound
    fields (:data:`ROOT_FIELDS`, :data:`SETTING_FIELDS`, :attr:`Field.report`,
    :attr:`Field.mask`), overriding whatever the form sent — each root joined to
    the stage's own tail. Narrowing a run to one image is a ``path_pattern``
    handed in through ``settings``. :data:`AUTO_FIELDS` never reach the argv,
    and neither do the knobs of a shut drawer.
    """
    cls = request_class(sc["request"])
    fs = [Field(**fd) if isinstance(fd, dict) else fd for fd in sc["fields"]]
    # A drawer's own checkbox decides whether the rest of it is even a value: the
    # stage ignores the knobs of a detector it is not running.
    gate_on = {
        f.dest: bool(values.get(f.dest, f.default)) for f in fs if f.gate == f.dest
    }
    ns: dict[str, Any] = {}
    for f in fs:
        if f.dest == "apply":
            ns[f.dest] = apply
            continue
        shut = f.gate and f.gate != f.dest and not gate_on.get(f.gate, True)
        if f.auto or f.dest in AUTO_FIELDS or shut:
            v = f.default
        elif f.bound:
            # Bound fields come from Settings only, so a stale value in a saved
            # form cannot win over the roots or the pattern the user set. The
            # exception is a :data:`PANEL_FIELDS` dest, where a typed value wins
            # and a blank means "whatever Settings says".
            v = bound_value(
                f,
                roots=roots,
                settings=settings,
                report_root=report_root,
                mask_root=mask_root,
            )
            if f.overridable:
                typed = values.get(f.dest)
                v = typed if str(typed or "").strip() else v
        else:
            v = values.get(f.dest, f.default)
        ns[f.dest] = _coerce(f, v)
    return cls.from_namespace(SimpleNamespace(**ns)).to_argv()


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
    """Every stage's schema, keyed by id. Imports every request module (all
    torch-free, pinned by ``tests/test_stage_requests.py``)."""
    return {s.id: schema(s) for s in STAGES}


def load_schemas() -> dict[str, dict[str, Any]]:
    """Every stage's schema, built in this process from the request classes.
    Cheap enough (no model library is imported) that there is nothing to cache."""
    return dump_schemas()
