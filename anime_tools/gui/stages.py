"""The web GUI over the stage registry: how a stage's request dataclass becomes
a JSON form schema, and how a form payload becomes the request's argv.

The registry itself is :mod:`anime_tools.stages.registry` (re-exported here);
this module adds the GUI's *bindings* — which flags are dataset roots, which
are ⚙ Settings values, which fold under Advanced — and stays torch-free: a
request class is imported to be described (:func:`schema`) or built
(:func:`build_argv`), and a stage whose deps are missing is listed as
*unavailable* rather than breaking the server.

The bindings are :mod:`~anime_tools.gui._stage_bindings` and the form field
:mod:`~anime_tools.gui._stage_field`, both re-exported here; this module builds
the schema and the argv from them.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from anime_tools._request import args_of

# ``REPLAY_REPORT_NAME`` is what ``--from_report --apply`` writes; read from the
# contract rather than the stage that owns it, so this module imports no stage.
from anime_tools.contract import GATE_ATTR, REPLAY_REPORT_NAME
from anime_tools.gui._stage_bindings import (
    AUTO_FIELDS,
    BASIC_FIELDS,
    EXPORT_STAGE,
    FOLLOW_UPS,
    MASK_FIELDS,
    MASK_SETTING,
    NO_PREFLIGHT,
    PANEL_FIELDS,
    PREPROCESS_SETTINGS_KEY,
    PREPROCESS_STAGE,
    REPLAY_FIELD,
    REPORT_INPUTS,
    REPORT_SETTING,
    ROOT_FIELDS,
    SCOPE_FIELD,
    SETTING_FIELDS,
    SETTINGS_KEY,
    UNBOUND,
    Bindings,
    mask_subpath,
    preprocess_for,
    report_subpath,
)
from anime_tools.gui._stage_field import Field, fields_of
from anime_tools.stages.registry import BY_ID, PANELS, STAGES, Stage, request_class

__all__ = [
    "AUTO_FIELDS",
    "BASIC_FIELDS",
    "BY_ID",
    "EXPORT_STAGE",
    "FOLLOW_UPS",
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
    "UNBOUND",
    "Bindings",
    "Field",
    "Stage",
    "build_argv",
    "fields_of",
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


def load_parser(stage: Stage):
    """The stage's CLI parser — generated from the same request the schema is."""
    return stage.request_class().parser()


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
    fs = [Field(a) for a in args_of(cls)]
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
        # Only a field the form actually shows can be folded away.
        f.advanced = bool(
            basic is not None
            and f.dest not in basic
            and f.shown
            and not f.required
            and f.gate != f.dest
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
        "fields": [f.to_json() for f in fs],
    }


def resolved_schema(sc: dict[str, Any], bindings: Bindings = UNBOUND) -> dict[str, Any]:
    """``sc`` with every :attr:`Field.overridable` default replaced by what
    Settings currently says (:meth:`Bindings.value`).

    The schemas are built once at startup and know nothing about a settings
    file, so the bound values a *form* has to show are filled in here, per
    request. Returns a copy, leaving the stored schema alone.
    """
    fields = sc.get("fields") or []
    if not any(f.get("overridable") for f in fields):
        return sc
    out = []
    for f, fd in zip(fields_of(sc), fields, strict=True):
        v = bindings.value(f) if f.overridable else None
        out.append({**fd, "default": v} if v is not None else fd)
    return {**sc, "fields": out}


def _blank(v: Any) -> bool:
    return v is None or v == "" or v == []


def _coerce(f: Field, v: Any) -> Any:
    """A form value as the request's parser would have left it on the namespace.

    Dispatched on the attributes ``_request.build_parser`` hands argparse — a
    flag taking several values by its ``nargs`` (or by being positional), the
    scalar conversion by its ``type`` — rather than on a second reading of the
    field's ``kind``, so a new kind needs no branch here. A blank falls back to
    the default, or is refused on a required field. A ``masks`` list travels as
    its ``role:kind:value`` strings, which its ``nargs`` already makes a list.
    """
    a = f.arg
    listy = a.nargs is not None or a.positional
    if listy and not _blank(v):
        items = v if isinstance(v, list) else str(v).split("\n")
        v = [str(x).strip() for x in items if str(x).strip()]
    if _blank(v):
        if a.required:
            raise ValueError(f"{a.flags[0] if a.flags else a.name} is required")
        return f.default
    if a.kind == "bool":
        return bool(v)
    if listy:
        return v
    return a.type(v) if a.type is not None else str(v)


def build_argv(
    sc: dict[str, Any],
    values: dict[str, Any],
    bindings: Bindings = UNBOUND,
    *,
    apply: bool = False,
    roots: Mapping[str, str] | None = None,
    settings: Mapping[str, str] | None = None,
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

    ``bindings`` fills the bound fields (:data:`ROOT_FIELDS`,
    :data:`SETTING_FIELDS`, :attr:`Field.report`, :attr:`Field.mask`), overriding
    whatever the form sent — each root joined to the stage's own tail. Narrowing a
    run to one image is a ``path_pattern`` handed in through its ``settings``. :data:`AUTO_FIELDS` never reach the argv,
    and neither do the knobs of a shut drawer.

    ``roots`` / ``settings`` / ``report_root`` / ``mask_root`` are the four
    keywords the trainer's preprocess form and task runner still pass
    (``anima_lora`` ``stage_form.py``, ``scripts/tasks/_common.py``); any of them
    builds the ``Bindings`` in place of ``bindings``.
    """
    if any(v is not None for v in (roots, settings, report_root, mask_root)):
        bindings = Bindings(
            roots=roots or {},
            settings=settings or {},
            report_root=report_root,
            mask_root=mask_root,
        )
    cls = request_class(sc["request"])
    fs = fields_of(sc)
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
            v = bindings.value(f)
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


def load_schemas() -> dict[str, dict[str, Any]]:
    """Every stage's schema, keyed by id, built in this process from the request
    classes. Imports every request module (all torch-free, pinned by
    ``tests/test_stage_requests.py``) and nothing heavier, so there is nothing
    to cache."""
    return {s.id: schema(s) for s in STAGES}
