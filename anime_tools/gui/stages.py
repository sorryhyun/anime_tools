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
"""What a replay writes, mirroring ``anime_tools.stages.replay``. Duplicated
rather than imported to keep this module torch-free."""

CACHE_ENV = "ANIME_TOOLS_CACHE"
CACHE_VERSION = 1
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
    "onnx",
)

_DIR_HINTS = ("dir", "src", "dst", "root", "source", "tokenizer", "qwen3")
"""Which of those name a *directory*, so the host's chooser opens in the mode
that can return one; everything else with a path in its name is a file. The
tokenizer flags are the odd ones out, so they are named here. Derived
server-side because the form never re-types argparse."""


SETTINGS_KEY = "stage_defaults"
"""Where :data:`SETTING_FIELDS`' values live in the settings file."""

SETTING_FIELDS: dict[str, str] = {
    # argparse dest → settings key. Stage-independent knobs that mean the same
    # thing everywhere they appear, so they are set once in ⚙ Settings. Like
    # ROOT_FIELDS they are hidden from the form and filled by :func:`build_argv`.
    "path_pattern": "path_pattern",
    "tagger_dir": "tagger_dir",
    "checkpoint": "checkpoint",
    "prompt_embed": "prompt_embed",
}

REPORT_SETTING = "report_root"
"""The directory every stage's report lands under, in :data:`SETTINGS_KEY`.

Not a :data:`SETTING_FIELDS` entry, because two stages sharing a
``--report_dir`` would have one stage's ``--from_report`` replay read the
other's report: each keeps its own sub-path (:func:`report_subpath`) and only
the *root* is the setting. Blank means "beside the ``dst`` root". The one report
a stage *reads* (:data:`REPORT_INPUTS`) is bound the same way.
"""

REPORT_INPUTS: dict[str, str] = {
    # stage id → the dest naming a report this stage *reads*. Bound to
    # :data:`REPORT_SETTING` exactly like the report a stage writes, but kept
    # out of :attr:`Stage.report`, which means "the report this run produces"
    # and is what the run bar fetches back and lets Undo replay.
    "audit_apply": "report",
    "export": "index",
}

SCOPE_FIELD = "path_pattern"
"""The :data:`SETTING_FIELDS` key the GUI narrows to run a stage on one image."""

PREPROCESS_STAGE = "resize"
"""The stage that populates the resized tree, run as a preflight rather than a
dock panel: every stage bound to ``dst`` walks ``workspace/resized/``, so an
image only in the caption master would be invisible to it — zero rows and no
writes, which reads like a bug. Idempotent, so the preflight is cheap."""

PREPROCESS_SETTINGS_KEY = "preprocess"
"""Where the resize form's values live in the settings file. It has no panel, so
its knobs are set once in ⚙ Settings."""

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
    "audit_apply": {"source": "src"},
    # Grouping and the two mask generators read the *resized* tree, like the
    # caption stages above: one decode substrate, one geometry. A mask cut from
    # master pixels is only sound while free-fit's crop stays sub-patch — for a
    # ratio-clamped image it lands off the subject.
    "groups": {"source_dir": "dst"},
    "masks_sam": {"image_dir": "dst"},
    "masks_mit": {"image_dir": "dst"},
    # Only the *merged* output is the masks root; each generator's --mask-dir
    # is an intermediate that has to differ from it, so it stays on the form.
    "masks_merge": {"output_dir": "masks"},
    # The one stage that runs the pipeline backwards: it reads the workspace
    # trees and writes ``src`` (a revised master, where the contract says the
    # master lives) and ``out`` (everything else).
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
    """Historical feature area (``tagger`` / ``stages`` / ...); informational only now that everything is a plain dependency."""
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
        notes="Writes the derived captions under the resized tree; the master is never edited.",
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
        "audit_apply",
        "Apply curated audit list",
        "anime_tools.stages.cli.audit_apply_curated",
        "Curate",
        "stages",
        short="Audit apply",
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
        "MIT text masks",
        "anime_tools.masking.cli.generate_masks_mit",
        "Masks",
        "masking",
        short="Text",
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
    :data:`PREPROCESS_STAGE` in front of it. ``audit_apply`` opens no pixels,
    and :data:`NO_PREFLIGHT` names the rest of the exceptions.
    """
    if stage_id in NO_PREFLIGHT:
        return None
    if "dst" in ROOT_FIELDS.get(stage_id, {}).values():
        return PREPROCESS_STAGE
    return None


BY_ID: dict[str, Stage] = {s.id: s for s in STAGES}


def report_subpath(default: str) -> str:
    """The part of a report default that is *not* a dataset root.

    Every ``--report_dir`` / ``--out`` default is ``<dataset root>/<this stage's
    own path>``; dropping the first component lets one :data:`REPORT_SETTING`
    move every report at once while each stage keeps a directory of its own.
    """
    parts = PurePosixPath(default).parts
    return PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else default


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
    """Bound to a dataset root: hidden from the form, filled by
    :func:`build_argv` from the Settings roots."""
    setting: str | None = None
    """Bound to a :data:`SETTING_FIELDS` key: hidden the same way, filled from
    the Settings dialog's stage defaults."""
    report: str | None = None
    """This stage's own path under the :data:`REPORT_SETTING` root: hidden from
    the form and filled by :func:`build_argv` as ``<root>/<report>``."""
    auto: bool = False
    """In :data:`AUTO_FIELDS`: never shown, never sent, always auto-detected."""


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
    for g in parser._action_groups:
        if g.title not in ("positional arguments", "options", "optional arguments"):
            for a in g._group_actions:
                groups[id(a)] = g.title or ""
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
                required=bool(a.required) or not flags,
                path=any(h in name for h in _PATH_HINTS),
                path_kind="dir" if any(h in name for h in _DIR_HINTS) else "file",
                group=groups.get(id(a), ""),
                negate=negate,
                label=label,
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
    report_dests = {stage.report[0] if stage.report else None} | {
        REPORT_INPUTS.get(stage.id)
    }
    for f in fs:
        f.root = bound.get(f.dest)
        f.setting = SETTING_FIELDS.get(f.dest)
        f.auto = f.dest in AUTO_FIELDS
        if f.dest in report_dests and isinstance(f.default, str):
            f.report = report_subpath(f.default)
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


def build_argv(
    fields: list[dict[str, Any]],
    values: dict[str, Any],
    *,
    apply: bool = False,
    roots: dict[str, str] | None = None,
    settings: dict[str, str] | None = None,
    report_root: str | None = None,
) -> list[str]:
    """Turn a ``{dest: value}`` form payload into argv for ``python -m <module>``.

    ``fields`` is the ``schema()["fields"]`` list (so the server never has to
    import the stage module). A value equal to the parser default (or empty) is
    omitted, so the CLI's own defaults stay in charge; ``apply`` toggles the
    ``--apply`` flag regardless of what the form sent, so the Dry run / Apply
    buttons are the only route.

    ``roots``, ``settings`` and ``report_root`` fill the bound fields
    (:data:`ROOT_FIELDS`, :data:`SETTING_FIELDS`, :attr:`Field.report`),
    overriding whatever the form sent — the report joined to the stage's own
    tail so the stages keep separate directories under the one root. Narrowing
    a run to one image is a ``path_pattern`` handed in through ``settings``.
    :data:`AUTO_FIELDS` never reach the argv at all.
    """
    argv: list[str] = []
    positional: list[str] = []
    for fd in fields:
        f = Field(**fd) if isinstance(fd, dict) else fd
        if f.auto or f.dest in AUTO_FIELDS:
            continue
        if f.dest == "apply":
            if apply:
                argv.append(f.flags[0])
            continue
        if f.root or f.setting or f.report:
            # Bound fields come from Settings and *only* from Settings: a stale
            # value left in a saved form must never win over the roots or the
            # pattern the user set, so `values` is not consulted at all.
            v = (
                (roots or {}).get(f.root or "")
                or (settings or {}).get(f.setting or "")
                or (f"{report_root}/{f.report}" if report_root and f.report else "")
            )
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
        if f.default is not None and v == f.default and not f.required:
            continue
        if f.flags:
            argv += [f.flags[0], str(v)]
        else:
            positional.append(str(v))
    return argv + positional


def form_values(fields: list[dict[str, Any]], values: dict[str, Any]) -> dict[str, Any]:
    """``values`` minus everything the form does not own.

    The GUI persists the last form per stage; bound and auto-detected dests are
    not the form's to remember. :func:`build_argv` already ignores them — this
    keeps them from being written down in the first place.
    """
    drop = {
        f["dest"]
        for f in fields
        if f.get("root")
        or f.get("setting")
        or f.get("report")
        or f.get("auto")
        or f["dest"] in AUTO_FIELDS
    }
    return {k: v for k, v in values.items() if k not in drop}


def report_path(
    stage: Stage,
    fields: list[dict[str, Any]],
    values: dict[str, Any],
    report_root: str | None = None,
) -> str | None:
    """Where this stage's report lands for this run (unresolved).

    The same answer :func:`build_argv` puts on the argv, so the GUI reads back
    exactly the file the child was told to write; with no ``report_root`` this
    falls back to the CLI's own default. A replay (``--from_report``) writes
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
# The child interpreter is on the GUI's startup path. Stage CLIs defer their
# heavy imports, so the dump is ~0.2s and the memo is not load-bearing; it keeps
# startup flat if a stage regains a slow import.


def cache_dir() -> Path:
    """Where the GUI keeps derived, throw-away state.

    Deliberately *not* under the curation home: a dataset tree is exactly what
    ``docs/contract.md`` says it is. ``$ANIME_TOOLS_CACHE`` overrides.
    """
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "anime_tools" / "gui"


def schema_cache_path() -> Path:
    return cache_dir() / "schemas.json"


def schema_cache_key() -> str:
    """What has to change for a cached dump to be wrong: the installed version,
    the interpreter, and ``(path, mtime_ns, size)`` for every ``.py`` under the
    package — whole-package, because a parser's defaults routinely come from the
    stage module behind its CLI. :func:`importlib.util.find_spec` keys on a
    stage module outside the package without importing it.
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
    hand-mangled, stale — is ``None``, i.e. "just dump again"; never an error."""
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
        # gone). The cache is an optimisation; never let it be the reason the
        # GUI has no stage list.
        return dump_schemas_in_child()
    cached = _read_schema_cache(path, key)
    if cached is not None:
        return cached
    schemas = dump_schemas_in_child()
    _write_schema_cache(path, key, schemas)
    return schemas
