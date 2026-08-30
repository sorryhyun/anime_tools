"""Stage registry for the web GUI: which CLIs are exposed, and how their
argparse parsers become a JSON form schema + back into an argv.

Qt/torch/FastAPI-free on purpose so it is unit-testable and cheap to import.
Every stage's ``build_parser()`` is imported lazily: a stage whose deps are not
installed (``cv2`` for masking, say) is listed as *unavailable* rather than
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
from pathlib import Path
from typing import Any

REPLAY_FIELD = "from_report"
"""``--from_report``: the argparse dest a replay-capable stage exposes."""
REPLAY_REPORT_NAME = "apply_report.json"
"""What a replay writes, mirroring ``anime_tools.stages.replay``. Duplicated
rather than imported: this module stays torch-free and cheap to import."""

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


ROOT_FIELDS: dict[str, dict[str, str]] = {
    # stage id → {argparse dest: dataset root name}. These fields are filled
    # from the Settings dialog's dataset roots (the same three trees the
    # sidebar joins), so no stage form re-asks for them.
    "autotag": {"src": "src", "dst": "dst"},
    "position": {"src": "src", "dst": "dst"},
    "correct": {"src": "src", "dst": "dst"},
    "audit": {"src": "src", "dst": "dst"},
    "audit_apply": {"source": "src"},
    "groups": {"source_dir": "src"},
    "masks_sam": {"image_dir": "src"},
    "masks_mit": {"image_dir": "src"},
    # Only the *merged* output is the masks root; each generator's --mask-dir
    # is an intermediate that has to differ from it, so it stays on the form.
    "masks_merge": {"output_dir": "masks"},
}


@dataclass(frozen=True)
class Stage:
    id: str
    title: str
    module: str
    group: str
    extra: str
    """Historical feature area (``tagger`` / ``stages`` / ...); informational only now that everything is a plain dependency."""
    report: tuple[str, str | None] | None = None
    """``(dest, filename)``: the form field naming the report dir (or file when
    ``filename`` is None) so the GUI can fetch the result after a run."""
    notes: str = ""


STAGES: tuple[Stage, ...] = (
    Stage(
        "autotag",
        "Autotag captions",
        "anime_tools.stages.cli.autotag_captions",
        "captions",
        "tagger",
        report=("report_dir", "report.json"),
        notes="Only `missing` mode is non-destructive.",
    ),
    Stage(
        "position",
        "Position captions",
        "anime_tools.stages.cli.position_captions",
        "captions",
        "stages",
        report=("report_dir", "report.json"),
    ),
    Stage(
        "correct",
        "Correct + mirror captions",
        "anime_tools.stages.cli.correct_captions",
        "captions",
        "tokenizers",
        notes="Writes the derived captions under the resized tree; the master is never edited.",
    ),
    Stage(
        "audit",
        "Multiview audit",
        "anime_tools.stages.cli.audit_multiview",
        "captions",
        "stages",
        report=("report_dir", "report.json"),
    ),
    Stage(
        "audit_apply",
        "Apply curated audit list",
        "anime_tools.stages.cli.audit_apply_curated",
        "captions",
        "stages",
    ),
    Stage(
        "groups",
        "Build groups",
        "anime_tools.grouping.cli.build_groups",
        "grouping",
        "grouping",
        report=("out", None),
    ),
    Stage(
        "masks_sam",
        "SAM3 subject masks",
        "anime_tools.masking.cli.generate_masks",
        "masking",
        "masking",
    ),
    Stage(
        "masks_mit",
        "MIT text masks",
        "anime_tools.masking.cli.generate_masks_mit",
        "masking",
        "masking",
    ),
    Stage(
        "masks_merge",
        "Merge masks",
        "anime_tools.masking.cli.merge_masks",
        "masking",
        "masking",
    ),
)

BY_ID: dict[str, Stage] = {s.id: s for s in STAGES}


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
    group: str = ""
    negate: str | None = None
    """For BooleanOptionalAction: the ``--no-…`` flag."""
    label: str = ""
    """What the form shows: the flag, or the dest for a ``store_false`` flag so
    a ticked box always means *on*."""
    root: str | None = None
    """Bound to a dataset root (``src``/``dst``/``masks``): the GUI hides the
    field and :func:`build_argv` fills it from the Settings roots."""


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
        "group": stage.group,
        "module": stage.module,
        "extra": stage.extra,
        "notes": stage.notes,
        "report": bool(stage.report),
    }
    try:
        parser = load_parser(stage)
    except ImportError as e:  # extra not installed
        return {**base, "available": False, "error": str(e), "fields": [], "doc": ""}
    fs = fields_of(parser)
    bound = ROOT_FIELDS.get(stage.id, {})
    for f in fs:
        f.root = bound.get(f.dest)
    return {
        **base,
        "available": True,
        "doc": parser.description or "",
        "apply": any(f.dest == "apply" for f in fs),
        # This stage can write a previous dry run's proposals instead of
        # recomputing them (``--from_report``); the GUI's Apply offers it.
        "replay": any(f.dest == REPLAY_FIELD for f in fs),
        "fields": [f.__dict__ for f in fs],
    }


def build_argv(
    fields: list[dict[str, Any]],
    values: dict[str, Any],
    *,
    apply: bool = False,
    roots: dict[str, str] | None = None,
) -> list[str]:
    """Turn a ``{dest: value}`` form payload into argv for ``python -m <module>``.

    ``fields`` is the ``schema()["fields"]`` list (so the server never has to
    import the stage module). A value equal to the parser default (or empty) is
    omitted, so the CLI's own defaults stay in charge; ``apply`` toggles the
    ``--apply`` flag regardless of what the form sent, so the Dry run / Apply
    buttons are the only route.

    ``roots`` (``{"src": …, "dst": …, "masks": …}``) fills every field bound by
    :data:`ROOT_FIELDS`, overriding whatever the form sent: the dataset roots
    are set once in Settings and no stage gets to disagree with them.
    """
    argv: list[str] = []
    positional: list[str] = []
    for fd in fields:
        f = Field(**fd) if isinstance(fd, dict) else fd
        if f.dest == "apply":
            if apply:
                argv.append(f.flags[0])
            continue
        bound = (roots or {}).get(f.root or "")
        v = bound if bound else values.get(f.dest, f.default)
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


def report_path(
    stage: Stage, fields: list[dict[str, Any]], values: dict[str, Any]
) -> str | None:
    """Where this stage's report lands for the given form values (unresolved).

    A replay (``--from_report``) writes :data:`REPLAY_REPORT_NAME` instead:
    ``--from_report`` and ``--report_dir`` normally name the same directory, so
    the stage refuses to clobber the dry run it is replaying.
    """
    if not stage.report:
        return None
    dest, filename = stage.report
    default = next((f["default"] for f in fields if f["dest"] == dest), None)
    base = values.get(dest) or default
    if not base:
        return None
    if filename and values.get(REPLAY_FIELD):
        filename = REPLAY_REPORT_NAME
    return f"{base}/{filename}" if filename else str(base)


def dump_schemas() -> dict[str, dict[str, Any]]:
    """Every stage's schema, keyed by id. Imports the CLI modules (some pull
    torch) — call it in a child process from anything long-lived."""
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
#
# The child interpreter costs seconds (one stage CLI imports torch at module
# level) and it is on the GUI's startup path, so its output is memoised on disk.


def cache_dir() -> Path:
    """Where the GUI keeps derived, throw-away state.

    Deliberately *not* under :func:`~anime_tools._env.curation_home`: a dataset
    tree is exactly what ``docs/contract.md`` says it is, and a cache that
    survives switching homes is the point. ``$ANIME_TOOLS_CACHE`` overrides;
    otherwise ``$XDG_CACHE_HOME`` (or ``~/.cache``) ``/anime_tools/gui`` — the
    same shape as grouping's ``$NEAR_TWIN_CACHE``.
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
    """What has to change for a cached dump to be wrong.

    The installed version and the interpreter, plus ``(path, mtime_ns, size)``
    for every ``.py`` under the installed package. Whole-package, not just the
    nine CLI shells: a parser's defaults routinely come from the stage module
    behind its CLI, and ``ROOT_FIELDS``/``STAGES``/:func:`fields_of` live in this
    file. Each stage's module file is resolved with
    :func:`importlib.util.find_spec`, which does *not* import it (the parent
    packages are docstring-only shells — nothing pulls torch), so a stage module
    that lives outside the package is keyed on too.
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
