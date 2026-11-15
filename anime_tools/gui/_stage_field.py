""":class:`Field`: one form field — the request's :class:`~anime_tools._request.Arg`
carried, plus what the GUI's bindings decide about it — and :func:`fields_of`,
a schema's wire dicts back as fields. Re-exported by :mod:`anime_tools.gui.stages`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass
from pathlib import PurePosixPath
from typing import Any

from anime_tools._request import Arg, args_of
from anime_tools.gui._stage_bindings import REPLAY_FIELD
from anime_tools.stages.registry import request_class

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


@dataclass
class Field:
    """One form field — an :class:`anime_tools._request.Arg` plus the GUI's
    bindings. Shipped to the browser as :meth:`to_json` (``frontend/src/types.ts``
    mirrors that dict).

    The :class:`Arg` is *carried*, not copied: everything the flag itself says —
    its dest, kind, flags, default, help, choices, whether it is required, its
    ``--no-`` spelling, its label — is a property reading the one the parser was
    generated from, so the form and the CLI cannot disagree about a field and
    :func:`_coerce` can dispatch on the very attributes ``build_parser`` hands
    argparse. Only what the *GUI* decides is stored here.
    """

    arg: Arg
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
    hidden like :attr:`report`. A list for the merge, which names its input
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

    # --- the Arg, as the browser reads it ---------------------------------

    @property
    def dest(self) -> str:
        return self.arg.name

    @property
    def kind(self) -> str:
        """``bool`` | ``int`` | ``float`` | ``str`` | ``enum`` | ``list`` | ``masks``."""
        return self.arg.kind

    @property
    def flags(self) -> list[str]:
        """``[]`` → positional."""
        return list(self.arg.flags)

    @property
    def default(self) -> Any:
        """The argv spelling of the field's default (a prompt list is its
        comma-separated string), JSON-ready; ``None`` for a required field."""
        return _json(self.arg.default)

    @property
    def choices(self) -> list[Any] | None:
        return list(self.arg.choices) if self.arg.choices else None

    @property
    def help(self) -> str:
        return self.arg.help

    @property
    def required(self) -> bool:
        return self.arg.required

    @property
    def negate(self) -> str | None:
        """For a bool without a ``store_false`` spelling: the ``--no-…`` flag."""
        return self.arg.negate

    @property
    def label(self) -> str:
        """What the form shows: the flag, or the dest for a ``store_false`` flag so
        a ticked box always means *on*."""
        a = self.arg
        return a.name if a.off else (a.flags[0] if a.flags else a.name)

    @property
    def gate(self) -> str | None:
        """The dest of the boolean this field hangs off — a *drawer*. The gate
        carries its own dest here, which is how the form tells the checkbox from
        what it folds away. A shut drawer's fields never reach the argv."""
        return self.arg.gate

    @property
    def group(self) -> str:
        return self.arg.group

    # --- what the bindings decide ------------------------------------------

    @property
    def bound(self) -> bool:
        return bool(self.root or self.setting or self.report or self.mask)

    @property
    def shown(self) -> bool:
        """Does the form draw this field?

        Not a bound one (the server fills it) unless it is :attr:`overridable`,
        not an :attr:`auto` one, and not the run bar's own ``--apply`` /
        ``--from_report`` — a stale path in a form field would turn the next Run
        into a replay. Decided here so the browser only filters on it.
        """
        return (
            (self.overridable or not (self.bound or self.auto))
            and self.dest != "apply"
            and self.dest != REPLAY_FIELD
        )

    @property
    def path(self) -> bool:
        """Does the ``…`` chooser belong beside this field?

        A bound field names a path by construction, so it says so rather than
        waiting for its flag's name to hint at it. That only matters for a field
        on a form (:data:`PANEL_FIELDS`), where it puts the right chooser beside
        Export's destinations.
        """
        if self.root or self.report or self.mask:
            return True
        return any(h in self._hint for h in _PATH_HINTS)

    @property
    def path_kind(self) -> str:
        """``dir`` | ``file``: which chooser the ``…`` beside a ``path`` field
        opens. Meaningless unless :attr:`path`."""
        if self.root or self.report or self.mask:
            if not self.root and PurePosixPath(str(self.default or "")).suffix:
                return "file"
            return "dir"
        return "dir" if any(h in self._hint for h in _DIR_HINTS) else "file"

    @property
    def _hint(self) -> str:
        """The dest as the ``…``-chooser hints are spelled (hyphenated)."""
        return self.arg.name.replace("_", "-")

    def to_json(self) -> dict[str, Any]:
        """The flat dict the browser gets: the Arg's half and the bindings' half."""
        return {
            "dest": self.dest,
            "kind": self.kind,
            "flags": self.flags,
            "default": self.default,
            "choices": self.choices,
            "help": self.help,
            "required": self.required,
            "path": self.path,
            "path_kind": self.path_kind,
            "group": self.group,
            "negate": self.negate,
            "label": self.label,
            "gate": self.gate,
            "root": self.root,
            "setting": self.setting,
            "report": self.report,
            "mask": self.mask,
            "auto": self.auto,
            "overridable": self.overridable,
            "advanced": self.advanced,
            "shown": self.shown,
        }

    @classmethod
    def from_json(cls, arg: Arg, fd: Mapping[str, Any]) -> Field:
        """The inverse of :meth:`to_json`, given the field's own :class:`Arg`."""
        return cls(
            arg=arg,
            root=fd.get("root"),
            setting=fd.get("setting"),
            report=fd.get("report"),
            mask=fd.get("mask"),
            auto=bool(fd.get("auto")),
            overridable=bool(fd.get("overridable")),
            advanced=bool(fd.get("advanced")),
        )


def fields_of(sc: dict[str, Any]) -> list[Field]:
    """One schema's fields back as :class:`Field` objects.

    The wire dict carries what the browser reads, not the parser's view of the
    flag, so each field's :class:`Arg` is re-attached from the request class the
    schema names — the same object :func:`schema` described it with. Which is why
    a *resolved* schema (:func:`resolved_schema`, whose defaults are the Settings
    values) and the stored one build the same argv.
    """
    args = {a.name: a for a in args_of(request_class(sc["request"]))}
    return [
        fd if isinstance(fd, Field) else Field.from_json(args[fd["dest"]], fd)
        for fd in sc["fields"]
    ]


def _json(value: Any) -> Any:
    if value is MISSING:
        return None
    if isinstance(value, tuple):
        return list(value)
    return value
