"""The base of every stage's request object: a frozen dataclass whose fields are
the stage's knobs, one per argparse ``dest``.

A request is the stage's primary surface. Its CLI is a shell that parses argv
into one (:meth:`Request.from_argv`), the trainer's daemon gets one back as
argv (:meth:`Request.to_argv`), and the web GUI draws its form off the same
field list (:func:`args_of`). The parser itself is *generated* from the
fields (:func:`build_parser` / :meth:`Request.parser`), so a flag's spelling,
default, help and grouping are written once, in the field's metadata —
``tests/test_masking_requests.py`` and ``tests/test_stage_requests.py``
round-trip each request through its parser.

Field metadata, all optional and all written through :func:`arg`:

- ``help``: the flag's help, shown by ``--help`` and as the form label's tooltip.
- ``flag``: the option string. Default ``--<name>``, with ``_`` spelled as the
  class's :attr:`Request.FLAG_SEP` (``-`` for the masking and grouping CLIs,
  ``_`` for the caption stages). Every flag with a separator in it also takes
  the other spelling as an alias (:func:`spellings`).
- ``off``: for a bool, the switch that spells ``False`` (a ``store_false``
  action: ``--keep_en`` for ``skip_en``). Without it a bool is
  ``--flag`` / ``--no-flag`` (``BooleanOptionalAction``).
- ``positional``: the field is a positional argument (its values, no flag).
- ``choices``: an enum; the parser refuses anything else and the form is a select.
- ``group``: the argparse argument group — a titled section of the form. A
  nested block's fields default to its class's :attr:`Request.GROUP`.
- ``gate``: the dest of the bool that switches this field on — a *drawer* in the
  form. The gate names itself, and the drawer's fields take its group.
- ``nargs`` / ``type`` / ``metavar``: for a flag taking several values
  (``--target_res 1024 1536``); a scalar's type comes from its default.
- ``read``: ``namespace value -> field value`` (``prompt_list`` turns a
  comma-separated flag into a tuple).
- ``write``: ``field value -> argv value``, the inverse of ``read``. A list or
  tuple result is spelled as one flag followed by every value (``nargs``).

A field whose default is itself a :class:`Request` is a nested block — the
detection flags both SAM3 stages share — read from the same flat namespace and
written inline.

``to_argv()`` omits a field at its default, so an argv is exactly what the
caller changed. A ``None`` off its default has no spelling and is refused.

Stdlib only (plus :mod:`anime_tools.contract`, which is too).
"""

from __future__ import annotations

import argparse
import inspect
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, ClassVar, Self

from anime_tools.contract import GATE_ATTR

FLAG = "flag"
OFF = "off"
POSITIONAL = "positional"
READ = "read"
WRITE = "write"
HELP = "help"
CHOICES = "choices"
GROUP = "group"
GATE = "gate"
NARGS = "nargs"
TYPE = "type"
METAVAR = "metavar"

KEYS = frozenset(
    {
        FLAG,
        OFF,
        POSITIONAL,
        READ,
        WRITE,
        HELP,
        CHOICES,
        GROUP,
        GATE,
        NARGS,
        TYPE,
        METAVAR,
    }
)


def arg(default: Any = MISSING, **meta: Any) -> Any:
    """A request field with its flag's metadata::

        score_threshold: float = arg(0.5, help="Subject confidence floor", group="detection")

    No ``default`` makes the field (and its flag) required. A ``None`` value in
    ``meta`` is the same as leaving the key out.
    """
    unknown = set(meta) - KEYS
    if unknown:
        raise TypeError(f"unknown field metadata {sorted(unknown)}")
    metadata = {k: v for k, v in meta.items() if v is not None and v != ""}
    if default is MISSING:
        return field(metadata=metadata)
    return field(default=default, metadata=metadata)


def flag_of(f, sep: str = "-") -> str:
    """The option string a dataclass field is parsed from."""
    return f.metadata.get(FLAG) or "--" + f.name.replace("_", sep)


def spellings(flag: str) -> tuple[str, ...]:
    """``flag`` plus its other spelling: ``--foo_bar`` takes ``--foo-bar`` too and
    the reverse, so neither package's convention breaks a saved command line.
    A one-word flag has one spelling."""
    body = flag[2:]
    other = body.replace("_", "-") if "_" in body else body.replace("-", "_")
    return (flag,) if other == body else (flag, "--" + other)


def default_of(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return MISSING


def argv_default(f) -> Any:
    """The field's default as the parser holds it: the ``write`` spelling of
    the field value (``()`` is ``""`` or ``"none"`` for a prompt list), so
    ``parse_args([])`` reads back through ``read`` as the request default."""
    default = default_of(f)
    if default is MISSING or default is None:
        return default
    write = f.metadata.get(WRITE)
    return write(default) if write is not None else default


@dataclass(frozen=True)
class Arg:
    """One field as its flag: what :func:`build_parser` declares and what the
    GUI form shows. ``default`` is the argv spelling (:func:`argv_default`), or
    :data:`MISSING` for a required field."""

    name: str
    flags: tuple[str, ...]
    """Canonical spelling first, then the alias; ``()`` for a positional. A
    ``store_false`` switch lists its ``off`` spellings."""
    kind: str
    """``bool`` | ``int`` | ``float`` | ``str`` | ``enum`` | ``list``."""
    default: Any
    help: str = ""
    choices: tuple[Any, ...] | None = None
    group: str = ""
    gate: str | None = None
    negate: str | None = None
    """``--no-<flag>`` for a bool without ``off``."""
    off: bool = False
    nargs: int | str | None = None
    type: type | None = None
    metavar: str | tuple[str, ...] | None = None

    @property
    def required(self) -> bool:
        return self.default is MISSING

    @property
    def positional(self) -> bool:
        return not self.flags


def _kind(f, default: Any) -> str:
    md = f.metadata
    if md.get(POSITIONAL) or md.get(NARGS) is not None:
        return "list"
    if md.get(CHOICES):
        return "enum"
    t = md.get(TYPE)
    if t is None and default is not MISSING and default is not None:
        t = type(default)
    if t is bool:
        return "bool"
    if t is int:
        return "int"
    if t is float:
        return "float"
    return "str"


def args_of(cls: type[Request]) -> list[Arg]:
    """Every field of ``cls`` as an :class:`Arg`, nested blocks flattened in
    place, in declaration order — the order the parser declares them and the
    form shows them."""
    out: list[Arg] = []
    gate_groups: dict[str, str] = {}
    block_group = cls.GROUP
    for f in fields(cls):
        default = default_of(f)
        if isinstance(default, Request):
            out.extend(args_of(type(default)))
            continue
        md = f.metadata
        gate = md.get(GATE)
        group = md.get(GROUP) or gate_groups.get(gate or "", "") or block_group
        if gate == f.name:
            gate_groups[gate] = group
        kind = _kind(f, default)
        off = md.get(OFF)
        if md.get(POSITIONAL):
            flags: tuple[str, ...] = ()
        elif kind == "bool" and off:
            flags = spellings(off)
        else:
            flags = spellings(flag_of(f, cls.FLAG_SEP))
        negate = "--no-" + flags[0][2:] if kind == "bool" and not off else None
        t = md.get(TYPE)
        if t is None and kind in ("int", "float"):
            t = int if kind == "int" else float
        choices = md.get(CHOICES)
        out.append(
            Arg(
                name=f.name,
                flags=flags,
                kind=kind,
                default=argv_default(f),
                help=md.get(HELP, ""),
                choices=tuple(choices) if choices else None,
                group=group,
                gate=gate,
                negate=negate,
                off=bool(off),
                nargs=md.get(NARGS),
                type=t,
                metavar=md.get(METAVAR),
            )
        )
    return out


def build_parser(cls: type[Request], **kw: Any) -> argparse.ArgumentParser:
    """The ``argparse`` parser ``cls.from_namespace`` reads, declared from
    :func:`args_of`. ``description`` defaults to the class docstring; a gated
    group is stamped with :data:`GATE_ATTR` for anyone still introspecting."""
    kw.setdefault("description", inspect.getdoc(cls))
    p = argparse.ArgumentParser(**kw)
    groups: dict[str, argparse._ArgumentGroup] = {}
    for a in args_of(cls):
        c: argparse._ActionsContainer = p
        if a.group:
            c = groups.get(a.group) or groups.setdefault(
                a.group, p.add_argument_group(a.group)
            )
            if a.gate == a.name:
                setattr(c, GATE_ATTR, a.gate)
        common: dict[str, Any] = {}
        if a.help:
            common["help"] = a.help.replace("%", "%%")
        if a.metavar:
            common["metavar"] = a.metavar
        if a.positional:
            if a.required:
                c.add_argument(a.name, nargs=a.nargs or "+", type=a.type, **common)
            else:
                c.add_argument(
                    a.name,
                    nargs=a.nargs or "*",
                    default=a.default,
                    type=a.type,
                    **common,
                )
        elif a.kind == "bool":
            action = "store_false" if a.off else argparse.BooleanOptionalAction
            c.add_argument(
                *a.flags, dest=a.name, action=action, default=a.default, **common
            )
        else:
            if a.required:
                common["required"] = True
            else:
                common["default"] = a.default
            if a.type is not None:
                common["type"] = a.type
            if a.nargs is not None:
                common["nargs"] = a.nargs
            if a.choices:
                common["choices"] = a.choices
            c.add_argument(*a.flags, dest=a.name, **common)
    return p


@dataclass(frozen=True, kw_only=True)
class Request:
    FLAG_SEP: ClassVar[str] = "-"
    """How ``_`` in a field name is spelled in its derived flag."""
    GROUP: ClassVar[str] = ""
    """The argument group this class's fields sit in unless a field names its
    own — set on a nested block (``DetectionRequest``), blank on a stage."""

    @classmethod
    def parser(cls, **kw: Any) -> argparse.ArgumentParser:
        """This request's CLI parser (:func:`build_parser`)."""
        return build_parser(cls, **kw)

    @classmethod
    def from_namespace(cls, ns: Any) -> Self:
        """The request an ``argparse`` namespace (or anything with the same
        attributes) describes. Validation is the dataclass's ``__post_init__``;
        a CLI turns its ``ValueError`` into ``parser.error``."""
        kw = {}
        for f in fields(cls):
            default = default_of(f)
            if isinstance(default, Request):
                kw[f.name] = type(default).from_namespace(ns)
                continue
            value = getattr(ns, f.name)
            read = f.metadata.get(READ)
            kw[f.name] = read(value) if read is not None else value
        return cls(**kw)

    @classmethod
    def from_argv(
        cls,
        parser: argparse.ArgumentParser | None = None,
        argv: list[str] | None = None,
    ) -> Self:
        """The CLI's whole job: parse, build, and report a bad request the way
        argparse reports a bad flag."""
        if parser is None:
            parser = cls.parser()
        try:
            return cls.from_namespace(parser.parse_args(argv))
        except ValueError as e:
            parser.error(str(e))

    def to_argv(self) -> list[str]:
        """This request as the argv its parser reads back; fields at their
        default are left out, positionals come last."""
        argv: list[str] = []
        positional: list[str] = []
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Request):
                argv.extend(value.to_argv())
                continue
            if value == default_of(f):
                continue
            write = f.metadata.get(WRITE)
            if write is not None:
                value = write(value)
            flag = flag_of(f, self.FLAG_SEP)
            if f.metadata.get(POSITIONAL):
                positional.extend(
                    [str(v) for v in value]
                    if isinstance(value, (list, tuple))
                    else [str(value)]
                )
            elif isinstance(value, bool):
                if value:
                    argv.append(flag)
                else:
                    argv.append(f.metadata.get(OFF) or "--no-" + flag[2:])
            elif value is None:
                raise ValueError(f"{f.name}=None has no argv spelling")
            elif isinstance(value, (list, tuple)):
                argv.extend([flag, *(str(v) for v in value)])
            else:
                argv.extend([flag, str(value)])
        return argv + positional
