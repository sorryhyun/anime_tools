"""The base of every stage's request object: a frozen dataclass whose fields are
the stage's knobs, one per argparse ``dest``.

A request is the stage's primary surface; its CLI is a shell that parses argv
into one (:meth:`Request.from_argv`) and the trainer's daemon gets one back as
argv (:meth:`Request.to_argv`). The two are inverses over the stage's own
``build_parser()`` — ``tests/test_masking_requests.py`` and
``tests/test_stage_requests.py`` round-trip each.

Field metadata steers the argv side, all optional:

- ``flag``: the option string. Default ``--<name>``, with ``_`` spelled as the
  class's :attr:`Request.FLAG_SEP` (``-`` for the masking and grouping CLIs,
  ``_`` for the caption stages).
- ``off``: for a bool, the switch that spells ``False`` (a ``store_false``
  action: ``--keep_en`` for ``skip_en``). Without it a bool is
  ``--flag`` / ``--no-flag`` (``BooleanOptionalAction``, or a ``store_true``
  that is only ever emitted as ``--flag``).
- ``positional``: the field is a positional argument (its values, no flag).
- ``read``: ``namespace value -> field value`` (``prompt_list`` turns a
  comma-separated flag into a tuple).
- ``write``: ``field value -> argv value``, the inverse of ``read``. A list or
  tuple result is spelled as one flag followed by every value (``nargs``).

A field whose default is itself a :class:`Request` is a nested block — the
detection flags both SAM3 stages share — read from the same flat namespace and
written inline.

``to_argv()`` omits a field at its default, so an argv is exactly what the
caller changed. A ``None`` off its default has no spelling and is refused.

Stdlib only.
"""

from __future__ import annotations

import argparse
from dataclasses import MISSING, dataclass, fields
from typing import Any, ClassVar, Self

FLAG = "flag"
OFF = "off"
POSITIONAL = "positional"
READ = "read"
WRITE = "write"


def flag_of(f, sep: str = "-") -> str:
    """The option string a dataclass field is parsed from."""
    return f.metadata.get(FLAG) or "--" + f.name.replace("_", sep)


def default_of(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return MISSING


@dataclass(frozen=True, kw_only=True)
class Request:
    FLAG_SEP: ClassVar[str] = "-"
    """How ``_`` in a field name is spelled in its derived flag."""

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
        cls, parser: argparse.ArgumentParser, argv: list[str] | None = None
    ) -> Self:
        """The CLI's whole job: parse, build, and report a bad request the way
        argparse reports a bad flag."""
        try:
            return cls.from_namespace(parser.parse_args(argv))
        except ValueError as e:
            parser.error(str(e))

    def to_argv(self) -> list[str]:
        """This request as the argv its stage's ``build_parser()`` reads back;
        fields at their default are left out."""
        argv: list[str] = []
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
                argv.extend(value if isinstance(value, (list, tuple)) else [str(value)])
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
        return argv
