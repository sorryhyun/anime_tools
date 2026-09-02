"""The base of every stage's request object: a frozen dataclass whose fields are
the stage's knobs, one per argparse ``dest``.

A request is the stage's primary surface; its CLI is a shell that parses argv
into one (:meth:`Request.from_namespace`) and the trainer's daemon gets one back
as argv (:meth:`Request.to_argv`). The two are inverses over the stage's own
``build_parser()`` — ``tests/test_masking_requests.py`` round-trips each.

Field metadata steers the argv side, all optional:

- ``flag``: the option string. Default ``--<name>`` with ``_`` spelled ``-``.
- ``positional``: the field is a positional argument (its values, no flag).
- ``read``: ``namespace value -> field value`` (``prompt_list`` turns a
  comma-separated flag into a tuple).
- ``write``: ``field value -> argv string``, the inverse of ``read``.

``to_argv()`` omits a field at its default, so an argv is exactly what the
caller changed. A bool is a switch: ``True`` emits ``--flag``, ``False`` emits
``--no-flag`` (``BooleanOptionalAction``). A ``None`` off its default has no
spelling and is refused.

Stdlib only.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any, Self

FLAG = "flag"
POSITIONAL = "positional"
READ = "read"
WRITE = "write"


def flag_of(f) -> str:
    """The option string a dataclass field is parsed from."""
    return f.metadata.get(FLAG) or "--" + f.name.replace("_", "-")


def default_of(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return MISSING


@dataclass(frozen=True, kw_only=True)
class Request:
    @classmethod
    def from_namespace(cls, ns: Any) -> Self:
        """The request an ``argparse`` namespace (or anything with the same
        attributes) describes. Validation is the dataclass's ``__post_init__``;
        a CLI turns its ``ValueError`` into ``parser.error``."""
        kw = {}
        for f in fields(cls):
            value = getattr(ns, f.name)
            read = f.metadata.get(READ)
            kw[f.name] = read(value) if read is not None else value
        return cls(**kw)

    def to_argv(self) -> list[str]:
        """This request as the argv its stage's ``build_parser()`` reads back;
        fields at their default are left out."""
        argv: list[str] = []
        for f in fields(self):
            value = getattr(self, f.name)
            if value == default_of(f):
                continue
            write = f.metadata.get(WRITE)
            if write is not None:
                value = write(value)
            flag = flag_of(f)
            if f.metadata.get(POSITIONAL):
                argv.extend(value if isinstance(value, (list, tuple)) else [str(value)])
            elif isinstance(value, bool):
                argv.append(flag if value else "--no-" + flag[2:])
            elif value is None:
                raise ValueError(f"{f.name}=None has no argv spelling")
            else:
                argv.extend([flag, str(value)])
        return argv
