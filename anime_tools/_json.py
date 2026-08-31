"""Reading and writing the package's JSON files, in one shape.

Roughly a dozen sites wrote a JSON file and half as many read one back, and
they disagreed on three things that matter:

* ``ensure_ascii`` — the default escapes every non-ASCII character, so a
  manifest full of Korean artist directories came back as ``\\uXXXX`` runs.
  Valid JSON, unreadable file. ``False`` here.
* ``encoding`` — a bare ``open(path)`` reads in the platform's locale codepage,
  which is not UTF-8 on Windows, and the Makefile runs there. Always UTF-8.
* whether the parent directory gets created before the write.

``indent=2`` is the canonical shape; ``write_json`` takes an override for the
rare file where size beats readability. No trailing newline — these files are
read by machines and their bytes are compared by skip checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["read_json", "write_json"]


def read_json(path: str | Path) -> Any:
    """Parse a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, *, indent: int | None = 2) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating the parent directory."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8"
    )
    return p
