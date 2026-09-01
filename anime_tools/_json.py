"""Reading and writing the package's JSON files, in one shape.

* ``ensure_ascii=False``, so non-ASCII directory names stay readable.
* ``encoding="utf-8"`` — a bare ``open(path)`` reads in the platform's locale
  codepage, which is not UTF-8 on Windows.
* the parent directory is created before the write.

``indent=2`` is the canonical shape, overridable. No trailing newline — skip
checks compare these files byte for byte.
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
