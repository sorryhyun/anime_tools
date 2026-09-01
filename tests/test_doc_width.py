"""Markdown prose stays inside the column cap.

The assertion is a fixpoint, not a width check: a file passes when
``scripts/wrap_md.py`` would not rewrite it, so the exemptions live in the
wrapper rather than being restated here.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "wrap_md", ROOT / "scripts" / "wrap_md.py"
)
wrap_md = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wrap_md)


def _tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "*.md"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / rel for rel in out.split("\0") if rel and "node_modules/" not in rel]


@pytest.mark.parametrize(
    "path", _tracked_markdown(), ids=lambda p: str(p.relative_to(ROOT))
)
def test_markdown_is_wrapped(path: Path):
    text = path.read_text(encoding="utf-8")
    wrapped = wrap_md.wrap_text(text)
    if wrapped == text:
        return
    culprits = [
        f"  {n}: ({len(line)} cols) {line[:70]}…"
        for n, (line, after) in enumerate(
            zip(text.split("\n"), wrapped.split("\n")), start=1
        )
        if line != after
    ][:5]
    pytest.fail(
        f"{path.relative_to(ROOT)} has prose over {wrap_md.CAP} columns; "
        f"run `python3 scripts/wrap_md.py {path.relative_to(ROOT)}`\n"
        + "\n".join(culprits)
    )


def test_wrapping_preserves_the_words():
    """A wrap is a soft break: the words survive it character-for-character."""
    import re

    long = (
        "- **`gui/`** (torch-free): one very long bullet that says a great many "
        "things about `stages.py`, and then keeps going — well past the cap — so "
        "that it has to be broken in several places. It ends with a sentence."
    )
    wrapped = wrap_md.wrap_text(long)
    assert "\n" in wrapped
    assert max(len(x) for x in wrapped.split("\n")) <= wrap_md.CAP
    assert re.sub(r"\s+", " ", wrapped) == re.sub(r"\s+", " ", long)
    assert wrap_md.wrap_text(wrapped) == wrapped, "wrapping must be idempotent"


def test_fences_and_tables_are_left_alone():
    text = "\n".join(
        [
            "```bash",
            "make gui  " + "# a deliberately aligned trailing comment " * 3,
            "```",
            "| a | " + "b " * 60 + "|",
            "#### " + "a heading that runs long " * 6,
        ]
    )
    assert wrap_md.wrap_text(text) == text
