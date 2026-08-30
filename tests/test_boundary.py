"""Dependency direction guard: ``anima_lora`` depends on ``anime_tools``, never
the reverse. No module in the package may import the trainer."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "anime_tools"
FORBIDDEN_ROOTS = (
    "library",
    "networks",
    "train",
    "anima_lora",
    "scripts",
    "gui",
    "bench",
)
_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(?P<from>[\w.]+)\s+import|import\s+(?P<mod>[\w.]+))", re.MULTILINE
)


@pytest.mark.parametrize(
    "path", sorted(PKG.rglob("*.py")), ids=lambda p: str(p.relative_to(PKG.parent))
)
def test_module_does_not_import_trainer(path: Path):
    text = path.read_text(encoding="utf-8")
    bad = []
    for m in _IMPORT_RE.finditer(text):
        mod = m.group("from") or m.group("mod")
        if any(mod == r or mod.startswith(r + ".") for r in FORBIDDEN_ROOTS):
            bad.append(f"{text.count(chr(10), 0, m.start()) + 1}: {m.group(0).strip()}")
    assert not bad, "\n".join(bad)


def test_captions_core_is_torch_free():
    import subprocess
    import sys

    code = (
        "import sys, anime_tools, anime_tools.captions, anime_tools.captions.correction, "
        "anime_tools.captions.variants, anime_tools.captions.index, anime_tools.tagger.dbv4_meta; "
        "assert 'torch' not in sys.modules"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
