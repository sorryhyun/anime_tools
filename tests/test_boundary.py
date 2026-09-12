"""No module in the package may import the trainer."""

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
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert r.returncode == 0, r.stderr


def test_contract_is_torch_free():
    """``anime_tools.contract`` is a stdlib-only leaf: the trainer and the GUI
    server read it without pulling a stage into their process."""
    import subprocess
    import sys

    code = (
        "import sys, anime_tools.contract; "
        "assert 'torch' not in sys.modules; "
        "loaded = sorted(m for m in sys.modules if m.startswith('anime_tools')); "
        "assert loaded == ['anime_tools', 'anime_tools.contract'], loaded"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert r.returncode == 0, r.stderr


def test_gui_server_is_torch_free():
    """The GUI process only spawns stages; model loading stays in the child."""
    import subprocess
    import sys

    pytest.importorskip("fastapi")
    code = (
        "import sys, anime_tools.gui.stages, anime_tools.gui.jobs, anime_tools.gui.server; "
        "anime_tools.gui.server.create_app(); "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert r.returncode == 0, r.stderr


def test_the_masking_library_does_not_import_a_stage():
    """The masking ↔ stages seam runs one way.

    ``masking/sam.py`` used to resolve its ``--prompt_embed`` through
    ``stages/instance_detection.py`` while ``stages/requests.py`` read its help
    strings out of ``masking/_sam3.py``, so a change to either side could be
    made correctly in one place and still be wrong. The prompt vocabulary is
    masking's (``masking/_prompts.py``) and a stage reaches in for it; nothing
    under ``masking/`` reaches back. The ``cli/probe_*`` tools are exempt: they
    are dev probes over a stage's own ``Detection`` geometry, not the library.
    """
    masking = PKG / "masking"
    bad = []
    for path in sorted(masking.rglob("*.py")):
        if path.parent.name == "cli":
            continue
        for m in _IMPORT_RE.finditer(path.read_text(encoding="utf-8")):
            mod = m.group("from") or m.group("mod")
            if mod.startswith("anime_tools.stages"):
                bad.append(f"{path.relative_to(PKG)}: {m.group(0).strip()}")
    assert not bad, "\n".join(bad)
