"""The counts the prose states are the counts the code has.

Every one of these was wrong at some point for the same reason: the number
sits mid-sentence in a paragraph that reads fine either way, in a file the
commit that changed the count never opened. ``9413178`` dropped a stage, edited
the root ``CLAUDE.md``, and still left "eleven" in three files.

So the number is checked rather than trusted. Each claim is a phrase anchored
tightly enough that only a real count matches it, swept over every tracked
markdown file — a new copy of the sentence in a new file is checked the day it
lands, which is the whole point. A claim that matches nowhere fails too, so
deleting the last copy does not silently retire the check.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# `issue.md` is the review backlog: it quotes the stale numbers on purpose, as
# the record of what was wrong.
EXEMPT = {"issue.md"}

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _stage_count() -> int:
    from anime_tools.stages.registry import STAGES

    return len(STAGES)


def _settings_pane_count() -> int:
    """``SETTINGS_PANES`` in ``frontend/src/config.ts`` — the browser owns the
    list, and no Python constant mirrors it."""
    src = (ROOT / "frontend" / "src" / "config.ts").read_text(encoding="utf-8")
    m = re.search(r"SETTINGS_PANES\s*=\s*\[(.*?)\]", src, re.DOTALL)
    assert m, "SETTINGS_PANES is not where config.ts used to spell it"
    return len(re.findall(r'"[^"]+"', m.group(1)))


CLAIMS = (
    ("stages", r"\ball\s+([a-z]+)\s+stages\b", _stage_count),
    (
        "Settings dialogs",
        r"\b([a-z]+)\s+Settings dialogs\b|\bSettings (?:is|는|은|は|是)\s*([a-z]+)\s+dialogs\b",
        _settings_pane_count,
    ),
)


def _tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z", "*.md"],
        capture_output=True,
        # Same reason as `test_doc_width`: the CJK guidebook filenames are
        # undecodable in cp949/cp932, so the decoding is pinned to UTF-8.
        encoding="utf-8",
        check=True,
    ).stdout
    paths = [rel for rel in out.split("\0") if rel and "node_modules/" not in rel]
    return [ROOT / rel for rel in paths if rel not in EXEMPT]


DOCS = _tracked_markdown()


@pytest.mark.parametrize("label, pattern, count_fn", CLAIMS, ids=lambda v: str(v)[:24])
def test_documented_count_matches_the_code(label, pattern, count_fn):
    expected = count_fn()
    wrong: list[str] = []
    seen = 0
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(pattern, text):
            word = next(g for g in m.groups() if g)
            if word not in NUMBER_WORDS:
                continue
            seen += 1
            if NUMBER_WORDS[word] != expected:
                rel = path.relative_to(ROOT)
                line = text.count("\n", 0, m.start()) + 1
                wrong.append(f"{rel}:{line} says {word!r}")
    assert seen, f"no file states a {label} count any more; retire the claim"
    assert not wrong, f"{label} is {expected}, but: " + "; ".join(wrong)


EXPOSES = re.compile(r"\bexposes all\s+([a-z]+)\s+names\b")


def _package_docs() -> list[Path]:
    """Every ``anime_tools/<pkg>/CLAUDE.md`` that states its ``__init__``'s
    export count."""
    return [p for p in DOCS if p.name == "CLAUDE.md" and EXPOSES.search(p.read_text())]


@pytest.mark.parametrize(
    "path", _package_docs(), ids=lambda p: str(p.relative_to(ROOT))
)
def test_documented_export_count_matches_that_package(path):
    """A package doc's "exposes all N names" is that package's own ``__all__``.

    Keyed on the file's directory rather than repo-wide, because three packages
    say the sentence about three different numbers.
    """
    import importlib

    module = importlib.import_module(".".join(path.parent.relative_to(ROOT).parts))
    word = EXPOSES.search(path.read_text()).group(1)
    assert word in NUMBER_WORDS, f"{path.name} states no number: {word!r}"
    assert NUMBER_WORDS[word] == len(module.__all__)


def test_the_pinned_version_example_is_the_installed_version():
    """``ANIME_TOOLS_VERSION=vN.N.N`` in the docs is this package's version.

    The README's example read ``v0.3.1`` against a current ``0.6.5`` — the same
    drift as the counts above, and the same fix. A bump that fails here wants
    one edit, and the message says which file.
    """
    from anime_tools import __version__

    expected = f"v{__version__}"
    wrong = []
    seen = 0
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"ANIME_TOOLS_VERSION=(v[0-9][^\s`]*)", text):
            seen += 1
            if m.group(1) != expected:
                line = text.count("\n", 0, m.start()) + 1
                wrong.append(f"{path.relative_to(ROOT)}:{line} pins {m.group(1)}")
    assert seen, "no doc shows how to pin a version any more"
    assert not wrong, f"the installed version is {expected}, but: " + "; ".join(wrong)
