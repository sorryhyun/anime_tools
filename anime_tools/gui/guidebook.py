"""The guidebook the ☰ menu opens, one file per language.

The four books live beside this module rather than under ``docs/`` because the
panel serves them: ``packages.find`` ships only ``anime_tools*``, so a
``uv tool install`` has no ``docs/`` tree to read and the menu row would open an
empty window on every installed copy. They stay plain markdown checked in at
100 columns like every other doc here — the browser renders them
(``frontend/src/components/Markdown.tsx``), and GitHub renders them in place.

Relative links inside a book are written from *this* directory
(``../../../docs/masking.md``), which is what the reader resolves against
`SOURCE_BASE` to send a click in the modal to GitHub.
"""

from __future__ import annotations

from pathlib import Path

from anime_tools.update import REPO_URL

DIR = Path(__file__).resolve().parent / "guidebooks"

#: Locale id (``frontend/src/i18n/``) -> the book's file name. The keys are the
#: panel's four languages; a locale with no book of its own falls back to
#: English rather than 404ing, so adding a UI language never breaks this row.
BOOKS = {
    "en": "guidebook.md",
    "ko": "가이드북.md",
    "ja": "ガイドブック.md",
    "zh": "指南书.md",
}
FALLBACK = "en"

#: What a book's own relative links are relative to, as a URL. The modal resolves
#: `../../../docs/x.md` against it, so a link out of the guidebook opens the file
#: on GitHub instead of dying inside a <dialog>.
SOURCE_BASE = f"{REPO_URL}/blob/main/anime_tools/gui/guidebooks/"


def load(lang: str) -> dict[str, str]:
    """The book for `lang`, or English if there is none.

    Answers what it actually read (`lang`), not what was asked for, so the modal
    can say which language is on screen.
    """
    lang = lang if lang in BOOKS else FALLBACK
    return {
        "lang": lang,
        "markdown": (DIR / BOOKS[lang]).read_text(encoding="utf-8"),
        "base": SOURCE_BASE,
    }
