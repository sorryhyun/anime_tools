"""A throwaway curation home for the examples that walk a dataset.

Every stage resolves its bare-relative paths (``image_dataset``, ``workspace/…``)
against the curation home — ``ANIME_TOOLS_HOME``, else ``ANIMA_HOME``, else the
current directory — so an example can point the whole package at a temp
directory by setting one variable before it builds a request.

Nothing here is part of the package: ``make_sandbox`` draws two synthetic images
with hand-written captions so the resize → correct → export chain has something
to chew on without a real dataset. Pass ``--home <dir>`` to any example to run
it on your own tree instead.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

CAPTIONS = {
    # Flat bag only — what a hand-written master usually looks like.
    "char_a/001": "safe, 1girl, solo, akita neru, long hair, yellow eyes, "
    "white shirt, looking at viewer, @artist_a",
    # Two subjects bound to positions — the clause grammar.
    "char_a/002": "safe, 2girls, akita neru, kasane teto, white socks. "
    "On the left, akita neru, yellow eyes. On the right, kasane teto, drill hair",
    # No artist handle; the correct stage can insert @no-artist here.
    "misc/003": "1boy, solo, black hair, sitting, simple background",
}


def make_sandbox(root: Path | None = None) -> Path:
    """Build ``<root>/image_dataset`` with three captioned images and make
    ``root`` the curation home. Returns ``root``."""
    from PIL import Image, ImageDraw

    root = Path(root) if root else Path(tempfile.mkdtemp(prefix="anime_tools_example_"))
    src = root / "image_dataset"
    for i, (rel, caption) in enumerate(CAPTIONS.items()):
        png = src / f"{rel}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        # Over the resize stage's 0.5 MP floor, and a different aspect per image
        # so the bucket report has more than one row.
        w, h = ((1024, 1024), (1280, 896), (896, 1280))[i % 3]
        img = Image.new("RGB", (w, h), (30 + 60 * i, 40, 80))
        ImageDraw.Draw(img).ellipse(
            (w // 4, h // 4, 3 * w // 4, 3 * h // 4), fill="white"
        )
        img.save(png)
        png.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
    os.environ["ANIME_TOOLS_HOME"] = str(root)
    return root


def home_from_args(home: str | None) -> Path:
    """``--home`` if given (made the curation home), else a fresh sandbox."""
    if home:
        os.environ["ANIME_TOOLS_HOME"] = str(Path(home).expanduser().resolve())
        return Path(os.environ["ANIME_TOOLS_HOME"])
    return make_sandbox()
