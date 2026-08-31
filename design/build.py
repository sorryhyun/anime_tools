#!/usr/bin/env python3
"""Assemble the design canvas from ``boards/`` and hand it to the /design seeder.

Each ``boards/<Name>.html`` is one artboard's body. This wraps it in the Design
Component envelope, inlines ``helmet.html`` with the Pretendard subset as a
base64 ``@font-face`` (the artboard iframe has no network egress, so the font
has to ride inside the file, exactly as ``frontend/build.ts`` does for the GUI),
and writes ``build/<Name>.dc.html``.

Seeding and publishing are the /design skill's job -- it ships the canvas
editor payload and the ``seed-canvas.mjs`` helper, which live in a bundled-skill
directory whose path changes with the Claude Code version. Point this at it with
``--seeder`` or ``$DESIGN_SKILL_DIR``, or run the assemble step alone and seed
by hand. Nothing here reads or writes the published artifact.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOARDS = HERE / "boards"
BUILD = HERE / "build"
FONT = HERE / "fonts" / "pretendard-latin.woff2"
CANVAS = HERE / "canvas.json"
HELMET = HERE / "helmet.html"

TITLE = "anime_tools Design System"
OUT_HTML = "anime-tools-design-system.html"

ENVELOPE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="./support.js"></script>
</head>
<body>
<x-dc>
{helmet}
{body}
</x-dc>
</body>
</html>
"""


def assemble() -> list[Path]:
    """boards/<Name>.html + helmet -> build/<Name>.dc.html. Returns them in
    canvas.json order, which is the order the seeder is given them in."""
    helmet = (
        HELMET.read_text(encoding="utf-8")
        .rstrip("\n")
        .replace("PRETENDARD_B64", base64.b64encode(FONT.read_bytes()).decode("ascii"))
    )
    canvas = json.loads(CANVAS.read_text(encoding="utf-8"))
    order = [a["file"] for a in canvas["artboards"]]

    listed = {p.stem + ".dc.html" for p in BOARDS.glob("*.html")}
    if missing := [f for f in order if f not in listed]:
        sys.exit(f"canvas.json lists artboards with no board source: {missing}")
    if extra := sorted(listed - set(order)):
        sys.exit(f"boards/ has sources canvas.json does not lay out: {extra}")

    BUILD.mkdir(exist_ok=True)
    out = []
    for name in order:
        body = (BOARDS / name.replace(".dc.html", ".html")).read_text(encoding="utf-8")
        dest = BUILD / name
        dest.write_text(
            ENVELOPE.format(helmet=helmet, body=body.rstrip("\n")), encoding="utf-8"
        )
        out.append(dest)
    shutil.copy(CANVAS, BUILD / "canvas.json")
    return out


def find_seeder(explicit: str | None) -> Path | None:
    for cand in (explicit, os.environ.get("DESIGN_SKILL_DIR")):
        if not cand:
            continue
        p = Path(cand)
        p = p if p.name == "seed-canvas.mjs" else p / "seed-canvas.mjs"
        if p.is_file():
            return p
        sys.exit(f"no seed-canvas.mjs at {p}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--seeder",
        metavar="PATH",
        help="the /design skill directory, or its seed-canvas.mjs "
        "(default: $DESIGN_SKILL_DIR; without either, only assemble)",
    )
    args = ap.parse_args()

    boards = assemble()
    total = sum(p.stat().st_size for p in boards) / 1024
    print(f"assembled {len(boards)} artboards into {BUILD}/ ({total:.0f} KB)")

    seeder = find_seeder(args.seeder)
    if seeder is None:
        print(
            "\nno seeder given -- to publish, run /design in Claude Code (it extracts\n"
            "the skill), then re-run with --seeder <that directory>."
        )
        return

    cmd = [
        shutil.which("node") or "node",
        str(seeder),
        "--template",
        str(seeder.parent / "payload.template.html"),
        "--out",
        str(BUILD / OUT_HTML),
        "--title",
        TITLE,
        "--canvas",
        str(BUILD / "canvas.json"),
    ]
    for p in boards:
        cmd += ["--artboard", str(p)]
    subprocess.run(cmd, check=True)
    subprocess.run([cmd[0], str(seeder), "--check", str(BUILD / OUT_HTML)], check=True)
    print(f"\npublish {BUILD / OUT_HTML} to the artifact (see README).")


if __name__ == "__main__":
    main()
