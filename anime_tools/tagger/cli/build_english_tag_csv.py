"""Build an English sibling of ``models/danbooru_tags_classified.csv``.

The shipped tag KB (``danbooru_tags_classified.csv``) comes from
``Localsmile/danbooru_KR_wiki_tag_search`` — Danbooru's English tag taxonomy
with **Korean** wiki descriptions. That keeps the GUI tag-explanation tooltip
gated to the Korean UI (see ``gui/tabs/image_tab.py::_on_tag_clicked`` and
``CONTRIBUTING.md`` §5).

This script regenerates the ``description`` column in English by joining the
existing CSV's tag names against the upstream the Korean repo itself translated
from — the Danbooru wiki, mirrored as ``isek-ai/danbooru-wiki-2024`` on the Hub.
It keeps ``name`` / ``category`` / ``post_count`` byte-for-byte from the base CSV
(so tag classification stays identical) and only swaps the description, writing
``danbooru_tags_classified.en.csv`` next to it.

The mirror is a single 45 MB parquet, read with ``pyarrow`` straight out of the
hub cache — no ``datasets`` round trip, so this stays a plain CLI rather than a
data-pipeline dependency.

Usage::

    python -m anime_tools.tagger.cli.build_english_tag_csv          # default paths
    python -m anime_tools.tagger.cli.build_english_tag_csv --revision 202408-at20240906

The GUI runs it for you: it is the ``danbooru_tags_en`` row of
:mod:`anime_tools.downloads`, i.e. a Download button in Settings › Models.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Callable
from pathlib import Path

from anime_tools._env import models_dir
from anime_tools.captions.correction import TAG_CSV_EN_NAME, TAG_CSV_NAME
from anime_tools.downloads import DANBOORU_WIKI_FILE, DANBOORU_WIKI_REPO

# ── DText → plain-text cleanup ────────────────────────────────────────────────
# Danbooru wiki bodies are DText, not MediaWiki. Strip the markup down to the
# bare prose so it renders cleanly in the Qt tooltip.
_WIKI_LINK_LABELLED = re.compile(r"\[\[[^\]|]+\|([^\]]+)\]\]")  # [[tag|label]] -> label
_WIKI_LINK = re.compile(r"\[\[([^\]]+)\]\]")  # [[tag]] -> tag
_SEARCH_LINK = re.compile(r"\{\{([^}]+)\}\}")  # {{search}} -> search
_NAMED_URL_BRACKET = re.compile(r'"([^"]+)":\[[^\]]+\]')  # "label":[url] -> label
_NAMED_URL = re.compile(r'"([^"]+)":https?://\S+')  # "label":url -> label
_BBCODE = re.compile(
    r"\[/?(?:b|i|u|s|tn|quote|code|spoiler|expand)(?:=[^\]]*)?\]", re.IGNORECASE
)
_HEADING = re.compile(r"(?m)^h[1-6]\.\s*")
_LIST_MARKER = re.compile(r"(?m)^\s*[*]+\s*")
_WS = re.compile(r"\s+")


def clean_dtext(body: str, *, max_len: int = 600) -> str:
    """Reduce a DText wiki body to a concise one-paragraph English blurb."""
    if not body:
        return ""
    # The first paragraph is the definition; trailing paragraphs are usage notes
    # / related-tag dumps that bloat the tooltip. Keep the definition.
    paragraph = body.strip().split("\n\n", 1)[0]
    text = _WIKI_LINK_LABELLED.sub(r"\1", paragraph)
    text = _WIKI_LINK.sub(r"\1", text)
    text = _SEARCH_LINK.sub(r"\1", text)
    text = _NAMED_URL_BRACKET.sub(r"\1", text)
    text = _NAMED_URL.sub(r"\1", text)
    text = _BBCODE.sub("", text)
    text = _HEADING.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _WS.sub(" ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0].rstrip() + "…"
    return text


def _norm_key(tag: str) -> str:
    return tag.strip().replace(" ", "_").lower()


def load_wiki_descriptions(
    revision: str | None, log: Callable[[str], None] = print
) -> dict[str, str]:
    """tag-key -> cleaned English description from the Danbooru wiki mirror."""
    import pyarrow.parquet as pq

    from anime_tools._hf import hf_download

    # No log line for the fetch: this is a cache hit whenever the catalog row
    # ran it, and hf_hub_download draws its own bar when it is not.
    path = hf_download(
        what="Danbooru wiki mirror",
        hint="python -m anime_tools.downloads danbooru_tags_en",
        repo_id=DANBOORU_WIKI_REPO,
        repo_type="dataset",
        filename=DANBOORU_WIKI_FILE,
        revision=revision,
    )
    # `tag` is the canonical tag; `title` mirrors it. Index both for the join.
    # Only the four columns we read come off disk -- the mirror also carries
    # `other_names` and per-row metadata this has no use for.
    table = pq.read_table(path, columns=["tag", "title", "body", "is_deleted"])
    out: dict[str, str] = {}
    for tag, title, body, deleted in zip(
        table.column("tag").to_pylist(),
        table.column("title").to_pylist(),
        table.column("body").to_pylist(),
        table.column("is_deleted").to_pylist(),
        strict=True,
    ):
        if deleted:
            continue
        text = clean_dtext(body or "")
        if not text:
            continue
        for name in (tag, title):
            if name:
                out.setdefault(_norm_key(name), text)
    log(f"  {len(out):,} tags carry an English description")
    return out


def build(
    src: Path,
    dst: Path,
    revision: str | None = None,
    log: Callable[[str], None] = print,
) -> None:
    if not src.exists():
        raise FileNotFoundError(
            f"source CSV not found: {src}. Run "
            "`python -m anime_tools.downloads danbooru_tags` first."
        )
    descriptions = load_wiki_descriptions(revision, log)

    total = matched = 0
    with (
        src.open("r", encoding="utf-8-sig", newline="") as fin,
        dst.open("w", encoding="utf-8", newline="") as fout,
    ):
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(
            fout, fieldnames=["name", "category", "post_count", "description"]
        )
        writer.writeheader()
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            total += 1
            desc = descriptions.get(_norm_key(name), "")
            if desc:
                matched += 1
            writer.writerow(
                {
                    "name": name,
                    "category": row.get("category", ""),
                    "post_count": row.get("post_count", ""),
                    "description": desc,
                }
            )

    pct = (100.0 * matched / total) if total else 0.0
    log(f"    ok  {dst}  ({dst.stat().st_size / 1e6:,.0f} MB)")
    log(f"  {matched:,}/{total:,} rows have an English description ({pct:.1f}%)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Defaults resolve at call time, not import time: the models dir moves
    # with ANIME_TOOLS_MODELS / the curation home, and it is where the
    # ``danbooru_tags`` download row puts the base CSV.
    ap.add_argument("--src", type=Path, default=models_dir() / TAG_CSV_NAME)
    ap.add_argument("--dst", type=Path, default=models_dir() / TAG_CSV_EN_NAME)
    ap.add_argument(
        "--revision",
        default=None,
        help="git tag/revision of isek-ai/danbooru-wiki-2024 (default: latest)",
    )
    args = ap.parse_args(argv)
    try:
        build(args.src, args.dst, args.revision)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
