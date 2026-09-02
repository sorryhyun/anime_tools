"""Build a method-agnostic typed-tag index from caption sidecars.

Walks caption ``.txt`` sidecars under a source dir, classifies each tag (parsed
through the caption grammar — position-clause tags included) into character /
copyright / artist / count via the Anima Tagger vocab (artist additionally by
the ``@`` prefix, exact and not limited by the vocab's frequency cutoff), and
writes one JSON index::

    {
      "meta":  {... vocab path+mtime, src, n_images, generated ...},
      "image_meta": {
        "<key>": {"path": "<rel>", "character": [...], "copyright": [...],
                  "artist": [...], "count": [...]},
      },
      "groups": {"character": {"<tag>": ["<key>", ...]}, "copyright": {...},
                 "artist": {...}}
    }

``<key>`` is the caption's path relative to the source root, extension stripped
and posix-normalized (e.g. ``en/1``), so the same bare filename may repeat
across folders (see :func:`caption_key`). It encodes **no sampling policy**.
"""

import argparse
import os
import re
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

from anime_tools import workspace as WS
from anime_tools._env import resolve_path
from anime_tools._json import write_json
from anime_tools._walk import IMAGE_EXTENSIONS, caption_key, safe_walk

# Never hand-split a caption.
from anime_tools.captions.position_clauses import parse_caption
from anime_tools.captions.taxonomy import is_artist_tag, is_count_tag, is_rating_tag
from anime_tools.captions.vocab_io import load_vocab, names_by_category
from anime_tools.path_filter import filter_paths_by_glob

DEFAULT_VOCAB = "models/captioners/anima-tagger-dbv4/vocab.json"
# A contract artifact (docs/contract.md §2), published to
# ``post_image_dataset/captions/`` by Export.
DEFAULT_OUT = f"{WS.REPORTS}/caption_index.json"
# Artist is detected by the `@` prefix (superset of the vocab artist list);
# character/copyright/count are classified by vocab membership.
VOCAB_AXES = ("character", "copyright", "count")
# Loaded but not emitted: `general` only serves the copyright-recovery veto in
# _vocab_typed_non_copyright.
VOCAB_LOAD_AXES = (*VOCAB_AXES, "general")

# Danbooru disambiguator form ``character_name (copyright_name)``. Vocab
# character names are frozen at its training cutoff, so newer ones miss the
# exact-membership classifier: a ``name (series)`` tag is a character when the
# series is a real franchise (a known vocab copyright, or co-tagged bare in the
# same caption) and not a generic disambiguator (``X (cosplay)``).
_PAREN_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)$")
_GENERIC_PAREN_QUALIFIERS = frozenset(
    {
        "cosplay",
        "costume",
        "alternate costume",
        "meme",
        "food",
        "fruit",
        "maid",
        "animal",
        "object",
    }
)

# Positional recovery of bare-name characters (no `(series)` disambiguator).
# Danbooru order is rigid — ``[rating] [count] [character…] [copyright…] @artist
# [general…]`` — so the character band is the pre-`@artist` run that isn't
# rating/count/copyright. Franchise sub-titles sit in that span but are
# copyright, and are excluded when they share a ≥4-char non-generic word with a
# known copyright in the same caption. These stopwords are too weak to anchor
# that test.
_COPYRIGHT_STOPWORDS = frozenset(
    {
        "club",
        "high",
        "school",
        "idol",
        "story",
        "world",
        "project",
        "series",
        "love",
        "live",
        "girl",
        "girls",
        "boy",
        "boys",
        "the",
        "and",
        "no",
    }
)


def _norm_words(tag: str) -> set[str]:
    """≥4-char alphanumeric words of a tag, minus generic franchise-title
    stopwords."""
    return {
        w
        for w in re.split(r"[^a-z0-9]+", tag)
        if len(w) >= 4 and w not in _COPYRIGHT_STOPWORDS
    }


def _load_vocab_sets(vocab_path: str) -> dict[str, set[str]]:
    """``axis -> {vocab name}`` for the axes the classifier tests membership on.

    Names are folded to the form caption tags arrive in: lowercased, with
    underscores left alone — to danbooru's vocab, two spellings that differ by
    an underscore are two different tags.
    """
    return names_by_category(
        load_vocab(vocab_path),
        VOCAB_LOAD_AXES,
        key=lambda n: n.strip().lower(),
    )


def _vocab_typed_non_copyright(tag: str, vsets: dict[str, set[str]]) -> bool:
    """True if the tagger vocab already types ``tag`` as something else.

    A tag carries exactly one vocab category, so a name the vocab calls
    ``general`` / ``character`` / ``count`` must never be promoted to copyright
    by the parenthetical heuristic below. Without the veto, homonym
    disambiguators (``lily (flower)``, ``piledriver (sex)``) leak their qualifier
    into the copyright vocab: one ``piledriver (sex)`` page made ``sex`` a
    "copyright" on 525 of 2951 colorize captions (measured 2026-08-15).
    """
    return any(tag in vsets.get(axis, ()) for axis in ("general", "character", "count"))


def _iter_captions(src: Path, path_pattern: str | None = None):
    """Yield ``(key, rel_path, text)`` for every ``.txt`` under ``src``.

    ``image_dataset`` is a symlink to a tree of (possibly symlinked) artist
    dirs, so the root is resolved and walked with ``followlinks=True``. The key
    is subdir-disambiguated (:func:`caption_key`), so the same bare stem may
    legally repeat across subfolders."""
    root = Path(os.path.realpath(src))
    for dirpath, _dirnames, filenames in safe_walk(root, followlinks=True):
        for name in filenames:
            if not name.endswith(".txt"):
                continue
            abs_path = Path(dirpath) / name
            if path_pattern and path_pattern != "*":
                keep = filter_paths_by_glob(
                    [str(abs_path)],
                    str(root),
                    path_pattern,
                )[0]
                if not keep:
                    image_sidecars = [
                        str(abs_path.with_suffix(ext)) for ext in IMAGE_EXTENSIONS
                    ]
                    keep = any(
                        filter_paths_by_glob(image_sidecars, str(root), path_pattern)
                    )
                if not keep:
                    continue
            try:
                text = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # Posix-normalize so the JSON artifact is OS-portable.
            rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
            yield caption_key(abs_path, root), rel, text


def _classify(
    text: str,
    vsets: dict[str, set[str]],
    *,
    recover_paren: bool = True,
    recover_positional: bool = True,
) -> dict[str, list[str]]:
    # Flat bag first (order preserved — the positional recovery below reads the
    # danbooru pre-`@artist` band off it), then each clause's tags, which belong
    # in the index but must stay out of the pre-artist band.
    parsed = parse_caption(text)
    tags = [
        t.strip().lower()
        for t in (*parsed.flat_tags, *(t for c in parsed.clauses for t in c.tags))
    ]
    tags = [t for t in tags if t]
    bare = set(tags)
    out: dict[str, list[str]] = {axis: [] for axis in (*VOCAB_AXES, "artist")}
    seen = {axis: set() for axis in out}

    def _add(axis: str, tag: str):
        if tag not in seen[axis]:
            seen[axis].add(tag)
            out[axis].append(tag)

    for tag in tags:
        if is_artist_tag(tag):
            _add("artist", tag)
            continue
        matched = False
        for axis in VOCAB_AXES:
            if tag in vsets[axis]:
                _add(axis, tag)
                matched = True
        if matched or not recover_paren:
            continue
        m = _PAREN_RE.match(tag)
        if m:
            series = m.group(2).strip()
            # A vocab-confirmed copyright is trusted outright; the co-tagged-bare
            # fallback is vetoed when the vocab already types the qualifier.
            if series not in _GENERIC_PAREN_QUALIFIERS and (
                series in vsets["copyright"]
                or (series in bare and not _vocab_typed_non_copyright(series, vsets))
            ):
                _add("character", tag)
                _add("copyright", series)

    if recover_positional:
        artist_at = next((i for i, t in enumerate(tags) if is_artist_tag(t)), None)
        if artist_at is not None:
            copy_words: set[str] = set()
            for cp in out["copyright"]:
                copy_words |= _norm_words(cp)
            for tag in tags[:artist_at]:
                if (
                    tag in seen["character"]
                    or tag in seen["copyright"]
                    or tag in seen["count"]
                ):
                    continue
                if is_rating_tag(tag) or is_count_tag(tag) or _PAREN_RE.match(tag):
                    continue
                if tag in vsets.get("general", ()):
                    continue  # vocab says descriptive — position can't override it
                if _norm_words(tag) & copy_words:
                    continue  # franchise sub-title of a known copyright
                _add("character", tag)

    # When `original` is the SOLE copyright, drop character tags so OC images
    # read as character-less (the contrastive `hard_original` tier). Crossover
    # images keep their characters.
    if set(out["copyright"]) == {"original"}:
        out["character"] = []
        seen["character"] = set()
    return out


def build_index(
    src: str | Path,
    vocab_path: str | Path,
    *,
    recover_paren: bool = True,
    recover_positional: bool = True,
    path_pattern: str | None = None,
) -> dict:
    vsets = _load_vocab_sets(vocab_path)
    image_meta: dict[str, dict] = OrderedDict()
    groups: dict[str, dict[str, list[str]]] = {
        axis: {} for axis in ("character", "copyright", "artist")
    }

    n_seen = 0
    for key, rel, text in sorted(_iter_captions(Path(src), path_pattern)):
        typed = _classify(
            text,
            vsets,
            recover_paren=recover_paren,
            recover_positional=recover_positional,
        )
        if key in image_meta:
            # Keys are subdir-disambiguated, so a repeat is the same file
            # reached twice (symlink loop under followlinks=True).
            continue
        n_seen += 1
        image_meta[key] = {
            "path": rel,
            "character": typed["character"],
            "copyright": typed["copyright"],
            "artist": typed["artist"],
            "count": typed["count"],
        }
        for axis in ("character", "copyright", "artist"):
            for tag in typed[axis]:
                groups[axis].setdefault(tag, []).append(key)

    for axis, by_tag in groups.items():
        groups[axis] = {tag: sorted(stems) for tag, stems in sorted(by_tag.items())}

    vstat = os.stat(vocab_path)
    return {
        "meta": {
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "src": str(src),
            "vocab_path": str(vocab_path),
            "vocab_mtime": datetime.fromtimestamp(vstat.st_mtime, UTC).isoformat(
                timespec="seconds"
            ),
            "n_images": n_seen,
            "axes": ["character", "copyright", "artist", "count"],
            "paren_recover": recover_paren,
            "positional_recover": recover_positional,
            "note": "method-agnostic typed-tag parse; sampling policy lives in method config",
        },
        "image_meta": image_meta,
        "groups": groups,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--src",
        default="image_dataset",
        help="Caption sidecar root (default: image_dataset)",
    )
    ap.add_argument(
        "--vocab",
        default=DEFAULT_VOCAB,
        help=f"Tagger vocab (default: {DEFAULT_VOCAB})",
    )
    ap.add_argument(
        "--out", default=DEFAULT_OUT, help=f"Output JSON (default: {DEFAULT_OUT})"
    )
    ap.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help=(
            "Only index captions whose path relative to --src matches this "
            "fnmatch glob. Use | to separate alternatives. Default: *"
        ),
    )
    ap.add_argument(
        "--no-paren-recover",
        action="store_true",
        help="Disable the danbooru `name (series)` character-recovery heuristic "
        "(exact vocab membership only).",
    )
    ap.add_argument(
        "--no-positional-recover",
        action="store_true",
        help="Disable the positional bare-name character recovery (pre-`@artist` "
        "band minus rating/count/copyright/franchise-sub-titles).",
    )
    args = ap.parse_args()

    # Anchor bare relatives under the curation home.
    src = resolve_path(args.src)
    out = resolve_path(args.out)
    index = build_index(
        src,
        resolve_path(args.vocab),
        recover_paren=not args.no_paren_recover,
        recover_positional=not args.no_positional_recover,
        path_pattern=args.path_pattern,
    )
    write_json(out, index)

    m = index["meta"]
    cov = {
        axis: sum(1 for v in index["image_meta"].values() if v[axis])
        for axis in ("character", "copyright", "artist")
    }
    n = m["n_images"] or 1
    print(f"caption index → {out}")
    print(f"  images: {m['n_images']}")
    for axis in ("character", "copyright", "artist"):
        print(
            f"  {axis:9s}: {cov[axis]:5d} imgs ({100 * cov[axis] / n:4.1f}%), "
            f"{len(index['groups'][axis]):4d} groups"
        )


if __name__ == "__main__":
    main()
