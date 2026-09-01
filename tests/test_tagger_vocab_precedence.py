"""``anime_tools.tagger.cli.vocab`` caption scanning.

* Root-order precedence: ``build_caption_index`` consumes paths in the order
  ``find_caption_files`` returns them, so a root listed first shadows later ones.
* Position clauses never reach the label space; only the flat bag trains.
"""

from __future__ import annotations

from pathlib import Path

from anime_tools.captions.tag_rules import TagRules
from anime_tools.tagger.cli.vocab import build_caption_index, find_caption_files

EMPTY_RULES = TagRules(
    replacements=(),
    remove=frozenset(),
    dedup={},
    category_overrides={},
    coverage_ignore=(),
)


def _write(root: Path, stem: str, caption: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{stem}.txt").write_text(caption, encoding="utf-8")


def test_first_root_wins_regardless_of_path_sort(tmp_path):
    # 'z_curated' sorts after 'a_raw': precedence is root order, not path order.
    curated = tmp_path / "z_curated"
    raw = tmp_path / "a_raw"
    _write(curated, "123", "nsfw, 1girl, curated_tag")
    _write(raw, "123", "questionable, 1girl, raw_tag")
    _write(raw, "456", "safe, 1boy")

    paths = find_caption_files([curated, raw])
    index = build_caption_index(paths, EMPTY_RULES)

    assert set(index) == {"123", "456"}
    cap_path, _img, tags = index["123"]
    assert cap_path.parent == curated
    assert "curated_tag" in tags and "raw_tag" not in tags


def test_position_clauses_dropped_from_tags(tmp_path):
    _write(
        tmp_path / "root",
        "789",
        "nsfw, 2girls, indoors. On the left, akita neru, yellow eyes. "
        "On the right, hatsune miku.",
    )
    index = build_caption_index(find_caption_files([tmp_path / "root"]), EMPTY_RULES)

    _cap, _img, tags = index["789"]
    assert tags == ["nsfw", "2girls", "indoors"]
