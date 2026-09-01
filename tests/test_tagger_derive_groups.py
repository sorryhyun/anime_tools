"""The one groups-derivation orchestration, shared by both of its callers.

``--mode derive_groups --apply`` and ``build_vocab --derive_groups`` used to
carry two copies of the same sequence: the ten-argument ``derive_rows`` call,
``merge_apply``, a back-up-then-write spelled two different ways, and the
``n_general`` coverage tally. They go through ``derive_from_args`` +
``write_merged_groups`` now. What stays with each caller is what genuinely
differs — where ``solo_sets`` comes from (the manifest, or the scan the build
already did) and where the KB comes from (``--tag_cache``, or ``find_tag_csv()``
with a warn-and-skip fallback) — which is what the last test here pins: give the
two the same corpus and they derive the same groups.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from anime_tools._json import read_json
from anime_tools.captions import tag_groups as tg
from anime_tools.captions.correction import load_tag_knowledge_base
from anime_tools.captions.tag_rules import TagRules
from anime_tools.tagger.cli.derive_groups import (
    derive_from_args,
    write_merged_groups,
)
from anime_tools.tagger.cli.vocab import cmd_build_vocab

EYE_PATH = "얼굴/눈 > 눈 색상"
HAIR_PATH = "머리카락 > 머리 길이"

# name,category,post_count,description — the KB's shape; the ``[대분류>소분류]``
# prefix of the description is the taxonomy path groups are bucketed by.
KB_CSV = """name,category,post_count,description
blue_eyes,0,900,"[얼굴/눈>눈 색상] 파란 눈"
red_eyes,0,800,"[얼굴/눈>눈 색상] 붉은 눈"
long_hair,0,700,"[머리카락>머리 길이] 긴 머리"
short_hair,0,600,"[머리카락>머리 길이] 짧은 머리"
1girl,0,500,"[인물 > 인원수] 한 명"
"""

CAPTIONS = {
    "a": "safe, 1girl, blue eyes, long hair",
    "b": "safe, 1girl, red eyes, short hair",
    "c": "safe, 1girl, blue eyes, short hair",
    "d": "safe, 1girl, red eyes, long hair",
}

NO_RULES = TagRules((), frozenset(), {}, {}, ())


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A caption root (with sibling images) and a KB csv → ``(roots, kb_csv)``."""
    root = tmp_path / "captions"
    root.mkdir(parents=True, exist_ok=True)
    for stem, text in CAPTIONS.items():
        (root / f"{stem}.txt").write_text(text, encoding="utf-8")
        (root / f"{stem}.png").touch()  # existence is all find_image_for_caption asks
    kb_csv = tmp_path / "danbooru_tags_classified.csv"
    kb_csv.write_text(KB_CSV, encoding="utf-8")
    return root, kb_csv


def _knobs(**over) -> dict:
    return {
        "min_group_size": 2,
        "min_member_freq": 1,
        "min_group_support": 1,
        "softmax_cooc_max": 0.05,
        "borderline_cooc_max": 0.20,
        **over,
    }


def _vocab(names: list[str]) -> dict:
    return {
        "tags": [
            {"name": n, "index": i, "category": "general", "freq": 100}
            for i, n in enumerate(names)
        ]
    }


def _solo_sets_from_captions() -> list[set[str]]:
    return [set(c.split(", ")) for c in CAPTIONS.values()]


# --------------------------------------------------------------------------- #
# derive_from_args — the ten-argument call, read off the one parser's namespace
# --------------------------------------------------------------------------- #


def test_derive_from_args_reads_the_dests_the_parser_declares(monkeypatch):
    """The knobs are lifted off ``args`` rather than respelled per call site,
    so a renamed flag must break here rather than in one of two branches."""
    monkeypatch.setattr(sys, "argv", ["prog", "--mode", "derive_groups"])
    from anime_tools.tagger.cli.main import parse_args

    args = parse_args()
    for dest in _knobs():
        assert hasattr(args, dest), f"--{dest} is gone from tagger.cli.main"


def test_derive_from_args_scores_and_counts(tmp_path):
    _, kb_csv = _corpus(tmp_path)
    kb = load_tag_knowledge_base(kb_csv)
    vocab = _vocab(["blue eyes", "red eyes", "long hair", "short hair", "solo"])

    rows, unmatched, n_general = derive_from_args(
        argparse.Namespace(**_knobs()),
        vocab,
        kb,
        NO_RULES,
        _solo_sets_from_captions(),
        source="a test corpus",
    )
    assert n_general == 5  # every vocab tag, matched or not
    assert unmatched == ["solo"]  # no taxonomy row in the KB
    by_path = {r["path"]: r for r in rows}
    assert set(by_path) == {EYE_PATH, HAIR_PATH}
    # Eye colour never doubles up on a solo image → mutually exclusive.
    assert by_path[EYE_PATH]["multi_rate"] == 0.0
    assert by_path[EYE_PATH]["mode"] == "softmax_when_solo"


def test_derive_from_args_honours_each_knob(tmp_path):
    _, kb_csv = _corpus(tmp_path)
    kb = load_tag_knowledge_base(kb_csv)
    vocab = _vocab(["blue eyes", "red eyes", "long hair", "short hair"])
    solo = _solo_sets_from_captions()

    def paths(**over):
        rows, _, _ = derive_from_args(
            argparse.Namespace(**_knobs(**over)),
            vocab,
            kb,
            NO_RULES,
            solo,
            source="a test corpus",
        )
        return {r["path"]: r for r in rows}

    assert paths(min_group_size=3) == {}  # both buckets hold exactly two
    assert paths(min_member_freq=101) == {}  # every member is freq=100
    # Below support, exclusivity is not trusted: multilabel with a low_support tier.
    assert paths(min_group_support=99)[EYE_PATH]["tier"] == "low_support"
    assert paths(min_group_support=99)[EYE_PATH]["mode"] == "multilabel"


# --------------------------------------------------------------------------- #
# write_merged_groups — back up, write, read back
# --------------------------------------------------------------------------- #


def _rows(tmp_path: Path) -> list[dict]:
    _, kb_csv = _corpus(tmp_path)
    rows, _, _ = derive_from_args(
        argparse.Namespace(**_knobs()),
        _vocab(["blue eyes", "red eyes", "long hair", "short hair"]),
        load_tag_knowledge_base(kb_csv),
        NO_RULES,
        _solo_sets_from_captions(),
        source="a test corpus",
    )
    return rows


def test_write_merged_groups_writes_and_reads_back(tmp_path):
    dest = tmp_path / "groups.yaml"
    merged = write_merged_groups(_rows(tmp_path), dest, None, min_group_size=2)

    # The returned object *is* the written file loaded through the real
    # consumer, so the caller never parses it a second time.
    assert {g.name for g in merged.groups} == {"eye_color", "hair_length"}
    assert merged.to_dict() == tg.load_groups(dest).to_dict()
    assert not dest.with_suffix(".yaml.bak").exists()  # nothing to back up


def test_write_merged_groups_backs_up_the_file_it_replaces(tmp_path):
    dest = tmp_path / "groups.yaml"
    dest.write_text("version: 1\n", encoding="utf-8")
    write_merged_groups(_rows(tmp_path), dest, None, min_group_size=2)

    backup = tmp_path / "groups.yaml.bak"
    # Both callers used to spell this themselves — ``.with_suffix(".yaml.bak")``
    # in one, ``dest.suffix + ".bak"`` in the other — and they agree only for a
    # ``.yaml`` dest, which ``--out_yaml`` does not have to be.
    assert backup.read_text(encoding="utf-8") == "version: 1\n"


def test_write_merged_groups_backs_up_even_when_dest_is_preserve(tmp_path):
    """``build_vocab`` used to skip the backup exactly here — the one case where
    the file being overwritten is the hand-curated one."""
    dest = tmp_path / "groups.yaml"
    dest.write_text(
        "version: 1\neye_color:\n  mode: softmax\n  tags: [blue eyes, red eyes]\n",
        encoding="utf-8",
    )
    merged = write_merged_groups(_rows(tmp_path), dest, dest, min_group_size=2)

    assert "eye_color" in (tmp_path / "groups.yaml.bak").read_text(encoding="utf-8")
    # Preserved verbatim, and the derived eye_color deferred to it.
    assert merged.by_name("eye_color").mode == "softmax"


def test_write_merged_groups_rejects_a_merge_that_cannot_be_loaded(tmp_path):
    """The read-back is the validation: it runs on both callers now, not one."""
    rows = _rows(tmp_path)
    preserve = tmp_path / "curated.yaml"
    # A group claiming a tag another group also claims — the loader's disjoint
    # invariant, which only the file on disk can violate.
    preserve.write_text(
        "version: 1\n"
        "one:\n  mode: multilabel\n  tags: [x]\n"
        "two:\n  mode: multilabel\n  tags: [x]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        write_merged_groups(rows, tmp_path / "groups.yaml", preserve, min_group_size=2)


# --------------------------------------------------------------------------- #
# The two callers, on one corpus
# --------------------------------------------------------------------------- #


def _build_args(tmp_path: Path, root: Path, kb_csv: Path, out_dir: Path, **over):
    return argparse.Namespace(
        out_dir=str(out_dir),
        rules=str(tmp_path / "rules.yaml"),
        caption_roots=[str(root)],
        tag_cache=str(kb_csv),
        groups=None,
        derive_groups=True,
        min_freq=1,
        val_frac=0.25,
        seed=0,
        **_knobs(**over),
    )


def test_build_vocab_derives_merges_and_bakes_the_groups_in(tmp_path):
    root, kb_csv = _corpus(tmp_path)
    (tmp_path / "rules.yaml").write_text("{}\n", encoding="utf-8")
    out_dir = tmp_path / "ckpt"

    cmd_build_vocab(_build_args(tmp_path, root, kb_csv, out_dir))

    groups = tg.load_groups(out_dir / "groups.yaml")
    assert {g.name for g in groups.groups} == {"eye_color", "hair_length"}
    vocab = read_json(out_dir / "vocab.json")
    assert [g["name"] for g in vocab["groups"]] == ["eye_color", "hair_length"]
    # Both promoted families are sentinel-typed, so each got a vocab slot.
    assert [t["name"] for t in vocab["tags"] if t["freq"] == 0] == [
        tg.sentinel_tag_name("eye_color"),
        tg.sentinel_tag_name("hair_length"),
    ]


def test_build_vocab_survives_a_missing_kb(tmp_path, caplog):
    """The tolerance that is ``vocab.py``'s own: no KB is a warning and a
    flat-vocab build, never a raise."""
    import anime_tools.captions.correction as C

    root, _ = _corpus(tmp_path)
    (tmp_path / "rules.yaml").write_text("{}\n", encoding="utf-8")
    tag_cache = tmp_path / "cache.json"  # not a .csv → find_tag_csv() is consulted
    tag_cache.write_text('{"blue_eyes": 0}', encoding="utf-8")
    out_dir = tmp_path / "ckpt"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "find_tag_csv", lambda *a, **k: None)
        cmd_build_vocab(_build_args(tmp_path, root, tag_cache, out_dir))

    assert "KB not found" in caplog.text
    assert not (out_dir / "groups.yaml").exists()
    assert read_json(out_dir / "vocab.json")["groups"] == []


def test_both_callers_derive_the_same_groups(tmp_path):
    """The shared half is shared: given one corpus, ``build_vocab
    --derive_groups`` and ``--mode derive_groups --apply`` write the same
    groups. Only the co-occurrence *source* is the caller's, so the standalone
    run is pointed at a checkpoint with no ``dataset.json`` — the caption-scan
    fallback, which is the scan the build did inline."""
    from anime_tools.tagger.cli.derive_groups import cmd_derive_groups

    root, kb_csv = _corpus(tmp_path)
    (tmp_path / "rules.yaml").write_text("{}\n", encoding="utf-8")
    out_dir = tmp_path / "ckpt"
    cmd_build_vocab(_build_args(tmp_path, root, kb_csv, out_dir))

    standalone = tmp_path / "standalone"
    standalone.mkdir()
    (standalone / "vocab.json").write_text(
        (out_dir / "vocab.json").read_text(encoding="utf-8"), encoding="utf-8"
    )  # no dataset.json → the caption-scan branch
    cmd_derive_groups(
        argparse.Namespace(
            out_dir=str(standalone),
            tag_cache=str(kb_csv),
            rules=str(tmp_path / "rules.yaml"),
            caption_roots=[str(root)],
            report=False,
            apply=True,
            out_yaml=None,
            preserve_groups=None,
            **_knobs(),
        )
    )

    def shape(path: Path):
        return {
            g.name: (g.mode, tuple(g.tags), g.sentinel)
            for g in tg.load_groups(path).groups
        }

    assert shape(standalone / "groups.yaml") == shape(out_dir / "groups.yaml")
