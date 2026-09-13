"""``aliases:`` in tag_rules.yaml — tag-level renames of retired booru names.

A raw ``replacements`` entry would also rewrite any longer tag the name
prefixes; an alias only ever renames a whole tag, and folds onto a target the
caption already carries.
"""

from __future__ import annotations

from anime_tools.captions import tag_rules as tr

RULES = tr.from_dict(
    {
        "replacements": {"&#039;": "'"},
        "aliases": {
            "silver hair": "grey hair",
            "light blue hair": "blue hair",
            "dark blue hair": "blue hair",
            "torn legwear": "torn clothes",
        },
        "remove": ["absurdres"],
        "bra": ["black bra"],
    }
)


def test_round_trip_keeps_aliases_and_no_alias_key_when_empty():
    assert tr.from_dict(RULES.to_dict()) == RULES
    assert "aliases" not in tr.from_dict({"remove": []}).to_dict()
    # aliases is a reserved key, never a dedup base
    assert "aliases" not in RULES.dedup


def test_alias_renames_whole_tags_only_and_keeps_order():
    tags = ["1girl", "silver hair", "silver hairband", "long hair"]
    assert tr.apply_rules(tags, RULES) == [
        "1girl",
        "grey hair",
        "silver hairband",
        "long hair",
    ]


def test_alias_folds_onto_a_target_already_present_once():
    assert tr.apply_rules(
        ["blue hair", "light blue hair", "dark blue hair"], RULES
    ) == ["blue hair"]
    assert tr.apply_rules(["light blue hair", "solo", "blue hair"], RULES) == [
        "blue hair",
        "solo",
    ]


def test_aliases_run_before_remove_and_dedup():
    # the remove list and dedup bases are spelled in live names; an alias lands
    # a tag on them before they are consulted
    rules = tr.from_dict(
        {"aliases": {"torn legwear": "absurdres"}, "remove": ["absurdres"]}
    )
    assert tr.apply_rules(["torn legwear", "solo"], rules) == ["solo"]


def test_parse_caption_applies_aliases():
    assert tr.parse_caption("safe, silver hair, torn clothes, torn legwear", RULES) == [
        "safe",
        "grey hair",
        "torn clothes",
    ]
