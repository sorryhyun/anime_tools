"""Sentinel-augmented softmax groups — the resolution layer.

Sentinel groups relax the softmax-group contract from "exactly one member
present" to "at most one": a synthetic ``<none:group>`` vocab slot that
``AnimaTagger.predict`` emits nothing for when it wins the group argmax.

1. Resolution folds the sentinel slot into ``tag_indices`` (last) and records
   ``sentinel_index``; a vocab built without the slot degrades to legacy.
2. ``$category:<cat>`` member markers expand against the vocab.
3. A sentinel name round-trips through ``sentinel_tag_name`` /
   ``is_sentinel_name``.
"""

from __future__ import annotations

from anime_tools.captions import tag_groups as tg

# --------------------------------------------------------------------------- #
# Resolution layer
# --------------------------------------------------------------------------- #


def _groups_yaml_dict(sentinel: bool) -> dict:
    return {
        "version": 1,
        "medium": {
            "mode": "softmax",
            "tags": ["watercolor", "oil painting"],
            **({"sentinel": True} if sentinel else {}),
        },
    }


def test_resolve_folds_sentinel_last_and_records_index():
    groups = tg.from_dict(_groups_yaml_dict(sentinel=True))
    vocab_idx = {
        "watercolor": 0,
        "oil painting": 1,
        tg.sentinel_tag_name("medium"): 2,
    }
    resolved, dropped = tg.resolve_groups(groups, vocab_idx)
    (g,) = resolved
    assert g.sentinel_index == 2
    assert g.tag_indices == (0, 1, 2)
    assert g.tag_names[-1] == tg.sentinel_tag_name("medium")
    assert not dropped


def test_resolve_without_slot_degrades_to_legacy():
    groups = tg.from_dict(_groups_yaml_dict(sentinel=True))
    vocab_idx = {"watercolor": 0, "oil painting": 1}  # pre-sentinel vocab
    resolved, dropped = tg.resolve_groups(groups, vocab_idx)
    (g,) = resolved
    assert g.sentinel_index is None
    assert g.tag_indices == (0, 1)
    assert tg.sentinel_tag_name("medium") in dropped


def test_expand_category_members():
    groups = tg.from_dict(
        {
            "version": 1,
            "artist": {
                "mode": "softmax",
                "sentinel": True,
                "tags": ["$category:artist"],
            },
        }
    )
    name_to_cat = {"@mignon": "artist", "@sincos": "artist", "blush": "general"}
    out = tg.expand_category_members(groups, name_to_cat)
    (g,) = out.groups
    assert g.tags == ("@mignon", "@sincos")
    assert g.sentinel


def test_sentinel_name_roundtrip():
    name = tg.sentinel_tag_name("body_shape")
    assert tg.is_sentinel_name(name)
    assert not tg.is_sentinel_name("body_shape")
    assert not tg.is_sentinel_name("<3")
