"""Position-aware caption clauses — parse/compose, variants, and the pipeline.

1. Clause parsing round-trips: ``.`` delimits clauses, ``,`` delimits tags.
2. Clauses are atomic under variant generation: a shuffled variant never moves a
   clause tag into the flat bag or into a different clause.
3. The pipeline never writes a clause it cannot ground — count disagreement, too
   few instances and hallucinated character names all skip.
4. The v2 rewrite removes a tag from the flat bag only when exactly one clause
   claims it, the bag corroborates a per-subject reading, no other crop kept it,
   and the winner clears the runner-up by the attribution margin (relative to
   the winner's own probability, since per-tag thresholds vary widely). Failing
   any rule degrades to v1 for that tag — bound and still flat.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anime_tools.captions.position_clauses import (
    assign_positions,
    compose_caption,
    has_clauses,
    horizontal_names,
    ordered_indices,
    parse_caption,
    tag_spans,
)

GT = (
    "sensitive, 2girls, original, @rurudo, cat girl, navel. "
    "On the left, red eyes, gray hair. "
    "On the right, aqua eyes, pink hair."
)


# ----- parse / compose ----------------------------------------------------


def test_parse_splits_flat_bag_from_clauses():
    parsed = parse_caption(GT)
    assert parsed.flat_tags == (
        "sensitive",
        "2girls",
        "original",
        "@rurudo",
        "cat girl",
        "navel",
    )
    assert [c.position for c in parsed.clauses] == ["left", "right"]
    assert parsed.clauses[0].tags == ("red eyes", "gray hair")
    # The clause-terminating period is not part of the last tag.
    assert parsed.clauses[1].tags == ("aqua eyes", "pink hair")


def test_parse_compose_round_trip():
    assert parse_caption(GT).render() == GT


def test_parse_caption_without_clauses_is_flat():
    parsed = parse_caption("1girl, blue hair, smile")
    assert not parsed.has_clauses
    assert parsed.render() == "1girl, blue hair, smile"


def test_parse_accepts_comma_form_header():
    # Some hand-written captions separate the first clause with a comma rather
    # than a period; both parse, and compose normalizes to the period form.
    parsed = parse_caption("1girl, white socks, On the left, red eyes")
    assert parsed.flat_tags == ("1girl", "white socks")
    assert parsed.clauses[0].tags == ("red eyes",)
    assert parsed.render() == "1girl, white socks. On the left, red eyes."


def test_punctuation_tags_survive_the_period_strip():
    parsed = parse_caption("1girl, :d, >_<. On the left, :3.")
    assert parsed.flat_tags == ("1girl", ":d", ">_<")
    assert parsed.clauses[0].tags == (":3",)


def test_has_clauses_detects_both_forms():
    assert has_clauses(GT)
    assert has_clauses("1girl, On the left, red eyes")
    assert not has_clauses("1girl, blue hair, on the beach")


def test_lowercase_period_form_parses_and_canonicalizes():
    """`has_clauses` and `parse_caption` agree on a lowercase period-form header."""
    caption = "safe, 2girls. on the left, red eyes. on the right, aqua eyes."
    assert has_clauses(caption)
    parsed = parse_caption(caption)
    assert parsed.flat_tags == ("safe", "2girls")
    assert [c.header for c in parsed.clauses] == ["On the left", "On the right"]
    assert parsed.clauses[0].tags == ("red eyes",)
    # Emission is canonical regardless of how it was written.
    assert parsed.render() == (
        "safe, 2girls. On the left, red eyes. On the right, aqua eyes."
    )


def test_a_lowercase_scene_tag_in_the_bag_is_not_a_header():
    """Only the period form is case-insensitive: a bare ``on the beach`` mid-bag
    is a scene tag, not a header."""
    parsed = parse_caption("1girl, blue hair, on the beach")
    assert parsed.flat_tags == ("1girl", "blue hair", "on the beach")
    assert not parsed.clauses


def test_tag_spans_slice_the_caption_they_came_from():
    """A span is offsets: slicing the source with it gives the tag back, with
    whitespace, the comma and the terminating period outside the span."""
    for span in tag_spans(GT):
        assert GT[span.start : span.end] == span.text
        assert span.text == span.text.strip()
        assert not span.text.endswith(".")
    # Two adjacent tags: the ", " between them belongs to neither.
    safe, girls = tag_spans(GT)[:2]
    assert GT[safe.end : girls.start] == ", "


def test_tag_spans_and_parse_caption_are_the_same_walk():
    """``parse_caption`` is written on top of ``tag_spans``; the two agree on every
    caption shape above."""
    for caption in (
        GT,
        "1girl, blue hair, smile",
        "1girl, white socks, On the left, red eyes",
        "safe, 2girls. on the left, red eyes. on the right, aqua eyes.",
        "1girl, :d, >_<. On the left, :3.",
        "1girl, blue hair, on the beach",
        "",
    ):
        parsed = parse_caption(caption)
        spans = tag_spans(caption)
        assert tuple(s.text for s in spans if s.clause < 0) == parsed.flat_tags
        # Lowercased on both sides: a span is the source slice, where ``header``
        # is the canonical emission form.
        assert [s.text.lower() for s in spans if s.kind == "header"] == [
            c.header.lower() for c in parsed.clauses
        ]
        for i, clause in enumerate(parsed.clauses):
            body = tuple(s.text for s in spans if s.clause == i and s.kind != "header")
            assert body == clause.tags


def test_tag_span_kinds_name_the_artist_and_the_header():
    kinds = {s.text: s.kind for s in tag_spans(GT)}
    assert kinds["@rurudo"] == "artist"
    assert kinds["On the left"] == "header"
    assert kinds["cat girl"] == "tag"
    # ``@ @`` is the booru eye-shape tag, not a handle.
    assert [s.kind for s in tag_spans("1girl, @ @")] == ["tag", "tag"]


def test_compose_without_clauses_is_a_plain_join():
    assert compose_caption(["a", "b"]) == "a, b"


# ----- position vocabulary ------------------------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [
        (2, ["left", "right"]),
        (3, ["left", "middle", "right"]),
        (4, ["leftmost", "second from left", "third from left", "rightmost"]),
    ],
)
def test_horizontal_names(n, expected):
    assert horizontal_names(n) == expected


def test_assign_positions_orders_left_to_right():
    # Deliberately out of order: the names must follow geometry, not input order.
    boxes = [(600, 0, 800, 400), (0, 0, 200, 400), (300, 0, 500, 400)]
    assert assign_positions(boxes, (1000, 400)) == ["right", "left", "middle"]


def test_assign_positions_is_row_aware_for_grid_sheets():
    # A 2x2 contact sheet: pure x-ordering would interleave the rows and call
    # the bottom-left view "left" alongside the top-left one.
    boxes = [
        (0, 0, 400, 400),  # top left
        (600, 0, 1000, 400),  # top right
        (0, 600, 400, 1000),  # bottom left
        (600, 600, 1000, 1000),  # bottom right
    ]
    assert assign_positions(boxes, (1000, 1000)) == [
        "top left",
        "top right",
        "bottom left",
        "bottom right",
    ]
    assert ordered_indices(boxes, (1000, 1000)) == [0, 1, 2, 3]


def test_single_subject_row_gets_the_bare_row_word():
    boxes = [(400, 0, 600, 300), (0, 700, 300, 1000), (700, 700, 1000, 1000)]
    assert assign_positions(boxes, (1000, 1000)) == [
        "top",
        "bottom left",
        "bottom right",
    ]


def test_magazine_layout_names_columns_not_fake_rows():
    """A full-height subject beside a column of stacked panels (`9760139`): the
    x-axis separates, so the stack takes `top left`/`bottom left` and the girl
    spanning the height is bare `right`.
    """
    boxes = [
        (0, 0, 455, 450),  # panel, upper left
        (0, 405, 478, 890),  # panel, lower left
        (440, 78, 625, 830),  # full-height girl, right
    ]
    assert assign_positions(boxes, (639, 900)) == [
        "top left",
        "bottom left",
        "right",
    ]
    # Reading order is column-major here: the left stack, then the girl.
    assert ordered_indices(boxes, (639, 900)) == [0, 1, 2]


def test_a_lone_end_hugging_panel_earns_its_row_qualifier():
    """`9760121`: the panel leaves the bottom half empty, so it reads `top left`;
    the girl spans the height and stays bare `right`."""
    boxes = [(375, 80, 600, 860), (0, 10, 420, 435)]
    assert assign_positions(boxes, (619, 900)) == ["right", "top left"]


def test_a_diagonal_pair_reads_by_its_corners():
    boxes = [(0, 0, 400, 500), (600, 400, 1000, 1000)]
    assert assign_positions(boxes, (1000, 1000)) == ["top left", "bottom right"]


def test_a_wide_panel_bleeding_under_a_neighbour_stays_in_its_lane():
    """`5828187`: the bottom-left panel's box reaches under the right girl,
    overlapping her x-extent completely; center-gap lanes keep it in the left
    column."""
    boxes = [(15, 0, 450, 515), (0, 395, 585, 890), (410, 90, 585, 820)]
    assert assign_positions(boxes, (639, 900)) == [
        "top left",
        "bottom left",
        "right",
    ]
    assert ordered_indices(boxes, (639, 900)) == [0, 1, 2]


def test_nested_boxes_fall_back_to_left_right_in_reading_order():
    """`5828184`: the lying girl's box contains the standing girl's x-extent, but
    their centers sit in separate lanes and the caption reads left→right."""
    boxes = [(375, 70, 600, 840), (0, 90, 619, 890)]
    assert assign_positions(boxes, (619, 900)) == ["right", "left"]
    assert ordered_indices(boxes, (619, 900)) == [1, 0]


def test_a_panel_reaching_the_halfway_line_stays_bare_left():
    """`6183990`: the panel's bottom reaches the halfway line (empty gap
    0.452 < _EDGE_CLEAR), so it is bare `left`, not `top left`."""
    boxes = [(16, 0, 491, 466), (406, 87, 573, 851)]
    assert assign_positions(boxes, (627, 900)) == ["left", "right"]


def test_same_height_subjects_stay_bare_left_right():
    """The row qualifier does not fire on ordinary side-by-side pairs."""
    boxes = [(0, 0, 400, 1000), (600, 380, 1000, 1000)]
    assert assign_positions(boxes, (1000, 1000)) == ["left", "right"]


# ----- caption variants: clauses are atomic -------------------------------


def _variants(*args, **kwargs):
    from anime_tools.captions.variants import generate_caption_variants

    return generate_caption_variants(*args, **kwargs)


def test_clause_tags_never_leak_into_the_flat_bag():
    random.seed(0)
    for text in _variants(GT, num_variants=12, tag_dropout_rate=0.3):
        parsed = parse_caption(text)
        flat = set(parsed.flat_tags)
        for clause_tag in ("red eyes", "gray hair", "aqua eyes", "pink hair"):
            assert clause_tag not in flat, text


def test_clause_tags_never_cross_between_clauses():
    random.seed(1)
    left, right = {"red eyes", "gray hair"}, {"aqua eyes", "pink hair"}
    for text in _variants(GT, num_variants=12, tag_dropout_rate=0.3):
        for clause in parse_caption(text).clauses:
            owner = left if clause.position == "left" else right
            assert set(clause.tags) <= owner, text


def test_clause_is_dropped_whole_or_kept_whole():
    random.seed(2)
    for text in _variants(GT, num_variants=16, tag_dropout_rate=0.5):
        for clause in parse_caption(text).clauses:
            expected = (
                ("red eyes", "gray hair")
                if clause.position == "left"
                else (
                    "aqua eyes",
                    "pink hair",
                )
            )
            assert sorted(clause.tags) == sorted(expected), text


def test_clause_dropout_rate_one_removes_every_clause():
    random.seed(3)
    out = _variants(GT, num_variants=6, tag_dropout_rate=0.0, clause_dropout_rate=1.0)
    assert out[0] == GT  # v0 is always pristine
    for text in out[1:]:
        assert not parse_caption(text).has_clauses


def test_clause_dropout_rate_zero_keeps_every_clause():
    random.seed(4)
    for text in _variants(
        GT, num_variants=8, tag_dropout_rate=0.9, clause_dropout_rate=0.0
    ):
        assert len(parse_caption(text).clauses) == 2


def test_artist_prefix_still_protected_with_clauses():
    random.seed(5)
    for text in _variants(GT, num_variants=8, tag_dropout_rate=1.0):
        assert "@rurudo" in parse_caption(text).flat_tags


def test_no_clause_caption_is_byte_identical_at_v0():
    raw = "@sincos,blue hair  ,1girl"
    assert _variants(raw, num_variants=1, tag_dropout_rate=0.0)[0] == raw


def test_clause_headers_are_never_identity_randomized():
    random.seed(6)
    pool = ["swing", "sodium", "awards"]
    for text in _variants(
        GT,
        num_variants=8,
        tag_dropout_rate=0.0,
        clause_dropout_rate=0.0,
        tag_randomize_rate=1.0,
        erasure_pool=pool,
    )[1:]:
        assert [c.position for c in parse_caption(text).clauses] == ["left", "right"]


# ----- order correction keeps clauses intact ------------------------------


def test_correct_caption_does_not_dissolve_clauses():
    from anime_tools.captions.correction import TagKnowledgeBase, correct_caption

    kb = TagKnowledgeBase({}, Path("stub.csv"))
    out = correct_caption(GT, kb).text
    parsed = parse_caption(out)
    assert [c.position for c in parsed.clauses] == ["left", "right"]
    assert parsed.clauses[0].tags == ("red eyes", "gray hair")
    # The flat bag is still reordered into buckets (rating first, artist slot).
    assert parsed.flat_tags[0] == "sensitive"
    assert "@rurudo" in parsed.flat_tags


# ----- pipeline -----------------------------------------------------------


@pytest.fixture
def pipeline_bits():
    from PIL import Image

    from anime_tools.stages.position_captions import (
        ClauseVocabulary,
        Detection,
        PositionCaptionOptions,
    )

    vocabulary = ClauseVocabulary(
        characters=frozenset({"akita neru", "hatsune miku"}),
        excluded=frozenset({"vocaloid"}),
        # The bag gate is derived from this set (ClauseVocabulary.gated_groups).
        # ``framing`` is exclusive too, hence the gate exemption pinned below.
        exclusive_groups=frozenset(
            {"hair_color", "eye_color", "body_shape", "framing"}
        ),
        tag_to_group={
            "blonde hair": "hair_color",
            "aqua hair": "hair_color",
            "green hair": "hair_color",
            "red eyes": "eye_color",
            "twintails": "hairstyle",
            "long hair": "hair_length",
            "large breasts": "body_shape",
            "flat chest": "body_shape",
            "maid": "costume",
            "playboy bunny": "costume",
            "ass": "body_parts",
            "thighs": "body_parts",
            "simple background": "background_detail",
            "white background": "background_detail",
            "full body": "framing",
            "ass focus": "framing",
            "close-up": "framing",
            "solo focus": "framing",
            "white border": "framing",
        },
    )
    image = Image.new("RGB", (1000, 500), "white")
    return image, vocabulary, Detection, PositionCaptionOptions


def _detector(boxes_by_threshold):
    """Stub ``detect_fn``: threshold → list of (box, score)."""
    from anime_tools.stages.position_captions import Detection

    def detect(image, score_threshold):
        for thr in sorted(boxes_by_threshold, reverse=True):
            if score_threshold >= thr:
                return [Detection(box=b, score=s) for b, s in boxes_by_threshold[thr]]
        lowest = min(boxes_by_threshold)
        return [Detection(box=b, score=s) for b, s in boxes_by_threshold[lowest]]

    return detect


def _tagger(per_crop):
    """Stub ``tag_fn`` returning ``per_crop`` predictions in call order."""
    calls = {"n": 0}

    def tag(crop):
        out = per_crop[calls["n"] % len(per_crop)]
        calls["n"] += 1
        return out

    return tag


_TWO_GIRLS_CAPTION = (
    "safe, 2girls, akita neru, hatsune miku, @channel, blonde hair, aqua hair, "
    "simple background"
)


def _two_girls_proposal(pipeline_bits, **option_overrides):
    """The canonical two-subject image: one name + one hair color per side."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    return propose_for_image(
        image,
        _TWO_GIRLS_CAPTION,
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {
                    "kept": {
                        "akita neru": 0.9,
                        "blonde hair": 0.8,
                        "simple background": 0.7,
                    },
                    "groups": {"hair_color": "blonde hair"},
                },
                {
                    "kept": {
                        "hatsune miku": 0.9,
                        "aqua hair": 0.8,
                        "twintails": 0.7,
                        "simple background": 0.7,
                    },
                    "groups": {"hair_color": "aqua hair"},
                },
            ]
        ),
        vocabulary=vocabulary,
        options=Options(**option_overrides),
    )


def test_propose_binds_hair_color_to_each_side(pipeline_bits):
    proposal = _two_girls_proposal(pipeline_bits)
    assert proposal.ok
    parsed = parse_caption(proposal.proposed)
    assert parsed.clauses[0].tags[:2] == ("akita neru", "blonde hair")
    assert parsed.clauses[1].tags[:2] == ("hatsune miku", "aqua hair")
    # Kept on BOTH crops → not attributable → stays out of every clause.
    assert not any("simple background" in c.tags for c in parsed.clauses)


def test_v2_moves_each_bound_attribute_out_of_the_flat_bag(pipeline_bits):
    """Every attribute is asserted exactly once, where it belongs."""
    proposal = _two_girls_proposal(pipeline_bits)
    parsed = parse_caption(proposal.proposed)
    # The bag keeps the cast list and what describes the image.
    assert parsed.flat_tags == (
        "safe",
        "2girls",
        "akita neru",
        "hatsune miku",
        "@channel",
        "simple background",
    )
    assert {m["tag"] for m in proposal.moved} == {"blonde hair", "aqua hair"}
    # Nothing is lost: every moved tag is still in the caption, in one clause.
    bound = {t for c in parsed.clauses for t in c.tags}
    assert {"blonde hair", "aqua hair"} <= bound


def test_the_cast_list_stays_in_the_flat_bag(pipeline_bits):
    """Names are bound *and* kept flat: the bag is the cast list, the clause says
    which one is where. No non-name attribute is duplicated this way."""
    proposal = _two_girls_proposal(pipeline_bits)
    flat = parse_caption(proposal.proposed).flat_tags
    assert "akita neru" in flat and "hatsune miku" in flat
    assert proposal.pinned["akita neru"] == "character-name"


def test_no_rewrite_restores_the_additive_v1_caption(pipeline_bits):
    proposal = _two_girls_proposal(pipeline_bits, rewrite=False)
    parsed = parse_caption(proposal.proposed)
    assert parsed.flat_tags == tuple(t.strip() for t in _TWO_GIRLS_CAPTION.split(","))
    assert not proposal.moved
    # …and the clauses are identical to the rewritten arm's.
    rewritten = parse_caption(_two_girls_proposal(pipeline_bits).proposed)
    assert [c.render() for c in parsed.clauses] == [
        c.render() for c in rewritten.clauses
    ]


def _views_proposal(pipeline_bits, caption, per_crop, **option_overrides):
    """A two-view sheet of one character: different outfit per view."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    return propose_for_image(
        image,
        caption,
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(per_crop),
        vocabulary=vocabulary,
        options=Options(**option_overrides),
    )


def test_a_single_characters_attributes_never_leave_the_bag(pipeline_bits):
    """Two views of one girl are not two girls: ``blonde hair`` stays flat, while
    the outfit that genuinely differs per view moves. Pinned at
    ``multi_view_gate=False``, i.e. the removal rule alone.
    """
    proposal = _views_proposal(
        pipeline_bits,
        "safe, 1girl, multiple views, akita neru, blonde hair, maid, playboy bunny",
        [
            {
                "kept": {"blonde hair": 0.9, "maid": 0.8},
                "scores": {"blonde hair": 0.9, "maid": 0.8, "playboy bunny": 0.01},
                "groups": {"hair_color": "blonde hair"},
            },
            {
                "kept": {"playboy bunny": 0.8},
                "scores": {"blonde hair": 0.2, "maid": 0.02, "playboy bunny": 0.8},
                "groups": {},
            },
        ],
        multi_view_gate=False,
    )
    assert proposal.ok
    parsed = parse_caption(proposal.proposed)
    assert "blonde hair" in parsed.flat_tags
    assert proposal.pinned["blonde hair"] == "sole-value"
    # …and it is still bound to the view it was read off (v1 behaviour for it).
    assert "blonde hair" in parsed.clauses[0].tags
    # The per-view outfits move: that is what the sheet was ambiguous about.
    assert "maid" not in parsed.flat_tags and "playboy bunny" not in parsed.flat_tags
    assert "maid" in parsed.clauses[0].tags
    assert "playboy bunny" in parsed.clauses[1].tags


def test_two_tone_hair_is_one_character_not_two(pipeline_bits):
    """``multicolored hair`` explains the second hair color without a second girl."""
    proposal = _views_proposal(
        pipeline_bits,
        "safe, 1girl, multiple views, multicolored hair, blonde hair, aqua hair, "
        "maid, playboy bunny",
        [
            {
                "kept": {"blonde hair": 0.9, "maid": 0.8},
                "scores": {"blonde hair": 0.9, "aqua hair": 0.1, "maid": 0.8},
                "groups": {"hair_color": "blonde hair"},
            },
            {
                "kept": {"aqua hair": 0.9, "playboy bunny": 0.8},
                "scores": {"blonde hair": 0.1, "aqua hair": 0.9, "playboy bunny": 0.8},
                "groups": {"hair_color": "aqua hair"},
            },
        ],
        # The removal layer again — the multi-view gate would keep both colors
        # out of the clauses entirely, leaving this rule nothing to decide.
        multi_view_gate=False,
    )
    assert {"blonde hair", "aqua hair"} <= set(
        parse_caption(proposal.proposed).flat_tags
    )
    assert proposal.pinned["blonde hair"] == "two-tone-marker"
    assert proposal.pinned["aqua hair"] == "two-tone-marker"


def test_a_contested_tag_stays_in_the_bag(pipeline_bits):
    """Below the margin the tag is shared, not attributable — v1 for that one tag."""
    proposal = _views_proposal(
        pipeline_bits,
        "safe, 2girls, akita neru, hatsune miku, maid, playboy bunny",
        [
            {
                "kept": {"akita neru": 0.9, "maid": 0.55},
                "scores": {"maid": 0.55, "playboy bunny": 0.01},
                "groups": {},
            },
            {
                # Just under the keep threshold: removing `maid` from the bag
                # would deny it of this girl.
                "kept": {"hatsune miku": 0.9, "playboy bunny": 0.9},
                "scores": {"maid": 0.45, "playboy bunny": 0.9},
                "groups": {},
            },
        ],
    )
    parsed = parse_caption(proposal.proposed)
    assert "maid" in parsed.flat_tags
    assert proposal.pinned["maid"] == "margin"
    assert "maid" in parsed.clauses[0].tags  # still bound, just also still flat
    assert "playboy bunny" not in parsed.flat_tags  # 0.9 vs 0.0 clears the margin


def test_a_low_threshold_tag_the_other_crop_never_saw_moves(pipeline_bits):
    """A tag whose F1 threshold is ~0.05 can be a decisive 0.34-vs-0.000 call
    without producing a 0.35 absolute gap, so the margin is relative to the
    winner.
    """
    proposal = _views_proposal(
        pipeline_bits,
        "safe, 2girls, akita neru, hatsune miku, sleeves past fingers, maid",
        [
            {
                # 0.342 clears the tag's own (low) threshold, so it is kept.
                "kept": {"akita neru": 0.9, "sleeves past fingers": 0.342},
                "scores": {"sleeves past fingers": 0.342, "maid": 0.01},
                "groups": {},
            },
            {
                "kept": {"hatsune miku": 0.9, "maid": 0.9},
                "scores": {"sleeves past fingers": 0.0, "maid": 0.9},
                "groups": {},
            },
        ],
    )
    parsed = parse_caption(proposal.proposed)
    assert "sleeves past fingers" not in parsed.flat_tags
    assert "sleeves past fingers" in parsed.clauses[0].tags
    assert "margin" not in proposal.pinned.get("sleeves past fingers", "")


def test_a_tag_the_other_crop_kept_stays_in_the_bag(pipeline_bits):
    """Kept on both crops but bound to one clause is a selection artifact, so the
    bag keeps the tag however wide the score gap."""
    from anime_tools.stages.position_captions import (
        ClauseVocabulary,
        RemovalPlan,
        plan_bag_removals,
    )

    vocabulary = ClauseVocabulary(tag_to_group={"maid": "costume"})
    plan = plan_bag_removals(
        ("2girls", "maid"),
        [["maid"], []],
        ["left", "right"],
        [{"maid": 0.99}, {"maid": 0.55}],
        [{"maid": 0.99}, {"maid": 0.55}],
        vocabulary=vocabulary,
        margin=0.25,
    )
    assert plan == RemovalPlan(moved=(), blocked={"maid": "multi-kept"})


def test_a_tag_bound_to_two_subjects_stays_in_the_bag(pipeline_bits):
    from anime_tools.stages.position_captions import (
        ClauseVocabulary,
        RemovalPlan,
        plan_bag_removals,
    )

    vocabulary = ClauseVocabulary(tag_to_group={"maid": "costume"})
    plan = plan_bag_removals(
        ("2girls", "maid"),
        [["maid"], ["maid"]],
        ["left", "right"],
        [{"maid": 0.9}, {"maid": 0.9}],
        [{"maid": 0.9}, {"maid": 0.9}],
        vocabulary=vocabulary,
        margin=0.35,
    )
    assert plan == RemovalPlan(moved=(), blocked={"maid": "multi-clause"})


def test_underscore_bag_tag_matches_space_form_clause_tag(pipeline_bits):
    """Comparison keys fold ``_`` to `` ``, so an underscore bag tag matches a
    space-form clause tag; the moved tag keeps the bag's own spelling."""
    from anime_tools.stages.position_captions import (
        ClauseVocabulary,
        plan_bag_removals,
    )

    vocabulary = ClauseVocabulary(tag_to_group={"speech bubble": "effect"})
    plan = plan_bag_removals(
        ("2girls", "speech_bubble"),
        [["speech bubble"], []],
        ["left", "right"],
        [{"speech bubble": 0.9}, {}],
        [{"speech bubble": 0.9}, {"speech bubble": 0.1}],
        vocabulary=vocabulary,
        margin=0.25,
    )
    assert plan.blocked == {}
    assert len(plan.moved) == 1
    assert plan.moved[0].tag == "speech_bubble"  # verbatim bag spelling
    assert plan.moved[0].position == "left"


def test_flatten_undoes_the_rewrite(pipeline_bits):
    from anime_tools.captions.position_clauses import flatten_caption

    proposal = _two_girls_proposal(pipeline_bits)
    flat = {t.strip() for t in flatten_caption(proposal.proposed).split(",")}
    # Tag set restored; order is not promised.
    assert {t.strip() for t in _TWO_GIRLS_CAPTION.split(",")} <= flat
    assert "On the" not in flatten_caption(proposal.proposed)


def test_flatten_folds_the_underscore_spelling_of_a_tag_it_returns():
    """The flatten dedupe keys on :func:`~anime_tools.captions.taxonomy.normalize_tag`,
    so a tag moved out in space form does not come back beside its underscore
    spelling."""
    from anime_tools.captions.position_clauses import flatten_caption, parse_caption

    caption = "safe, 1girl, speech_bubble, blue eyes. On the left, speech bubble."
    flat = parse_caption(flatten_caption(caption)).flat_tags
    assert list(flat) == ["safe", "1girl", "speech_bubble", "blue eyes"]
    # The bag's own spelling survives: comparison folds, output does not.
    assert "speech bubble" not in flat


def test_count_mismatch_is_skipped_not_guessed(pipeline_bits):
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 3girls, blonde hair, aqua hair",
        detect_fn=_detector(
            {0.3: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger([{"kept": {}, "groups": {}}]),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.status == "skip:count-mismatch"
    assert proposal.proposed is None


def _two_hair_colors():
    return _tagger(
        [
            {"kept": {"blonde hair": 0.8}, "groups": {"hair_color": "blonde hair"}},
            {"kept": {"aqua hair": 0.8}, "groups": {"hair_color": "aqua hair"}},
            {"kept": {"green hair": 0.8}, "groups": {"hair_color": "green hair"}},
        ]
    )


def test_a_detected_male_is_inside_the_count_range(pipeline_bits):
    # `expected` counts girls, but the `girl` prompt picks up males
    # inconsistently, so the range is girls..girls+boys.
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "sensitive, 1boy, 2girls, blonde hair, aqua hair",
        detect_fn=_detector(
            {
                0.5: [
                    ((0, 0, 300, 500), 0.9),
                    ((350, 0, 650, 500), 0.9),
                    ((700, 0, 1000, 500), 0.9),
                ]
            }
        ),
        tag_fn=_two_hair_colors(),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.ok, proposal.status
    # One boy of slack, not unlimited slack.
    over = propose_for_image(
        image,
        "sensitive, 1boy, 2girls, blonde hair, aqua hair",
        detect_fn=_detector(
            {
                0.5: [
                    ((0, 0, 240, 500), 0.9),
                    ((260, 0, 490, 500), 0.9),
                    ((510, 0, 740, 500), 0.9),
                    ((760, 0, 1000, 500), 0.9),
                ]
            }
        ),
        tag_fn=_two_hair_colors(),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert over.status == "skip:count-mismatch"


def test_retry_fires_when_the_caption_gives_no_count(pipeline_bits):
    # A `multiple views` sheet reports expected=None on purpose. Gating the
    # low-threshold retry on `if expected` would skip it for that whole
    # population.
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        # Both colors are in the bag so the identity gate stays out of the way —
        # this test is about the retry, not about clause content.
        "sensitive, 1girl, multiple views, blonde hair, aqua hair",
        detect_fn=_detector(
            {
                0.5: [((0, 0, 400, 500), 0.9)],
                0.35: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.4)],
            }
        ),
        tag_fn=_two_hair_colors(),
        vocabulary=vocabulary,
        # The multi-view gate stays off, or hair color never reaches a clause and
        # the row skips before the retry's effect can be read.
        options=Options(multi_view_gate=False),
    )
    assert proposal.ok, proposal.status
    assert proposal.detected == 2


def test_a_nested_subject_survives_dedupe():
    # A real second subject is as nested as a group box, so the rule is opt-in.
    from anime_tools.stages.position_captions import Detection, dedupe_detections

    host = Detection(box=(0, 0, 1000, 500), score=0.9)
    inside = Detection(box=(400, 200, 600, 480), score=0.6)
    assert len(dedupe_detections([host, inside], 0.65)) == 2
    # Opting in suppresses the contained box (the group-box case).
    kept = dedupe_detections([host, inside], 0.65, containment_threshold=0.8)
    assert kept == [host]
    # Plain IoU duplicates still go, either way.
    dup = Detection(box=(10, 10, 990, 490), score=0.7)
    assert dedupe_detections([host, dup], 0.65) == [host]


def test_mask_containment_separates_a_fragment_from_a_second_subject():
    # Both pairs below are box-nested to ~1.0; only the mask tells them apart.
    from anime_tools.stages.position_captions import (
        Detection,
        box_containment,
        dedupe_detections,
        mask_containment,
    )

    figure = np.zeros((100, 100), dtype=np.float32)
    figure[:, 0:50] = 1.0  # a girl occupying the left half
    fragment = np.zeros((100, 100), dtype=np.float32)
    fragment[0:30, 0:50] = 1.0  # her head — a subset of her own mask
    neighbour = np.zeros((100, 100), dtype=np.float32)
    neighbour[0:30, 50:100] = 1.0  # a second girl, same box, disjoint pixels

    host = Detection(box=(0, 0, 100, 100), score=0.9, mask=figure)
    part = Detection(box=(0, 0, 50, 30), score=0.6, mask=fragment)
    other = Detection(box=(0, 0, 100, 30), score=0.6, mask=neighbour)

    assert mask_containment(part, host) == pytest.approx(1.0)
    assert mask_containment(other, host) == pytest.approx(0.0)
    # Box geometry cannot separate them: both are fully inside the host box.
    assert box_containment(part.box, host.box) == pytest.approx(1.0)
    assert box_containment(other.box, host.box) == pytest.approx(1.0)

    assert dedupe_detections([host, part], 0.65, mask_containment_threshold=0.8) == [
        host
    ]
    kept = dedupe_detections([host, other], 0.65, mask_containment_threshold=0.8)
    assert len(kept) == 2


def test_mask_containment_falls_back_to_boxes_without_masks():
    # Stub detections and part boxes carry no mask; the rule abstains.
    from anime_tools.stages.position_captions import Detection, dedupe_detections

    host = Detection(box=(0, 0, 1000, 500), score=0.9)
    inside = Detection(box=(400, 200, 600, 480), score=0.6)
    assert (
        len(dedupe_detections([host, inside], 0.65, mask_containment_threshold=0.8))
        == 2
    )
    # Above 1.0 disables the rule outright.
    speckle = np.zeros((100, 100), dtype=np.float32)
    speckle[:, 0:50] = 1.0
    a = Detection(box=(0, 0, 100, 100), score=0.9, mask=speckle)
    b = Detection(box=(0, 0, 50, 100), score=0.6, mask=speckle)
    assert len(dedupe_detections([a, b], 0.65, mask_containment_threshold=1.01)) == 2


def test_mask_box_fill():
    from anime_tools.stages.position_captions import Detection, mask_box_fill

    assert mask_box_fill(Detection(box=(0, 0, 10, 10), score=0.5)) is None
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[0:50, 0:100] = 1.0
    det = Detection(box=(0, 0, 100, 100), score=0.5, mask=mask)
    assert mask_box_fill(det) == pytest.approx(0.5)
    # The window clips to the mask, so an out-of-range box does not blow up.
    wide = Detection(box=(-10, -10, 200, 50), score=0.5, mask=mask)
    assert mask_box_fill(wide) == pytest.approx(1.0)


def test_fill_ratio_swaps_the_survivor_inside_a_matched_pair():
    # The 5847152 shape: a near-empty duplicate outscores the clean mask by a
    # hair; the tie-break lets the clean mask survive.
    from anime_tools.stages.position_captions import Detection, dedupe_detections

    speckle = np.zeros((100, 100), dtype=np.float32)
    speckle[0:5, :] = 1.0  # fill 0.05 in its own box
    clean = np.zeros((100, 100), dtype=np.float32)
    clean[0:90, :] = 1.0  # fill 1.0 in its own box
    garbage = Detection(box=(0, 0, 100, 100), score=0.9, mask=speckle)
    good = Detection(box=(0, 0, 100, 90), score=0.87, mask=clean)  # IoU 0.9

    # Off (default): score alone picks the survivor.
    assert dedupe_detections([garbage, good], 0.65) == [garbage]
    # On: the pair is matched either way, but the far better mask wins.
    kept = dedupe_detections([garbage, good], 0.65, fill_ratio_threshold=2.0)
    assert kept == [good]
    # Count is invariant: an unrelated box is untouched.
    other = Detection(box=(200, 200, 300, 300), score=0.5, mask=clean)
    kept = dedupe_detections([garbage, good, other], 0.65, fill_ratio_threshold=2.0)
    assert len(kept) == 2 and good in kept and other in kept


def test_fill_ratio_below_threshold_keeps_score_order():
    # A benign inversion (ratio < R) must not swap.
    from anime_tools.stages.position_captions import Detection, dedupe_detections

    kept_mask = np.zeros((100, 100), dtype=np.float32)
    kept_mask[0:40, :] = 1.0  # fill 0.40
    better_mask = np.zeros((100, 100), dtype=np.float32)
    better_mask[0:50, :] = 1.0  # fill ~0.56 in a (0,0,100,90) box: ratio ~1.4
    a = Detection(box=(0, 0, 100, 100), score=0.9, mask=kept_mask)
    b = Detection(box=(0, 0, 100, 90), score=0.85, mask=better_mask)
    assert dedupe_detections([a, b], 0.65, fill_ratio_threshold=2.0) == [a]


def test_fill_ratio_falls_back_to_score_when_a_mask_is_missing():
    from anime_tools.stages.position_captions import Detection, dedupe_detections

    clean = np.ones((100, 100), dtype=np.float32)
    a = Detection(box=(0, 0, 100, 100), score=0.9)  # no mask (stub / part box)
    b = Detection(box=(0, 0, 100, 90), score=0.85, mask=clean)
    assert dedupe_detections([a, b], 0.65, fill_ratio_threshold=2.0) == [a]
    c = Detection(box=(0, 0, 100, 90), score=0.85)  # candidate lacks the mask
    d = Detection(box=(0, 0, 100, 100), score=0.9, mask=clean)
    assert dedupe_detections([d, c], 0.65, fill_ratio_threshold=2.0) == [d]


def test_an_inset_is_too_small_to_be_a_subject(pipeline_bits):
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "sensitive, 2girls, blonde hair, aqua hair",
        detect_fn=_detector(
            {
                0.5: [
                    ((0, 0, 400, 500), 0.9),
                    # 0.4% of the canvas, and disjoint: containment misses it.
                    ((900, 400, 950, 440), 0.8),
                ]
            }
        ),
        tag_fn=_two_hair_colors(),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.status == "skip:too-few-instances"
    assert proposal.detected == 1


def test_open_ended_crowd_counts_defer_to_detection():
    # `6+girls` is "six or more", not six.
    from anime_tools.stages.position_captions import (
        caption_boy_count,
        caption_subject_count,
    )

    assert caption_subject_count("safe, 6+girls, cheering") is None
    assert caption_subject_count("safe, 2girls, blonde hair") == 2
    assert caption_boy_count("safe, 1boy, 2girls") == 1
    assert caption_boy_count("safe, 2girls, blonde hair") == 0
    assert caption_boy_count("safe, multiple boys, 2girls") is None
    assert caption_boy_count("safe, 3+boys, 2girls") is None


def test_comic_panels_defer_the_count_to_detection():
    """A comic page draws the same girl once per panel — 1girl, 2 subjects."""
    from anime_tools.stages.position_captions import (
        caption_subject_count,
        is_candidate,
    )

    for layout in ("comic", "2koma", "4koma", "silent comic", "sequential"):
        caption = f"safe, 1girl, {layout}, blonde hair"
        assert caption_subject_count(caption) is None, layout
        assert is_candidate(caption) == (True, "panel-layout"), layout
    # A girls-count alongside the layout tag does not override it.
    assert caption_subject_count("safe, 1girl, 2koma, blonde hair") is None


def test_koma_count_bounds_a_page_that_has_no_girls_count_check():
    """``Nkoma`` names the panel count, restoring the ceiling the layout waived."""
    from anime_tools.stages.position_captions import caption_panel_ceiling

    assert caption_panel_ceiling("safe, 1girl, 2koma") == 2
    assert caption_panel_ceiling("safe, 2girls, 1boy, 2koma") == 6
    assert caption_panel_ceiling("safe, 4koma") == 4  # nobody counted, 1/panel
    # Unbounded layouts stay unbounded: no panel count to multiply.
    assert caption_panel_ceiling("safe, 1girl, comic") is None
    assert caption_panel_ceiling("safe, 1girl, multiple views") is None
    assert caption_panel_ceiling("safe, 1girl, multiple 4koma") is None
    # An open-ended character count cannot produce a ceiling either.
    assert caption_panel_ceiling("safe, multiple girls, 2koma") is None
    assert caption_panel_ceiling("safe, 1girl, multiple boys, 2koma") is None


def test_a_koma_page_over_its_ceiling_is_skipped(pipeline_bits):
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 1girl, 2koma, blonde hair",  # ceiling 2
        detect_fn=_detector(
            {
                0.5: [
                    ((0, 0, 300, 500), 0.9),
                    ((320, 0, 620, 500), 0.9),
                    ((640, 0, 940, 500), 0.9),
                ]
            }
        ),
        tag_fn=_two_hair_colors(),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.status == "skip:count-mismatch"
    assert proposal.detected == 3


def test_a_koma_page_at_its_ceiling_proposes(pipeline_bits):
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 1girl, 2koma, blonde hair, aqua hair",  # ceiling 2
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_two_hair_colors(),
        vocabulary=vocabulary,
        # The multi-view gate is off so the koma page still has clause content.
        options=Options(multi_view_gate=False),
    )
    assert proposal.ok, proposal.status
    assert proposal.detected == 2


def test_page_number_is_not_a_layout_tag():
    """``page number`` is a margin annotation, not a panel layout."""
    from anime_tools.stages.position_captions import (
        caption_subject_count,
        is_candidate,
    )

    caption = "safe, 1girl, page number, blonde hair"
    assert caption_subject_count(caption) == 1
    assert is_candidate(caption) == (False, "single-subject")


def test_low_threshold_retry_recovers_the_missing_instance(pipeline_bits):
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    detect = _detector(
        {
            0.5: [((0, 0, 300, 500), 0.9), ((350, 0, 650, 500), 0.9)],
            0.3: [
                ((0, 0, 300, 500), 0.9),
                ((350, 0, 650, 500), 0.9),
                ((700, 0, 1000, 500), 0.35),
            ],
        }
    )
    proposal = propose_for_image(
        image,
        "safe, 3girls, blonde hair, aqua hair, green hair",
        detect_fn=detect,
        tag_fn=_tagger(
            [
                {"kept": {"blonde hair": 0.8}, "groups": {}},
                {"kept": {"aqua hair": 0.8}, "groups": {}},
                {"kept": {"green hair": 0.8}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.detected == 3
    assert [c.position for c in parse_caption(proposal.proposed).clauses] == [
        "left",
        "middle",
        "right",
    ]


def _part_detector(boxes_by_prompt):
    """Stub ``part_detect_fn``: prompt → list of (box, score)."""
    from anime_tools.stages.position_captions import Detection

    def detect(image, prompt, score_threshold):
        return [
            Detection(box=b, score=s, source=prompt)
            for b, s in boxes_by_prompt.get(prompt, ())
            if s >= score_threshold
        ]

    return detect


def test_part_prompt_recovers_a_headless_panel(pipeline_bits):
    """One full body plus a headless close-up panel."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 1girl, multiple views, blonde hair",
        detect_fn=_detector({0.5: [((700, 0, 1000, 500), 0.9)]}),
        part_detect_fn=_part_detector({"buttocks": [((0, 0, 600, 500), 0.8)]}),
        # Crops are tagged in reading order, so the recovered left panel first.
        tag_fn=_tagger(
            [
                {"kept": {"blonde hair": 0.8, "ass": 0.7}, "groups": {}},
                {"kept": {"blonde hair": 0.8, "twintails": 0.7}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        # Multi-view gate off: this is about the second grounding pass and the
        # reading order it lands in.
        options=Options(part_prompts=("buttocks",), multi_view_gate=False),
    )
    assert proposal.ok, proposal.status
    assert proposal.detected == 2
    # ``detections`` is merged (subjects first); ``instances`` is in reading order.
    assert [d["source"] for d in proposal.detections] == ["subject", "buttocks"]
    assert [(i.position, i.source) for i in proposal.instances] == [
        ("left", "buttocks"),
        ("right", "subject"),
    ]
    assert [c.position for c in parse_caption(proposal.proposed).clauses] == [
        "left",
        "right",
    ]


def test_part_prompts_are_inert_when_the_subject_prompt_suffices(pipeline_bits):
    """No undershoot means no second grounding pass."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    calls = []

    def part_detect(image, prompt, score_threshold):
        calls.append(prompt)
        return []

    proposal = propose_for_image(
        image,
        "safe, 2girls, blonde hair, aqua hair",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        part_detect_fn=part_detect,
        tag_fn=_tagger(
            [
                {"kept": {"blonde hair": 0.8}, "groups": {}},
                {"kept": {"aqua hair": 0.8}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(part_prompts=("buttocks", "hips")),
    )
    assert proposal.ok
    assert calls == []
    assert proposal.detected == 2


def test_part_crop_carries_no_hair_or_eye_color(pipeline_bits):
    """A headless crop carries no identity tags, however the tagger guesses."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 1girl, multiple views, blonde hair",
        detect_fn=_detector({0.5: [((700, 0, 1000, 500), 0.9)]}),
        part_detect_fn=_part_detector({"buttocks": [((0, 0, 600, 500), 0.8)]}),
        # Crops are tagged in reading order: recovered left panel, then full body.
        tag_fn=_tagger(
            [
                # Part crop: the tagger invents a hair and eye color.
                {
                    "kept": {
                        "green hair": 0.8,
                        "red eyes": 0.7,
                        "twintails": 0.6,
                        "ass": 0.9,
                    },
                    "groups": {"hair_color": "green hair", "eye_color": "red eyes"},
                },
                # Full body: real hair color, and it is in the caption.
                {"kept": {"blonde hair": 0.8}, "groups": {"hair_color": "blonde hair"}},
            ]
        ),
        vocabulary=vocabulary,
        # ``allow_identity`` is a per-crop rule; the multi-view gate is off so it
        # is observable on its own.
        options=Options(part_prompts=("buttocks",), multi_view_gate=False),
    )
    assert proposal.ok, proposal.status
    part = next(i for i in proposal.instances if i.source == "buttocks")
    assert "green hair" not in part.tags
    assert "red eyes" not in part.tags
    assert "twintails" not in part.tags  # hairstyle is identity too
    assert part.tags == ["ass"]  # only what a headless crop can show


def test_clause_cannot_contradict_a_hair_color_the_caption_named(pipeline_bits):
    """The flat bag outranks the crop tagger on identity groups."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    # One girl, two views. The caption says blonde; the back view guesses green.
    tags = _tagger(
        [
            {"kept": {"blonde hair": 0.8}, "groups": {"hair_color": "blonde hair"}},
            {
                "kept": {"green hair": 0.8, "red eyes": 0.7},
                "groups": {"hair_color": "green hair", "eye_color": "red eyes"},
            },
        ]
    )
    proposal = propose_for_image(
        image,
        "safe, 1girl, multiple views, blonde hair",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=tags,
        vocabulary=vocabulary,
        # The multi-view gate is the stricter layer above and is off here.
        options=Options(multi_view_gate=False),
    )
    assert proposal.ok, proposal.status
    clauses = parse_caption(proposal.proposed).clauses
    assert not any("green hair" in c.tags for c in clauses)
    # ``red eyes`` survives: the caption named no eye color, so it is new
    # information rather than a contradiction.
    assert any("red eyes" in c.tags for c in clauses)


def test_bag_gate_still_allows_every_color_the_caption_listed(pipeline_bits):
    """A real 2girls image names both colors, so neither clause is blocked."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, blonde hair, aqua hair",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {"kept": {"blonde hair": 0.8}, "groups": {"hair_color": "blonde hair"}},
                {"kept": {"aqua hair": 0.8}, "groups": {"hair_color": "aqua hair"}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.ok
    clauses = parse_caption(proposal.proposed).clauses
    assert [c.tags[0] for c in clauses] == ["blonde hair", "aqua hair"]


_CROWDED_CAPTION = (
    "safe, 2girls, akita neru, hatsune miku, blonde hair, aqua hair, maid, ass"
)


def _crowded_proposal(pipeline_bits, **option_overrides):
    """A crop the tagger is far more talkative about than the caption: four kept
    tags are in the bag, five are not."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    return propose_for_image(
        image,
        _CROWDED_CAPTION,
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {
                    "kept": {
                        "akita neru": 0.9,
                        "blonde hair": 0.8,
                        "maid": 0.7,
                        "ass": 0.6,
                        # Nothing in the caption claims any of these.
                        "red eyes": 0.75,
                        "long hair": 0.95,
                        "large breasts": 0.9,
                        "thighs": 0.85,
                        "playboy bunny": 0.5,
                    },
                    "groups": {"hair_color": "blonde hair"},
                },
                {
                    "kept": {"hatsune miku": 0.9, "aqua hair": 0.8},
                    "groups": {"hair_color": "aqua hair"},
                },
            ]
        ),
        vocabulary=vocabulary,
        options=Options(**option_overrides),
    )


def test_a_clause_fills_from_the_bag_before_inventing(pipeline_bits):
    """Every bag tag this crop kept is bound and exactly one novel tag rides along;
    ordering is the ranking (name → hair → eyes → the rest)."""
    proposal = _crowded_proposal(pipeline_bits)
    assert proposal.ok, proposal.status
    clause = parse_caption(proposal.proposed).clauses[0]
    assert clause.tags == ("akita neru", "blonde hair", "red eyes", "maid", "ass")
    assert proposal.instances[0].novel == 1


def test_max_novel_tags_zero_never_invents(pipeline_bits):
    proposal = _crowded_proposal(pipeline_bits, max_novel_tags=0)
    bag = set(parse_caption(_CROWDED_CAPTION).flat_tags)
    bound = {t for c in parse_caption(proposal.proposed).clauses for t in c.tags}
    assert bound <= bag
    assert all(inst.novel == 0 for inst in proposal.instances)


def test_the_bag_blind_budget_is_still_reachable(pipeline_bits):
    """``max_novel_tags == max_clause_tags`` is the bag-blind A/B arm."""
    proposal = _crowded_proposal(pipeline_bits, max_novel_tags=8)
    clause = parse_caption(proposal.proposed).clauses[0]
    assert len(clause.tags) == 8
    assert {"large breasts", "thighs", "long hair"} <= set(clause.tags)


def test_the_novel_budget_never_costs_a_move(pipeline_bits):
    """Tightening the budget removes padding, never a binding: the moved set is
    equal in both directions."""
    budgeted = _crowded_proposal(pipeline_bits)
    bag_blind = _crowded_proposal(pipeline_bits, max_novel_tags=8)
    assert {m["tag"] for m in budgeted.moved} == {m["tag"] for m in bag_blind.moved}
    assert {"maid", "ass"} <= {m["tag"] for m in budgeted.moved}
    # …and the padding really is gone.
    assert budgeted.instances[0].novel == 1
    assert bag_blind.instances[0].novel > 1


def test_the_bag_gate_covers_every_exclusive_group_not_just_identity(pipeline_bits):
    """An exclusive group holds one value, so a crop naming a second one (here
    ``large breasts`` against the bag's ``flat chest``) is a contradiction."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits

    def run(**overrides):
        return propose_for_image(
            image,
            "safe, 2girls, flat chest, blonde hair, aqua hair",
            detect_fn=_detector(
                {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
            ),
            tag_fn=_tagger(
                [
                    {
                        "kept": {"blonde hair": 0.8, "large breasts": 0.9},
                        "groups": {"hair_color": "blonde hair"},
                    },
                    {"kept": {"aqua hair": 0.8}, "groups": {"hair_color": "aqua hair"}},
                ]
            ),
            vocabulary=vocabulary,
            options=Options(**overrides),
        )

    gated = run()
    assert gated.ok, gated.status
    assert not any(
        "large breasts" in c.tags for c in parse_caption(gated.proposed).clauses
    )
    # The novel budget is not what blocked it: with the gate off it takes the
    # same free slot.
    ungated = run(bag_gated_identity=False)
    assert any(
        "large breasts" in c.tags for c in parse_caption(ungated.proposed).clauses
    )


def test_a_kept_bag_value_outranks_the_softmax_winner(pipeline_bits):
    """The crop's softmax winner (``green hair``) is gate-rejected, so the group
    falls back to the kept bag value rather than emptying."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, blonde hair, aqua hair",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {
                    "kept": {"blonde hair": 0.7, "green hair": 0.9},
                    "groups": {"hair_color": "green hair"},
                },
                {"kept": {"aqua hair": 0.8}, "groups": {"hair_color": "aqua hair"}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.ok, proposal.status
    clauses = parse_caption(proposal.proposed).clauses
    assert clauses[0].tags[0] == "blonde hair"
    assert not any("green hair" in c.tags for c in clauses)


def test_part_top_up_stops_at_the_target(pipeline_bits):
    """A fragmenting part prompt binds no more clauses than the target needs."""
    from anime_tools.stages.position_captions import detect_subjects

    image, _, _Detection, Options = pipeline_bits
    parts = {
        "thighs": [
            ((0, 0, 300, 240), 0.67),
            ((0, 260, 300, 500), 0.65),
            ((320, 0, 620, 240), 0.63),
            ((320, 260, 620, 500), 0.61),
        ]
    }
    dets = detect_subjects(
        image,
        _detector({0.5: [((700, 0, 1000, 500), 0.9)]}),
        Options(part_prompts=("thighs",)),
        None,  # ``multiple views`` → expected unknown → target is min_instances
        _part_detector(parts),
    )
    assert len(dets) == 2
    assert [d.source for d in dets] == ["subject", "thighs"]
    assert dets[1].score == 0.67  # the highest-scoring part box, not the first


def test_part_box_nested_in_a_subject_is_dropped(pipeline_bits):
    """A part nested in a subject is that subject's body, not a new position."""
    from anime_tools.stages.position_captions import merge_part_detections

    _, _, Detection, _ = pipeline_bits
    subjects = [Detection(box=(0, 0, 400, 500), score=0.9)]
    parts = [
        Detection(box=(50, 200, 350, 480), score=0.8, source="buttocks"),  # nested
        Detection(box=(600, 0, 1000, 500), score=0.7, source="hips"),  # its own panel
    ]
    merged = merge_part_detections(
        subjects, parts, iou_threshold=0.65, containment_threshold=0.7
    )
    assert [d.source for d in merged] == ["subject", "hips"]


def test_unlisted_character_name_is_rejected(pipeline_bits):
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, blonde hair, aqua hair",  # no character named
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {"kept": {"akita neru": 0.99, "blonde hair": 0.8}, "groups": {}},
                {"kept": {"hatsune miku": 0.99, "aqua hair": 0.8}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    clauses = parse_caption(proposal.proposed).clauses
    assert all("akita neru" not in c.tags for c in clauses)
    assert clauses[0].tags == ("blonde hair",)


def test_multiple_views_clauses_carry_only_what_differs(pipeline_bits):
    # A `1girl, multiple views` outfit sheet: only the outfit varies per view.
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    shared = {"hatsune miku": 0.99, "aqua hair": 0.9, "twintails": 0.8}
    proposal = propose_for_image(
        image,
        "safe, 1girl, multiple views, hatsune miku, aqua hair, twintails, maid, swimsuit",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {
                    "kept": {**shared, "maid": 0.8},
                    "groups": {"hair_color": "aqua hair"},
                },
                {
                    "kept": {**shared, "swimsuit": 0.8},
                    "groups": {"hair_color": "aqua hair"},
                },
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    clauses = parse_caption(proposal.proposed).clauses
    assert clauses[0].tags == ("maid",)
    assert clauses[1].tags == ("swimsuit",)
    # ...and the shared attributes are still asserted in the flat bag.
    assert "hatsune miku" in parse_caption(proposal.proposed).flat_tags


def test_keep_shared_tags_restores_the_repeated_attributes(pipeline_bits):
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    shared = {"hatsune miku": 0.99, "aqua hair": 0.9}
    proposal = propose_for_image(
        image,
        "safe, 1girl, multiple views, hatsune miku, aqua hair, maid, swimsuit",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {"kept": {**shared, "maid": 0.8}, "groups": {}},
                {"kept": {**shared, "swimsuit": 0.8}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        # ``--keep_shared_tags`` and the multi-view gate suppress overlapping
        # sets; the gate is off so this flag's own effect is what is measured.
        options=Options(discriminative_only=False, multi_view_gate=False),
    )
    assert "hatsune miku" in parse_caption(proposal.proposed).clauses[0].tags


# ----- the multi-view gate ------------------------------------------------


def _layout_proposal(pipeline_bits, caption, per_crop, **option_overrides):
    """Two subjects on a repeated-subject page: same girl, drawn twice."""
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    return propose_for_image(
        image,
        caption,
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(per_crop),
        vocabulary=vocabulary,
        options=Options(**option_overrides),
    )


# The crop tagger disagreeing with itself across two views of one girl: each tag
# reaches only one crop, so it reads as "attributable" when it is a miss.
_DISAGREEING_VIEWS = [
    {
        "kept": {
            "hatsune miku": 0.9,  # name the other crop dropped
            "aqua hair": 0.9,  # appearance
            "large breasts": 0.8,  # body shape
            "ass": 0.7,  # anatomy
            "maid": 0.8,  # outfit — the only real per-view difference
        },
        "groups": {"hair_color": "aqua hair"},
    },
    {
        "kept": {"twintails": 0.8, "long hair": 0.7, "playboy bunny": 0.8},
        "groups": {},
    },
]

_LAYOUT_CAPTION = (
    "safe, 1girl, {layout}, hatsune miku, aqua hair, twintails, long hair, "
    "large breasts, ass, maid, playboy bunny"
)


@pytest.mark.parametrize("layout", ["multiple views", "2koma", "comic"])
def test_a_repeated_subject_page_binds_only_what_a_view_can_differ_in(
    pipeline_bits, layout
):
    """One girl drawn twice: her name and traits belong to her, not a view. Every
    ``_LAYOUT_TAGS`` member takes the gate, not just ``multiple views``.
    """
    proposal = _layout_proposal(
        pipeline_bits, _LAYOUT_CAPTION.format(layout=layout), _DISAGREEING_VIEWS
    )
    assert proposal.ok, proposal.status
    parsed = parse_caption(proposal.proposed)
    bound = {t for c in parsed.clauses for t in c.tags}
    # Nothing the character owns: name, hair, eyes, body shape.
    assert bound.isdisjoint(
        {"hatsune miku", "aqua hair", "twintails", "long hair", "large breasts"}
    )
    # …only what one view has and the other does not: the outfits and the anatomy
    # visible in one panel (`body_parts` is gated by `--gate_view_anatomy`).
    assert bound == {"maid", "playboy bunny", "ass"}
    # A tag that never reached a clause cannot leave the bag, so every suppressed
    # trait is still asserted, flat.
    assert {
        "hatsune miku",
        "aqua hair",
        "twintails",
        "long hair",
        "large breasts",
    } <= set(parsed.flat_tags)


def test_the_gate_is_off_for_a_genuine_multi_character_image(pipeline_bits):
    """No layout tag means two different girls, so identity binds."""
    proposal = _layout_proposal(
        pipeline_bits,
        "safe, 2girls, hatsune miku, aqua hair, large breasts, ass, maid, "
        "playboy bunny",
        _DISAGREEING_VIEWS,
    )
    assert proposal.ok, proposal.status
    bound = {t for c in parse_caption(proposal.proposed).clauses for t in c.tags}
    assert {"hatsune miku", "aqua hair", "large breasts", "ass"} <= bound


def test_bind_view_traits_reverts_the_gate(pipeline_bits):
    """``--bind_view_traits`` is the A/B arm that reverts the gate."""
    proposal = _layout_proposal(
        pipeline_bits,
        _LAYOUT_CAPTION.format(layout="multiple views"),
        _DISAGREEING_VIEWS,
        multi_view_gate=False,
    )
    bound = {t for c in parse_caption(proposal.proposed).clauses for t in c.tags}
    assert {"hatsune miku", "aqua hair", "large breasts", "ass"} <= bound


def test_an_excluded_tag_cannot_ride_the_priority_path_into_a_clause(pipeline_bits):
    """``excluded`` is consulted on every path, including the exclusive-group step
    a deprecated alias like ``light brown hair`` enters through."""
    from anime_tools.stages.position_captions import ClauseVocabulary

    _, _, _, _Options = pipeline_bits
    vocabulary = ClauseVocabulary(
        excluded=frozenset({"vocaloid", "light brown hair"}),
        exclusive_groups=frozenset({"hair_color"}),
        tag_to_group={
            "light brown hair": "hair_color",
            "aqua hair": "hair_color",
            "maid": "costume",
        },
    )
    tags = vocabulary.select(
        {"light brown hair": 0.9, "vocaloid": 0.9, "maid": 0.8},
        {"hair_color": "light brown hair"},
        flat_bag=frozenset({"light brown hair", "vocaloid", "maid"}),
        attributable=frozenset({"light brown hair", "maid"}),
        shared=frozenset(),
        max_tags=8,
        name_confidence=0.5,
        allow_unlisted_names=False,
    )
    assert tags == ["maid"]


def test_indistinguishable_subjects_are_skipped(pipeline_bits):
    # Nothing varies between the crops, so no clause can be grounded.
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    same = {"hatsune miku": 0.99, "aqua hair": 0.9, "twintails": 0.8}
    proposal = propose_for_image(
        image,
        "safe, 2girls, hatsune miku, aqua hair, twintails",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger([{"kept": same, "groups": {}}, {"kept": same, "groups": {}}]),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.status == "skip:no-discriminative-tags"
    assert proposal.proposed is None


def test_differing_hair_colors_still_bind_for_two_characters(pipeline_bits):
    # Two girls with different hair keep their hair in their clause.
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, akita neru, hatsune miku, blonde hair, aqua hair",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {
                    "kept": {"akita neru": 0.9, "blonde hair": 0.8},
                    "groups": {"hair_color": "blonde hair"},
                },
                {
                    "kept": {"hatsune miku": 0.9, "aqua hair": 0.8},
                    "groups": {"hair_color": "aqua hair"},
                },
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    clauses = parse_caption(proposal.proposed).clauses
    assert clauses[0].tags == ("akita neru", "blonde hair")
    assert clauses[1].tags == ("hatsune miku", "aqua hair")


def test_copyright_tags_stay_out_of_clauses(pipeline_bits):
    # A franchise tag describes the image, not a subject.
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, vocaloid, blonde hair, aqua hair",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {"kept": {"blonde hair": 0.8, "vocaloid": 0.95}, "groups": {}},
                {"kept": {"aqua hair": 0.8, "twintails": 0.7}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert all(
        "vocaloid" not in c.tags for c in parse_caption(proposal.proposed).clauses
    )


def test_only_one_member_of_an_exclusive_group_per_clause(pipeline_bits):
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, green hair, aqua hair, blonde hair",
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                # Two hair colors on one crop: the group winner wins and the
                # runner-up does not follow it in through the ranked path.
                {
                    "kept": {"green hair": 0.9, "aqua hair": 0.85, "twintails": 0.7},
                    "groups": {"hair_color": "green hair"},
                },
                {"kept": {"blonde hair": 0.8}, "groups": {}},
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
    )
    left = parse_caption(proposal.proposed).clauses[0].tags
    assert "green hair" in left
    assert "aqua hair" not in left


def test_single_detection_is_skipped(pipeline_bits):
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        "safe, 2girls, blonde hair",
        detect_fn=_detector({0.3: [((0, 0, 400, 500), 0.9)]}),
        tag_fn=_tagger([{"kept": {}, "groups": {}}]),
        vocabulary=vocabulary,
        options=Options(),
    )
    assert proposal.status == "skip:too-few-instances"


# ----- candidate prefilter ------------------------------------------------


@pytest.mark.parametrize(
    "caption,expected",
    [
        ("safe, 3girls, blonde hair", True),
        ("safe, 1girl, multiple views, maid", True),
        ("safe, multiple girls, blonde hair", True),
        ("safe, 1girl, blonde hair", False),
        (GT, False),  # already has clauses
    ],
)
def test_is_candidate(caption, expected):
    from anime_tools.stages.position_captions import is_candidate

    assert is_candidate(caption)[0] is expected


def test_multiple_views_defers_the_count_to_detection():
    # `1girl, multiple views` is routinely four bindable views, so the girls-count
    # must not gate it.
    from anime_tools.stages.position_captions import caption_subject_count

    assert caption_subject_count("safe, 1girl, multiple views, maid") is None
    assert caption_subject_count("safe, 3girls, blonde hair") == 3
    assert caption_subject_count("safe, 1girl, blonde hair") == 1


def test_mask_blanking_removes_the_neighbour(pipeline_bits):
    import numpy as np
    from PIL import Image

    from anime_tools.stages.position_captions import Detection, crop_instance

    # Left half red (the subject), right half blue (a neighbour in the padded box).
    pixels = np.zeros((100, 200, 3), dtype=np.uint8)
    pixels[:, :100] = (255, 0, 0)
    pixels[:, 100:] = (0, 0, 255)
    image = Image.fromarray(pixels)
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[:, :100] = 1

    crop = np.asarray(
        crop_instance(
            image, Detection(box=(0, 0, 200, 100), score=0.9, mask=mask), pad=0.0
        )
    )
    assert not (crop == (0, 0, 255)).all(axis=-1).any()
    assert (crop == (255, 0, 0)).all(axis=-1).any()


# ----- where the rewrite lands --------------------------------------------


def _corpus(tmp_path, caption):
    """A one-image master + resized pair, ready for ``run_position_captions``."""
    from PIL import Image

    src, dst = tmp_path / "image_dataset", tmp_path / "resized"
    (src / "artistA").mkdir(parents=True)
    (dst / "artistA").mkdir(parents=True)
    Image.new("RGB", (1000, 500), "white").save(dst / "artistA" / "a.png")
    (src / "artistA" / "a.txt").write_text(caption, encoding="utf-8")
    return src, dst


def _run_io(pipeline_bits, src, dst, **kwargs):
    from anime_tools.stages.position_captions import run_position_captions

    _, vocabulary, _, Options = pipeline_bits
    return run_position_captions(
        resized_dir=dst,
        source_dir=src,
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            [
                {
                    "kept": {"akita neru": 0.9, "blonde hair": 0.8},
                    "groups": {"hair_color": "blonde hair"},
                },
                {
                    "kept": {"hatsune miku": 0.9, "aqua hair": 0.8},
                    "groups": {"hair_color": "aqua hair"},
                },
            ]
        ),
        vocabulary=vocabulary,
        options=Options(),
        **kwargs,
    )


def test_apply_writes_the_resized_caption_and_never_the_master(pipeline_bits, tmp_path):
    """Clauses are generated data: the hand-written master stays byte-identical."""
    src, dst = _corpus(tmp_path, _TWO_GIRLS_CAPTION)

    _rows, stats = _run_io(pipeline_bits, src, dst, apply=True)

    assert stats.written == 1
    assert (src / "artistA" / "a.txt").read_text(encoding="utf-8") == _TWO_GIRLS_CAPTION
    written = (dst / "artistA" / "a.txt").read_text(encoding="utf-8")
    assert has_clauses(written)
    assert "On the left" in written


def test_dry_run_writes_nothing_at_all(pipeline_bits, tmp_path):
    src, dst = _corpus(tmp_path, _TWO_GIRLS_CAPTION)

    _rows, stats = _run_io(pipeline_bits, src, dst)

    assert stats.proposed == 1 and stats.written == 0
    assert not (dst / "artistA" / "a.txt").exists()


def test_apply_drops_the_stale_variant_sidecar(pipeline_bits, tmp_path):
    """The sidecar wins at encode time, so a pre-clause one is dropped."""
    from anime_tools.captions.variants import variants_sidecar_path

    src, dst = _corpus(tmp_path, _TWO_GIRLS_CAPTION)
    sidecar = variants_sidecar_path(dst / "artistA" / "a.png")
    sidecar.write_text(f"# stale\nv0\t{_TWO_GIRLS_CAPTION}\n", encoding="utf-8")

    _run_io(pipeline_bits, src, dst, apply=True)

    assert not sidecar.exists()


def test_a_second_pass_reads_the_derived_caption_and_skips_it(pipeline_bits, tmp_path):
    """Idempotent: the clauses it wrote are what `is_candidate` sees next run."""
    src, dst = _corpus(tmp_path, _TWO_GIRLS_CAPTION)
    _run_io(pipeline_bits, src, dst, apply=True)
    first = (dst / "artistA" / "a.txt").read_text(encoding="utf-8")

    _rows, stats = _run_io(pipeline_bits, src, dst, apply=True)

    assert stats.written == 0
    assert stats.skipped.get("already-has-clauses") == 1
    assert (dst / "artistA" / "a.txt").read_text(encoding="utf-8") == first


def test_flatten_backs_out_the_derived_caption_only(pipeline_bits, tmp_path):
    from anime_tools.stages.position_captions import flatten_captions

    src, dst = _corpus(tmp_path, _TWO_GIRLS_CAPTION)
    _run_io(pipeline_bits, src, dst, apply=True)

    _rows, stats = flatten_captions(resized_dir=dst, source_dir=src, apply=True)

    assert stats.written == 1
    assert not has_clauses((dst / "artistA" / "a.txt").read_text(encoding="utf-8"))
    assert (src / "artistA" / "a.txt").read_text(encoding="utf-8") == _TWO_GIRLS_CAPTION


# ----- framing: which view a clause is describing --------------------------


def test_framing_binds_to_the_view_it_describes(pipeline_bits):
    """A headless close-up panel says so in its own clause: `framing` is the only
    group that separates it from a full-body view."""
    _, vocabulary, _, _ = pipeline_bits
    tags = vocabulary.select(
        {"ass focus": 0.77, "underwear": 0.8, "denim": 0.6},
        {"framing": "ass focus"},
        flat_bag=frozenset({"full body", "underwear", "denim", "ass"}),
        attributable=frozenset({"ass focus", "underwear"}),
        shared=frozenset(),
        max_tags=8,
        name_confidence=0.5,
        allow_unlisted_names=False,
        max_novel_tags=1,
    )
    assert "ass focus" in tags


def test_the_bag_gate_does_not_claim_framing(pipeline_bits):
    """`framing` is exclusive, so it would land in the derived bag gate; the
    exemption keeps the bag's `full body` (true of the other panel) from
    pinning every clause.
    """
    _, vocabulary, _, _ = pipeline_bits
    gated = vocabulary.gated_groups()
    assert "framing" not in gated
    # …while real subject attributes stay gated: one girl has one body shape.
    assert "body_shape" in gated


def test_page_level_framing_never_binds_to_a_view(pipeline_bits):
    """`solo focus` / `white border` are filed under `framing` but describe the
    page, and v2 moves a bound tag out of the bag, so they must not bind."""
    _, vocabulary, _, _ = pipeline_bits
    for tag in ("solo focus", "white border"):
        assert vocabulary.is_scene_tag(tag)
        assert not vocabulary.is_subject_tag(tag)
    tags = vocabulary.select(
        {"solo focus": 0.9, "white border": 0.9, "maid": 0.8},
        {"framing": "solo focus"},
        flat_bag=frozenset({"solo focus", "white border", "maid"}),
        attributable=frozenset({"solo focus", "white border", "maid"}),
        shared=frozenset(),
        max_tags=8,
        name_confidence=0.5,
        allow_unlisted_names=False,
    )
    assert tags == ["maid"]


def test_framing_survives_the_multi_view_gate(pipeline_bits):
    """`view_invariant` drops what the character owns; how a panel is cropped is
    owned by the panel, so framing survives."""
    _, vocabulary, _, _ = pipeline_bits
    tags = vocabulary.select(
        {"ass focus": 0.77, "aqua hair": 0.95, "ass": 0.9},
        {"framing": "ass focus", "hair_color": "aqua hair"},
        flat_bag=frozenset({"aqua hair", "ass", "full body"}),
        attributable=frozenset({"ass focus"}),
        shared=frozenset(),
        max_tags=8,
        name_confidence=0.5,
        allow_unlisted_names=False,
        view_invariant=True,
        max_novel_tags=1,
    )
    # Hair belongs to the character and is gated; framing and the visible anatomy
    # are facts about the view.
    assert tags == ["ass focus", "ass"]


def test_two_panels_of_the_same_kind_still_share_their_framing(pipeline_bits):
    """Known limitation: a sheet whose views are both backside crops has
    `ass focus` on both, so `discriminative_only` suppresses it on both."""
    _, vocabulary, _, _ = pipeline_bits
    tags = vocabulary.select(
        {"ass focus": 0.6, "denim": 0.7},
        {"framing": "ass focus"},
        flat_bag=frozenset({"denim", "ass"}),
        attributable=frozenset({"denim"}),
        shared=frozenset({"ass focus"}),
        max_tags=8,
        name_confidence=0.5,
        allow_unlisted_names=False,
        max_novel_tags=1,
    )
    assert "ass focus" not in tags


def test_no_framing_restores_the_pre_change_clause(pipeline_bits):
    """`--no_framing` is the A side: same call, framing gone, nothing else moves."""
    _, vocabulary, _, _ = pipeline_bits
    kwargs = {
        "flat_bag": frozenset({"full body", "underwear", "denim"}),
        "attributable": frozenset({"ass focus", "underwear"}),
        "shared": frozenset(),
        "max_tags": 8,
        "name_confidence": 0.5,
        "allow_unlisted_names": False,
        "max_novel_tags": 1,
    }
    args = (
        {"ass focus": 0.77, "underwear": 0.8, "denim": 0.6},
        {"framing": "ass focus"},
    )
    on = vocabulary.select(*args, bind_framing=True, **kwargs)
    off = vocabulary.select(*args, bind_framing=False, **kwargs)
    assert on[0] == "ass focus"
    assert "ass focus" not in off
    assert off == [t for t in on if t != "ass focus"]


def test_visible_anatomy_binds_on_a_view_layout(pipeline_bits):
    """What anatomy is *visible* is a fact about the panel, so `ass` separates a
    from-behind panel from a front one even under the view gate."""
    proposal = _layout_proposal(
        pipeline_bits,
        _LAYOUT_CAPTION.format(layout="multiple views"),
        _DISAGREEING_VIEWS,
    )
    bound = {t for c in parse_caption(proposal.proposed).clauses for t in c.tags}
    assert "ass" in bound
    # …and the character's own traits are still gated.
    assert bound.isdisjoint({"hatsune miku", "aqua hair", "large breasts"})


def test_gate_view_anatomy_restores_the_old_gate(pipeline_bits):
    """The A side of the anatomy A/B: `body_parts` back in the invariant set."""
    proposal = _layout_proposal(
        pipeline_bits,
        _LAYOUT_CAPTION.format(layout="multiple views"),
        _DISAGREEING_VIEWS,
        bind_view_anatomy=False,
    )
    parsed = parse_caption(proposal.proposed)
    bound = {t for c in parsed.clauses for t in c.tags}
    assert "ass" not in bound
    assert "ass" in set(parsed.flat_tags)  # suppressed, never destroyed


# ----- the clause policy is configuration, not code ------------------------


def test_shipped_policy_is_internally_consistent():
    """``configs/clause_vocabulary.yaml`` still says what the pipeline assumes;
    each coupling below fails silently if the YAML drifts."""
    from anime_tools.captions.clause_vocabulary import default_clause_groups

    cfg = default_clause_groups()
    assert set(cfg.priority) <= cfg.subject
    assert cfg.identity <= cfg.subject
    assert cfg.bag_gated <= cfg.subject
    assert "framing" in cfg.subject and "framing" in cfg.ungated_exclusive
    assert set(cfg.multi_value_markers) <= cfg.character_invariant
    # The anatomy gate is opt-in, so `body_parts` is named on its own.
    assert cfg.view_anatomy and cfg.view_anatomy.isdisjoint(cfg.view_invariant)


def test_view_invariant_defaults_to_the_character_invariant_set(tmp_path):
    """The shipped YAML omits the key; the two sets are one rule read twice."""
    from anime_tools.captions.clause_vocabulary import load_clause_groups

    path = tmp_path / "policy.yaml"
    path.write_text("character_invariant_groups: [hair_color]\n", encoding="utf-8")
    assert load_clause_groups(path).view_invariant == frozenset({"hair_color"})


def test_the_policy_drives_selection(pipeline_bits):
    """Editing the YAML changes what binds: dropping `framing` from
    `subject_groups` + `priority_groups` is the config spelling of
    ``--no_framing``."""
    import dataclasses

    _, vocabulary, _, _ = pipeline_bits
    cfg = vocabulary.clause_groups
    no_framing = dataclasses.replace(
        cfg,
        subject=cfg.subject - {"framing"},
        priority=tuple(g for g in cfg.priority if g != "framing"),
    )
    kwargs = {
        "flat_bag": frozenset({"maid", "ass focus"}),
        "attributable": frozenset({"ass focus"}),
        "shared": frozenset(),
        "max_tags": 8,
        "name_confidence": 0.5,
        "allow_unlisted_names": False,
        "max_novel_tags": 1,
    }
    args = ({"ass focus": 0.77, "maid": 0.7}, {"framing": "ass focus"})
    assert "ass focus" in vocabulary.select(*args, **kwargs)
    unbound = dataclasses.replace(vocabulary, clause_groups=no_framing)
    assert "ass focus" not in unbound.select(*args, **kwargs)


def test_a_typo_in_the_policy_is_reported(tmp_path, caplog):
    """A group the checkpoint does not declare matches nothing, and is warned about
    rather than silently disabling its rule."""
    import json
    import logging

    from anime_tools.captions.clause_vocabulary import load_clause_vocabulary

    (tmp_path / "vocab.json").write_text(
        json.dumps({"tags": [{"name": "blonde hair", "category": "general"}]}),
        encoding="utf-8",
    )
    (tmp_path / "groups.yaml").write_text(
        "hair_color:\n  mode: softmax_when_solo\n  tags: [blonde hair]\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        vocabulary = load_clause_vocabulary(tmp_path)
    # The shipped policy names many groups this toy checkpoint lacks.
    assert "silently disable" in caplog.text
    assert vocabulary.gated_groups() >= {"hair_color"}


# ----- bag-tag keep relaxation --------------------------------------------


_RELAX_CAPTION = (
    "safe, 2girls, akita neru, hatsune miku, @channel, blonde hair, aqua hair, "
    "maid, playboy bunny, simple background"
)


def _relax_predictions(left_scores, right_scores, thresholds):
    """Two crops with full scores/thresholds, kept = the strict decision."""

    def crop(scores, groups):
        return {
            "kept": {t: s for t, s in scores.items() if s >= thresholds.get(t, 1.0)},
            "scores": dict(scores),
            "thresholds": dict(thresholds),
            "groups": groups,
        }

    return [
        crop(left_scores, {"hair_color": "blonde hair"}),
        crop(right_scores, {"hair_color": "aqua hair"}),
    ]


_RELAX_THRESHOLDS = {
    "akita neru": 0.5,
    "hatsune miku": 0.5,
    "blonde hair": 0.5,
    "aqua hair": 0.5,
    "maid": 0.8,
    "playboy bunny": 0.8,
}


def _relax_proposal(pipeline_bits, left_scores, right_scores, **option_overrides):
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    return propose_for_image(
        image,
        _RELAX_CAPTION,
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(
            _relax_predictions(left_scores, right_scores, _RELAX_THRESHOLDS)
        ),
        vocabulary=vocabulary,
        options=Options(**option_overrides),
    )


_RELAX_LEFT = {"akita neru": 0.9, "blonde hair": 0.8, "maid": 0.45}
_RELAX_RIGHT = {"hatsune miku": 0.9, "aqua hair": 0.8, "maid": 0.05}


def test_bag_relax_defaults_to_the_shipped_recipe(pipeline_bits):
    """The shipped default is 0.35/0.85: `maid` at 0.45 clears 0.8 * 0.35 on the
    left crop with no options set."""
    from anime_tools.stages.position_captions import PositionCaptionOptions

    assert PositionCaptionOptions().bag_relax == 0.35
    assert PositionCaptionOptions().bag_word_relax == 0.85
    proposal = _relax_proposal(pipeline_bits, _RELAX_LEFT, _RELAX_RIGHT)
    assert "maid" in parse_caption(proposal.proposed).clauses[0].tags


def test_bag_relax_one_restores_the_strict_pipeline(pipeline_bits):
    """`1.0` is the off switch: `maid` is under its 0.8 threshold on both crops,
    so the strict pipeline never sees it."""
    proposal = _relax_proposal(
        pipeline_bits, _RELAX_LEFT, _RELAX_RIGHT, bag_relax=1.0, bag_word_relax=1.0
    )
    parsed = parse_caption(proposal.proposed)
    assert not any("maid" in c.tags for c in parsed.clauses)
    assert "maid" in parsed.flat_tags


def test_bag_relax_binds_and_moves_a_corroborated_tag(pipeline_bits):
    """0.45 clears 0.8 * 0.5 on the left crop only, so it is attributable and
    moves (the 5828184 `black panties` shape: 0.498 vs 0.066 against 0.800)."""
    proposal = _relax_proposal(pipeline_bits, _RELAX_LEFT, _RELAX_RIGHT, bag_relax=0.5)
    parsed = parse_caption(proposal.proposed)
    assert "maid" in parsed.clauses[0].tags
    assert "maid" not in parsed.flat_tags
    assert "maid" in {m["tag"] for m in proposal.moved}


def test_bag_relax_never_admits_a_novel_tag(pipeline_bits):
    """Relaxation reads the bag only: a tag the caption never contained stays
    invisible however low the floor goes."""
    left = dict(_RELAX_LEFT, apron=0.79)  # sub-threshold and NOT in the bag
    thresholds = dict(_RELAX_THRESHOLDS, apron=0.8)
    from anime_tools.stages.position_captions import propose_for_image

    image, vocabulary, _, Options = pipeline_bits
    proposal = propose_for_image(
        image,
        _RELAX_CAPTION,
        detect_fn=_detector(
            {0.5: [((0, 0, 400, 500), 0.9), ((600, 0, 1000, 500), 0.9)]}
        ),
        tag_fn=_tagger(_relax_predictions(left, _RELAX_RIGHT, thresholds)),
        vocabulary=vocabulary,
        options=Options(bag_relax=0.1),
    )
    assert "apron" not in proposal.proposed


def test_bag_relax_min_score_floors_near_noise_admissions(pipeline_bits):
    """A relaxed admission still needs the absolute floor: `maid` at 0.29 clears
    the relaxed threshold (0.8 * 0.35 = 0.28) but not the 0.3 default floor.
    With the floor off it binds again."""
    left = dict(_RELAX_LEFT, maid=0.29)
    floored = _relax_proposal(pipeline_bits, left, _RELAX_RIGHT)
    parsed = parse_caption(floored.proposed)
    assert not any("maid" in c.tags for c in parsed.clauses)
    assert "maid" in parsed.flat_tags

    unfloored = _relax_proposal(
        pipeline_bits, left, _RELAX_RIGHT, bag_relax_min_score=0.0
    )
    assert "maid" in parse_caption(unfloored.proposed).clauses[0].tags


def test_bag_relax_min_score_never_touches_the_taggers_own_keeps(pipeline_bits):
    """The floor gates only the relax path: a tag the tagger keeps at its own
    threshold binds even when that score is under the floor."""
    left = dict(_RELAX_LEFT, maid=0.85)  # above its own 0.8 threshold
    proposal = _relax_proposal(
        pipeline_bits, left, _RELAX_RIGHT, bag_relax_min_score=0.9
    )
    assert "maid" in parse_caption(proposal.proposed).clauses[0].tags


def test_bag_word_relax_compounds_per_extra_word(pipeline_bits):
    """`playboy bunny` (two words) at 0.55: 0.8 * 0.7 = 0.56 misses, and the 0.9
    word bonus lowers the floor to 0.504."""
    left = dict(_RELAX_LEFT, **{"playboy bunny": 0.55})
    without = _relax_proposal(
        pipeline_bits, left, _RELAX_RIGHT, bag_relax=0.7, bag_word_relax=1.0
    )
    assert "playboy bunny" not in {
        t for c in parse_caption(without.proposed).clauses for t in c.tags
    }
    bonus = _relax_proposal(
        pipeline_bits, left, _RELAX_RIGHT, bag_relax=0.7, bag_word_relax=0.9
    )
    assert "playboy bunny" in parse_caption(bonus.proposed).clauses[0].tags


def test_bag_relax_blocks_a_move_the_strict_sets_grant(pipeline_bits):
    """Cuts both ways: a rival crop's borderline score becomes a keep, so the tag
    is shared and stays flat. The rival's 0.6 clears the 0.25 attribution margin
    (1 - 0.6/0.9 = 0.33), so only the relaxed kept set catches it.
    """
    left = dict(_RELAX_LEFT, maid=0.9)
    right = dict(_RELAX_RIGHT, maid=0.6)  # sub-threshold, but clearly there
    strict = _relax_proposal(
        pipeline_bits, left, right, bag_relax=1.0, bag_word_relax=1.0
    )
    assert "maid" in {m["tag"] for m in strict.moved}
    relaxed = _relax_proposal(pipeline_bits, left, right, bag_relax=0.7)
    parsed = parse_caption(relaxed.proposed)
    assert "maid" in parsed.flat_tags
    assert not any("maid" in c.tags for c in parsed.clauses)
    assert "maid" not in {m["tag"] for m in relaxed.moved}
