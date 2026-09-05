"""``--caption_drop_groups`` (GH #95): strip tag *kinds* at mirror time."""

from __future__ import annotations

from pathlib import Path

from anime_tools.captions.correction import (
    CaptionCorrectionOptions,
    correct_caption,
    load_tag_knowledge_base,
)
from anime_tools.captions.tag_drop_groups import (
    parse_drop_groups,
    should_drop_tag,
    tag_drop_group,
)


def _kb(tmp_path: Path):
    path = tmp_path / "tags.csv"
    path.write_text(
        """name,category,post_count,description
1girl,0,10,"[인물 > 인원수] count"
solo,0,10,"[인물 > 인원수] count"
hatsune_miku,4,10,"[캐릭터 > 보컬로이드/버추얼] character"
vocaloid,3,10,"[작품/출처 > 게임] copyright"
sincos,1,10,"[아티스트 > 개인] artist"
highres,5,10,"[메타 > 화질/해상도] meta"
long_hair,0,10,"[머리카락 > 머리 길이] hair"
school_uniform,0,10,"[의상 > 상의] clothing"
backlighting,0,10,"[효과/연출 > 조명] lighting"
speech_bubble,0,10,"[효과/연출 > 텍스트/말풍선] text"
sitting,0,10,"[포즈/구도 > 포즈] pose"
from_above,0,10,"[포즈/구도 > 시점/앵글] angle"
elf,0,10,"[인물 > 종족/비인간] species"
mystery_tag,0,10,no taxonomy path here""",
        encoding="utf-8",
    )
    return load_tag_knowledge_base(path)


def test_parse_drop_groups_normalizes_slugs_and_keeps_literal_paths():
    assert parse_drop_groups(" Artist, lighting ,, pose") == (
        "artist",
        "lighting",
        "pose",
    )
    assert parse_drop_groups(["clothing", "효과/연출 > 조명"]) == (
        "clothing",
        "효과/연출 > 조명",
    )
    assert parse_drop_groups("") == ()
    assert parse_drop_groups(None) == ()


def test_tag_drop_group_shape_beats_kb(tmp_path):
    kb = _kb(tmp_path)
    assert tag_drop_group("@sincos", kb) == "artist"
    assert tag_drop_group("@never seen", kb) == "artist"  # dataset-only artist
    assert tag_drop_group("1girl", kb) == "count"
    assert tag_drop_group("hatsune miku", kb) == "character"
    assert tag_drop_group("vocaloid", kb) == "copyright"
    assert tag_drop_group("highres", kb) == "meta"
    assert tag_drop_group("school uniform", kb) == "clothing"
    assert tag_drop_group("backlighting", kb) == "effect"  # coarse only
    assert tag_drop_group("safe", kb) is None
    assert tag_drop_group("mystery tag", kb) is None
    assert tag_drop_group("not in kb", kb) is None


def test_should_drop_fine_selector_and_literal_path(tmp_path):
    kb = _kb(tmp_path)
    assert should_drop_tag("backlighting", kb, ("lighting",))
    assert not should_drop_tag("speech bubble", kb, ("lighting",))
    assert should_drop_tag("speech bubble", kb, ("effect",))
    assert should_drop_tag("speech bubble", kb, ("효과/연출 > 텍스트/말풍선",))
    assert should_drop_tag("from above", kb, ("framing",))
    assert not should_drop_tag("sitting", kb, ("framing",))
    assert should_drop_tag("sitting", kb, ("pose",))


def test_structural_tags_never_fall_to_content_selectors(tmp_path):
    kb = _kb(tmp_path)
    # ``1girl`` sits under ``인물 > 인원수`` in the KB, but ``person`` must not eat it.
    assert not should_drop_tag("1girl", kb, ("person",))
    assert should_drop_tag("elf", kb, ("person",))
    assert should_drop_tag("elf", kb, ("species",))
    # Unknown tags and ratings are never dropped, whatever the selector.
    assert not should_drop_tag("mystery tag", kb, ("effect", "clothing", "person"))
    assert not should_drop_tag("not in kb", kb, ("artist", "clothing"))
    assert not should_drop_tag("nsfw", kb, ("meta", "person"))


def test_correct_caption_drops_groups_flat(tmp_path):
    kb = _kb(tmp_path)
    caption = (
        "safe, 1girl, hatsune miku, vocaloid, @sincos, long hair, "
        "school uniform, backlighting, sitting, mystery tag"
    )
    result = correct_caption(
        caption,
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=False, drop_groups=("artist", "lighting", "clothing")
        ),
    )
    assert result.text == (
        "safe, 1girl, hatsune miku, vocaloid, long hair, sitting, mystery tag"
    )
    assert result.dropped_tags == ("@sincos", "school uniform", "backlighting")
    assert result.changed


def test_drop_artist_then_insert_no_artist_marks_the_slot(tmp_path):
    kb = _kb(tmp_path)
    result = correct_caption(
        "1girl, @sincos, long hair",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True, drop_groups=("artist",)
        ),
    )
    assert result.text == "1girl, @no-artist, long hair"
    assert result.inserted_no_artist


def test_trigger_word_survives_its_own_group(tmp_path):
    kb = _kb(tmp_path)
    result = correct_caption(
        "1girl, @sincos, long hair",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=False, trigger_word="@sincos", drop_groups=("artist",)
        ),
    )
    assert result.text == "1girl, @sincos, long hair"
    assert result.dropped_tags == ()


def test_drop_groups_inside_position_clauses(tmp_path):
    kb = _kb(tmp_path)
    caption = (
        "1girl, @sincos, backlighting. On the left, hatsune miku, school uniform, "
        "long hair. On the right, school uniform."
    )
    result = correct_caption(
        caption,
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=False, drop_groups=("clothing",)
        ),
    )
    # Clothing gone from bag-side clause tags; the right clause emptied → removed whole.
    assert (
        result.text
        == "1girl, @sincos, backlighting. On the left, hatsune miku, long hair."
    )
    assert result.dropped_tags == ("school uniform", "school uniform")
    # Empty selector = no-op, clauses round-trip verbatim.
    same = correct_caption(
        caption, kb, options=CaptionCorrectionOptions(insert_no_artist=False)
    )
    assert same.dropped_tags == ()
    assert "On the right, school uniform." in same.text


def test_text_clause_passes_through_correction_untouched(tmp_path):
    kb = _kb(tmp_path)
    caption = (
        "1girl, @sincos, long hair, speech bubble. On the left, backlighting. "
        'Japanese text reads as "long hair, please", "sincos".'
    )
    result = correct_caption(
        caption,
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=False, drop_groups=("artist", "lighting")
        ),
    )
    # The quoted lines are not taxonomy tags: nothing inside them is dropped
    # or reordered, and the sentence still composes last.
    assert result.text == (
        "1girl, long hair, speech bubble. "
        'Japanese text reads as "long hair, please", "sincos".'
    )
    assert result.dropped_tags == ("@sincos", "backlighting")
