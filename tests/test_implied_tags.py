from anime_tools.captions.implied_tags import drop_implied_tags, implies


def test_suffix_and_extra_pairs_imply():
    assert implies("very long hair", "long hair")
    assert implies("diagonal-striped neckerchief", "neckerchief")
    assert implies("bowtie", "bow")
    assert implies("red bowtie", "bow")
    assert implies("absurdly long hair", "very long hair")


def test_bowtie_supersedes_bow_and_neckerchief_with_the_same_modifier():
    assert implies("diagonal-striped bowtie", "diagonal-striped bow")
    assert implies("bowtie", "neckerchief")
    assert implies("diagonal-striped bowtie", "diagonal-striped neckerchief")
    assert implies("red bowtie", "red bow")
    assert not implies("bowtie", "hair bow")
    assert not implies("red bowtie", "diagonal-striped bow")


def test_meaning_changing_prefixes_do_not_imply():
    assert not implies("no pants", "pants")
    assert not implies("spoken heart", "heart")
    assert not implies("looking back", "back")
    assert not implies("long hair", "long hair")


def test_bag_cut_by_any_scope_clause_only_by_its_own():
    text = (
        "1girl, 2girls, bow, long hair, neckerchief, diagonal-striped neckerchief. "
        "On the left, red bowtie, long hair, very long hair. On the right, long hair."
    )
    out = drop_implied_tags(text)
    assert out.text == (
        "1girl, 2girls, diagonal-striped neckerchief. "
        "On the left, red bowtie, very long hair. On the right, long hair."
    )
    assert sorted(out.dropped_tags) == ["bow", "long hair", "long hair", "neckerchief"]


def test_keep_shields_and_untouched_caption_is_byte_identical():
    text = "1girl,  long hair, very long hair"
    assert drop_implied_tags(text, keep=["long hair"]).text == text
    assert drop_implied_tags("1girl,  smile").text == "1girl,  smile"
