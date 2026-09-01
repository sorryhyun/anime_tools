"""OCR: the language classifier, the sidecar, and what a run does to a caption.

The split these pin is the whole design. A run answers two questions and writes
two places: the recognized text goes into ``{stem}.ocr.txt``, and the caption
gets at most a *script tag* — never the string, because the caption grammar has
no clause that could hold one. So the shape that looks like a bug and is not is a
Japanese-only image: sidecar written, caption untouched, zero proposals.

None of this loads a model. The classifier and the sidecar are stdlib, the stage
takes its reader as an argument, and the CTC decode is exercised against a
hand-built logit array — so the suite stays CPU-only and needs no 139 MB
download.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anime_tools.captions.history import read_history
from anime_tools.captions.ocr_sidecar import (
    OCR_SIDECAR_SUFFIX,
    OcrLine,
    ocr_sidecar_path,
    read_ocr,
    write_ocr_for,
)
from anime_tools.ocr.script import LANGS, parse_langs, script_of, tags_for
from anime_tools.stages.ocr import (
    OcrOptions,
    add_script_tags,
    keep_lines,
    run_ocr_captions,
)


def line(text: str, lang: str, *, seq: int = 1, score: float = 0.95, box=(0, 0, 40, 20)):
    return OcrLine(seq=seq, box=box, lang=lang, score=score, text=text)


# ---- the classifier ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("SUMMER SALE", "en"),
        ("Open 24h", "en"),
        ("2024", "en"),
        ("こんにちは", "ja"),
        ("「テスト」", "ja"),
        # Kanji *with* kana is Japanese however much kanji there is -- the whole
        # reason kana is tested before han.
        ("日本語のテスト", "ja"),
        ("中文测试", "zh"),
        # Pure han is indistinguishable, and reads as zh by construction.
        ("漢字", "zh"),
        # Nothing that votes: the detector found a texture, not a line.
        ("!?", "other"),
        ("", "other"),
    ],
)
def test_a_line_is_classified_by_the_script_it_contains(text, lang):
    assert script_of(text) == lang


def test_the_allowlist_refuses_a_typo_rather_than_matching_nothing():
    assert parse_langs("en, ja ,zh") == ("en", "ja", "zh")
    assert parse_langs("ja,ja") == ("ja",)
    # A silently-empty allowlist reads exactly like a clean corpus, which is why
    # this raises instead.
    with pytest.raises(ValueError, match="unknown"):
        parse_langs("en,kr")
    with pytest.raises(ValueError):
        parse_langs("  ")


def test_korean_is_not_in_the_vocabulary_because_the_model_cannot_read_it():
    assert "ko" not in LANGS


# ---- the tag map ------------------------------------------------------


def test_japanese_earns_no_tag_and_that_is_the_point():
    """Danbooru tags text that is *not* Japanese -- there is no ``japanese_text``
    -- so emitting one would put a tag in every caption that the correction pass
    types as unknown forever after."""
    assert tags_for(["ja"]) == ()
    assert tags_for(["en"]) == ("english text",)
    assert tags_for(["zh"]) == ("chinese text",)
    assert tags_for(["other"]) == ()


def test_two_languages_are_bilingual_even_when_one_of_them_is_nameless():
    # ja contributes no tag of its own and still makes the image bilingual.
    assert tags_for(["en", "ja"]) == ("english text", "bilingual text")
    assert tags_for(["en", "zh"]) == ("english text", "chinese text", "bilingual text")


# ---- the sidecar ------------------------------------------------------


def test_the_sidecar_round_trips_through_a_tab_and_a_multi_dot_stem(tmp_path: Path):
    assert ocr_sidecar_path(Path("a/b.c.txt")).name == "b.c" + OCR_SIDECAR_SUFFIX
    caption = tmp_path / "a.b.txt"
    # A tab inside the text is why the text field is last.
    lines = [line("こん\tにちは", "ja"), line("SALE", "en", seq=2)]
    p = write_ocr_for(caption, lines)
    assert p.name == "a.b" + OCR_SIDECAR_SUFFIX
    assert read_ocr(p) == lines


def test_no_text_deletes_the_sidecar_rather_than_writing_an_empty_one(tmp_path: Path):
    caption = tmp_path / "a.txt"
    p = write_ocr_for(caption, [line("SALE", "en")])
    assert p.is_file()
    # A re-run over re-cropped pixels that no longer hold the text must not leave
    # the old claim standing.
    write_ocr_for(caption, [])
    assert not p.exists()
    assert read_ocr(p) == []


def test_a_damaged_record_costs_its_line_and_never_the_run(tmp_path: Path):
    p = tmp_path / "a.ocr.txt"
    p.write_text(
        "# header\n"
        "\n"
        "1\t0,0,10,10\ten\t0.9\tkept\n"
        "two\t0,0,10,10\ten\t0.9\tbad seq\n"
        "3\tnot,a,box,x\ten\t0.9\tbad box\n"
        "4\t0,0,10,10\ten\tNaN-ish\tbad score\n"
        "5\t0,0,10,10\ten\n",
        encoding="utf-8",
    )
    assert [ln.text for ln in read_ocr(p)] == ["kept"]


# ---- what reaches the caption -----------------------------------------


def test_the_allowlist_drops_a_line_whole_and_renumbers_what_is_left():
    lines = [line("SALE", "en", seq=1), line("!?", "other", seq=2), line("A", "ja", seq=3)]
    kept = keep_lines(lines, ("en", "ja"))
    # Renumbered *after* the filter: a gap in a sidecar's sequence reads as a bug.
    assert [(ln.seq, ln.lang) for ln in kept] == [(1, "en"), (2, "ja")]


def test_a_tag_the_caption_already_carries_is_not_added_again():
    lines = keep_lines([line("SALE", "en")], ("en",))
    # The underscore spelling is the same tag -- normalize_tag, not lower().
    assert add_script_tags("1girl, english_text", lines) == ("1girl, english_text", ())
    # And one bound inside a position clause counts as present, so a run after
    # the clause rewrite cannot re-flatten it into the bag.
    caption = "1girl. On the left, english text"
    assert add_script_tags(caption, lines)[1] == ()


def test_the_tag_lands_in_the_flat_bag_and_the_clauses_survive():
    lines = keep_lines([line("SALE", "en")], ("en",))
    out, added = add_script_tags("safe, 1girl. On the left, solo", lines)
    assert added == ("english text",)
    assert out == "safe, 1girl, english text. On the left, solo."


# ---- the stage --------------------------------------------------------


def _dataset(tmp_path: Path, caption: str = "safe, 1girl"):
    from PIL import Image

    src = tmp_path / "image_dataset"
    dst = tmp_path / "workspace" / "resized"
    dst.mkdir(parents=True)
    src.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(dst / "a.png")
    (src / "a.txt").write_text(caption, encoding="utf-8")
    return src, dst


def _run(src, dst, lines, *, apply: bool, **kw):
    return run_ocr_captions(
        resized_dir=dst,
        source_dir=src,
        read_fn=lambda _p: list(lines),
        options=OcrOptions(**kw),
        apply=apply,
    )


def test_a_dry_run_reports_every_line_and_writes_nothing(tmp_path: Path):
    src, dst = _dataset(tmp_path)
    rows, stats = _run(src, dst, [line("SALE", "en")], apply=False)
    assert stats.lines == 1 and stats.proposed == 1 and stats.written == 0
    assert rows[0].added == ("english text",)
    # The report carries the sidecar it would have written, before it exists.
    assert rows[0].to_row()["lines"][0]["text"] == "SALE"
    assert not (dst / "a.ocr.txt").exists()
    assert not (dst / "a.txt").exists()


def test_an_applied_run_writes_the_sidecar_and_the_tag(tmp_path: Path):
    src, dst = _dataset(tmp_path)
    rows, stats = _run(src, dst, [line("SALE", "en")], apply=True)
    assert stats.written == 1 and stats.sidecars == 1
    # The tag goes into the *revised* caption; the master is never touched.
    assert (dst / "a.txt").read_text(encoding="utf-8") == "safe, 1girl, english text"
    assert (src / "a.txt").read_text(encoding="utf-8") == "safe, 1girl"
    assert [ln.text for ln in read_ocr(dst / "a.ocr.txt")] == ["SALE"]


def test_a_japanese_only_image_gets_a_sidecar_and_no_caption_change(tmp_path: Path):
    """The shape that looks wrong and is not."""
    src, dst = _dataset(tmp_path)
    rows, stats = _run(src, dst, [line("こんにちは", "ja")], apply=True)
    assert stats.lines == 1 and stats.sidecars == 1
    assert stats.proposed == 0 and stats.written == 0
    assert stats.skipped["no-tags"] == 1
    assert read_ocr(dst / "a.ocr.txt")[0].text == "こんにちは"
    assert not (dst / "a.txt").exists()


def test_no_tags_makes_the_stage_a_pure_reader(tmp_path: Path):
    src, dst = _dataset(tmp_path)
    _, stats = _run(src, dst, [line("SALE", "en")], apply=True, tags=False)
    assert stats.sidecars == 1 and stats.written == 0
    assert not (dst / "a.txt").exists()


def test_the_replaced_caption_becomes_a_version_rather_than_being_gone(tmp_path: Path):
    src, dst = _dataset(tmp_path)
    (dst / "a.txt").write_text("safe, 1girl", encoding="utf-8")
    _run(src, dst, [line("SALE", "en")], apply=True)
    # A run writes with no Apply gate in front of it, so what it replaced has to
    # survive as a badge -- filed under this stage's own hand.
    history = read_history(dst / "a.history.txt")
    assert [(h.by, h.text) for h in history] == [("ocr", "safe, 1girl")]


def test_an_uncaptioned_image_is_skipped_the_way_every_stage_skips_one(tmp_path: Path):
    from PIL import Image

    src, dst = _dataset(tmp_path)
    Image.new("RGB", (64, 64), "white").save(dst / "b.png")
    _, stats = _run(src, dst, [line("SALE", "en")], apply=True)
    assert stats.seen == 2 and stats.skipped["no-caption"] == 1
    assert not (dst / "b.ocr.txt").exists()


# ---- the pieces of the engine that need no weights ---------------------


def test_ctc_decode_collapses_repeats_and_drops_blanks():
    """The one place an off-by-one would not fail but would silently return
    fluent-looking garbage, so it is pinned against a hand-built array."""
    np = pytest.importorskip("numpy")
    from anime_tools.ocr._onnx import TextRecognizer

    # index 0 is the blank; the vocabulary is ['<blank>', 'a', 'b', ' '].
    rec = TextRecognizer(session=None, vocab=["<blank>", "a", "b", " "])
    logits = np.zeros((1, 7, 4), dtype="float32")
    # a a _ a b b _  ->  "a" "a" "b"  (repeats collapse only across a blank)
    for step, cls in enumerate([1, 1, 0, 1, 2, 2, 0]):
        logits[0, step, cls] = 1.0
    (text, score) = rec._decode(logits)[0]
    assert text == "aab"
    assert score == pytest.approx(1.0)


def test_an_all_blank_row_is_an_empty_string_and_not_a_crash():
    np = pytest.importorskip("numpy")
    from anime_tools.ocr._onnx import TextRecognizer

    rec = TextRecognizer(session=None, vocab=["<blank>", "a"])
    logits = np.zeros((1, 4, 2), dtype="float32")
    logits[0, :, 0] = 1.0
    assert rec._decode(logits)[0] == ("", 0.0)


def test_the_vocabulary_is_derived_from_the_graph_not_assumed():
    """A dictionary and a graph that are not a pair must fail loudly: the
    shipped medium model answers 18,710 classes to an 18,708-entry dict, i.e.
    a blank *and* the space upstream appends."""
    from anime_tools.ocr._onnx import TextRecognizer

    class FakeOut:
        def __init__(self, n):
            self.shape = ["N", "T", n]

    class FakeSession:
        def __init__(self, n):
            self._n = n

        def get_outputs(self):
            return [FakeOut(self._n)]

    chars = ["a", "b", "c"]
    assert TextRecognizer._vocab(chars, FakeSession(5)) == ["<blank>", "a", "b", "c", " "]
    assert TextRecognizer._vocab(chars, FakeSession(4)) == ["<blank>", "a", "b", "c"]
    with pytest.raises(RuntimeError, match="not a pair"):
        TextRecognizer._vocab(chars, FakeSession(9))


def test_reading_order_runs_across_a_row_before_down_the_page():
    from anime_tools.ocr._onnx import reading_order

    lines = [
        line("bottom", "en", box=(10, 200, 90, 230)),
        line("right", "en", box=(300, 10, 380, 40)),
        line("left", "en", box=(10, 12, 90, 42)),
    ]
    assert [ln.text for ln in reading_order(lines)] == ["left", "right", "bottom"]


def test_the_loader_and_the_download_catalog_name_one_directory():
    """A download and a load that spell the same path twice can drift, and the
    symptom is a Download button that appears to do nothing."""
    from anime_tools.downloads import by_id, default_ppocr_det_dir, default_ppocr_rec_dir

    rows = by_id()
    assert rows["ppocr_det"].dest == default_ppocr_det_dir()
    assert rows["ppocr_rec"].dest == default_ppocr_rec_dir()
    # Two directories, not one: both repos ship a file called inference.onnx.
    assert default_ppocr_det_dir() != default_ppocr_rec_dir()
    for row in (rows["ppocr_det"], rows["ppocr_rec"]):
        assert row.stages == ("ocr",)
        assert set(row.files) == {"inference.onnx", "inference.yml"}
