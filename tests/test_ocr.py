"""OCR: the sidecar, the content floors and reading order, and what a run writes.

A run writes the read text to ``{stem}.ocr.txt`` in the OCR tree and reads or
writes no caption. Nothing here loads a model: the stage takes its reader as an
argument, and the floors run against hand-built lines.
"""

from __future__ import annotations

from pathlib import Path

from anime_tools.captions.ocr_sidecar import (
    OCR_SIDECAR_SUFFIX,
    OcrLine,
    ocr_sidecar_path,
    read_ocr,
    write_ocr_for,
)
from anime_tools.stages.ocr import number_lines, read_tree


def line(
    text: str,
    *,
    seq: int = 1,
    score: float = 0.95,
    box=(0, 0, 400, 40),
    det: float = 0.0,
):
    return OcrLine(seq=seq, box=box, score=score, text=text, det=det)


# ---- the sidecar ------------------------------------------------------


def test_the_sidecar_round_trips_through_a_tab_and_a_multi_dot_stem(tmp_path: Path):
    assert ocr_sidecar_path(Path("a/b.c.txt")).name == "b.c" + OCR_SIDECAR_SUFFIX
    # A tab inside the text is why the text field is last.
    lines = [line("こん\tにちは", det=0.875), line("SALE", seq=2, score=0.5)]
    p = write_ocr_for(tmp_path, Path("a.b.txt"), lines)
    assert p.name == "a.b" + OCR_SIDECAR_SUFFIX
    assert read_ocr(p) == lines
    # seq ⇥ box ⇥ det ⇥ score ⇥ text
    assert p.read_text(encoding="utf-8").splitlines()[1:] == [
        "1\t0,0,400,40\t0.875\t0.950\tこん\tにちは",
        "2\t0,0,400,40\t0.000\t0.500\tSALE",
    ]


def test_a_legacy_four_field_record_reads_with_no_detector_score(tmp_path: Path):
    p = tmp_path / "a.ocr.txt"
    p.write_text(
        "# header\n1\t0,0,10,10\t0.000\tヒキィ\n2\t0,20,10,30\t0.000\tこん\tにちは\n",
        encoding="utf-8",
    )
    assert read_ocr(p) == [
        line("ヒキィ", box=(0, 0, 10, 10), score=0.0),
        line("こん\tにちは", seq=2, box=(0, 20, 10, 30), score=0.0),
    ]


def test_the_sidecar_mirrors_the_resized_tree_and_digs_its_own_subdir(tmp_path: Path):
    """The OCR tree joins by relative path, digging a nested subdir as needed."""
    p = write_ocr_for(tmp_path, Path("artist/a.txt"), [line("SALE")])
    assert p == tmp_path / "artist" / ("a" + OCR_SIDECAR_SUFFIX)
    assert read_ocr(p)[0].text == "SALE"


def test_no_text_deletes_the_sidecar_rather_than_writing_an_empty_one(tmp_path: Path):
    p = write_ocr_for(tmp_path, Path("a.txt"), [line("SALE")])
    assert p.is_file()
    # A re-run that finds no text must not leave the old claim standing.
    write_ocr_for(tmp_path, Path("a.txt"), [])
    assert not p.exists()
    assert read_ocr(p) == []


def test_a_damaged_record_costs_its_line_and_never_the_run(tmp_path: Path):
    p = tmp_path / "a.ocr.txt"
    p.write_text(
        "# header\n"
        "\n"
        "1\t0,0,10,10\t0.9\tkept\n"
        "two\t0,0,10,10\t0.9\tbad seq\n"
        "3\tnot,a,box,x\t0.9\tbad box\n"
        "4\t0,0,10,10\tNaN-ish\tbad score\n"
        "5\t0,0,10,10\n"
        "6\t0,0,10,10\tNaN-ish\t0.9\tbad det\n",
        encoding="utf-8",
    )
    assert [ln.text for ln in read_ocr(p)] == ["kept"]


# ---- what reaches the sidecar -----------------------------------------


def test_every_line_the_reader_hands_over_reaches_the_sidecar_in_order():
    """The stage is not where a line is dropped — the reader has already
    filtered and joined — so every line it is handed is renumbered in order."""
    lines = [line("SALE", seq=7), line("!?", seq=9), line("心", seq=40)]
    kept = number_lines(lines)
    assert [(ln.seq, ln.text) for ln in kept] == [(1, "SALE"), (2, "!?"), (3, "心")]


# ---- the stage --------------------------------------------------------


def _dataset(tmp_path: Path):
    from PIL import Image

    dst = tmp_path / "workspace" / "resized"
    ocr = tmp_path / "workspace" / "ocr"
    dst.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(dst / "a.png")
    return dst, ocr


def _run(dst, ocr, lines, *, apply: bool, **kw):
    return read_tree(
        resized_dir=dst,
        ocr_dir=ocr,
        read_fn=lambda _p: list(lines),
        apply=apply,
        **kw,
    )


def test_a_dry_run_reports_every_line_and_writes_nothing(tmp_path: Path):
    dst, ocr = _dataset(tmp_path)
    rows, stats = _run(dst, ocr, [line("SALE")], apply=False)
    assert stats.lines == 1 and stats.with_text == 1 and stats.sidecars == 0
    # The report carries the sidecar it would have written, before it exists.
    assert rows[0].to_row()["lines"][0]["text"] == "SALE"
    assert not ocr.exists()


def test_an_applied_run_writes_the_sidecar_into_the_ocr_tree(tmp_path: Path):
    dst, ocr = _dataset(tmp_path)
    _, stats = _run(dst, ocr, [line("SALE")], apply=True)
    assert stats.sidecars == 1
    assert [ln.text for ln in read_ocr(ocr / "a.ocr.txt")] == ["SALE"]
    # And nowhere near the resized tree.
    assert not (dst / "a.ocr.txt").exists()


def test_the_stage_never_touches_a_caption(tmp_path: Path):
    """Nothing in the resized tree changes: no caption, history or variants."""
    dst, ocr = _dataset(tmp_path)
    (dst / "a.txt").write_text("safe, 1girl", encoding="utf-8")
    _run(dst, ocr, [line("SALE")], apply=True)
    assert (dst / "a.txt").read_text(encoding="utf-8") == "safe, 1girl"
    assert not (dst / "a.history.txt").exists()
    assert not (dst / "a.variants.txt").exists()


def test_a_japanese_only_image_gets_a_sidecar_like_any_other(tmp_path: Path):
    dst, ocr = _dataset(tmp_path)
    _, stats = _run(dst, ocr, [line("こんにちは")], apply=True)
    assert stats.lines == 1 and stats.sidecars == 1
    assert read_ocr(ocr / "a.ocr.txt")[0].text == "こんにちは"


def test_an_uncaptioned_image_is_read_like_any_other(tmp_path: Path):
    """No caption is needed, so having none is not a reason to skip an image."""
    from PIL import Image

    dst, ocr = _dataset(tmp_path)
    Image.new("RGB", (64, 64), "white").save(dst / "b.png")
    _, stats = _run(dst, ocr, [line("SALE")], apply=True)
    assert stats.seen == 2 and stats.sidecars == 2
    assert not stats.skipped
    assert read_ocr(ocr / "b.ocr.txt")[0].text == "SALE"


def test_an_image_with_no_text_is_counted_and_leaves_no_sidecar(tmp_path: Path):
    dst, ocr = _dataset(tmp_path)
    _, stats = _run(dst, ocr, [], apply=True)
    assert stats.with_text == 0 and stats.skipped["no-text"] == 1
    assert not (ocr / "a.ocr.txt").exists()


def test_the_batched_reader_answers_what_the_one_at_a_time_reader_does(tmp_path: Path):
    """``read_iter_fn`` is a pure substitution for ``read_fn``."""
    from PIL import Image

    dst, ocr = _dataset(tmp_path)
    for name in ("b", "c"):
        Image.new("RGB", (64, 64), "white").save(dst / f"{name}.png")
    lines = [line("SALE")]

    one, stats_one = read_tree(
        resized_dir=dst, ocr_dir=ocr, read_fn=lambda _p: list(lines), apply=False
    )
    many, stats_many = read_tree(
        resized_dir=dst,
        ocr_dir=ocr,
        read_fn=lambda _p: list(lines),
        read_iter_fn=lambda paths: (list(lines) for _ in paths),
        apply=False,
    )
    assert [r.to_row() for r in one] == [r.to_row() for r in many]
    assert stats_one.lines == stats_many.lines == 3


def test_the_reader_gets_the_whole_run_in_one_call(tmp_path: Path):
    """The stage does no chunking of its own.

    A slice the size of the reader's chunk is a chunk with nothing decoded behind
    it, which idles the GPU for every decode; the reader prefetches across the run
    instead, and only can if it is handed the run.
    """
    from PIL import Image

    dst, ocr = _dataset(tmp_path)
    for i in range(40):
        Image.new("RGB", (64, 64), "white").save(dst / f"b{i:02d}.png")
    calls: list[int] = []

    def reader(paths):
        calls.append(len(paths))
        for _ in paths:
            yield [line("SALE")]

    _, stats = read_tree(
        resized_dir=dst,
        ocr_dir=ocr,
        read_fn=lambda _p: [],
        read_iter_fn=reader,
        apply=False,
    )
    assert calls == [41] and stats.lines == 41


def test_a_result_is_written_before_the_reader_has_finished(tmp_path: Path):
    """An ``--apply`` streams: the reader is still working when the first sidecar
    lands, which is what lets its decode overlap the stage's writes."""
    from PIL import Image

    dst, ocr = _dataset(tmp_path)
    for name in ("b", "c"):
        Image.new("RGB", (64, 64), "white").save(dst / f"{name}.png")
    seen: list[list[str]] = []

    def reader(paths):
        for _ in paths:
            seen.append(sorted(p.name for p in ocr.glob("*.ocr.txt")))
            yield [line("SALE")]

    read_tree(
        resized_dir=dst,
        ocr_dir=ocr,
        read_fn=lambda _p: [],
        read_iter_fn=reader,
        apply=True,
    )
    # The third image is read only after the first two sidecars are on disk.
    assert seen == [[], ["a.ocr.txt"], ["a.ocr.txt", "b.ocr.txt"]]


def test_the_ocr_stage_resolves_one_device_and_gives_it_to_both_models(
    tmp_path, monkeypatch
):
    """The detector and the VL reader both run on torch, so the stage asks once.

    Until 2026-09-09 the detector was an ONNX session and this stage resolved
    *two* devices — ``resolve_onnx_device`` for the detector, ``resolve_device``
    for the reader — because a torch CUDA probe cost an onnxruntime run 1.8x.
    Both halves are torch now, and the whole point of the single answer is that
    the two models cannot land on different devices, so this asks for one probe
    and the same string reaching both loads.
    """
    import anime_tools._device as device_mod
    import anime_tools.ocr as ocr_pkg
    import anime_tools.ocr.sfx as sfx_mod
    from anime_tools.stages.ocr import run_ocr
    from anime_tools.stages.requests import OcrRequest

    dst, ocr_dir = _dataset(tmp_path)
    probes: list[str | None] = []
    given: list[str] = []

    class _Engine:
        def read(self, path):
            return []

        def read_iter(self, paths):
            return ([] for _ in paths)

    class _Reader:
        def read_boxes_scored(self, *a, **kw):
            return []

    def fake_resolve(requested=None):
        probes.append(requested)
        return "cuda:7"

    def fake_load_ocr(*, device, **_kw):
        given.append(device)
        return _Engine()

    def fake_reader_load(*, device, batch_size):
        given.append(device)
        return _Reader()

    monkeypatch.setattr(device_mod, "resolve_device", fake_resolve)
    monkeypatch.setattr(ocr_pkg, "load_ocr", fake_load_ocr)
    monkeypatch.setattr(sfx_mod.SfxReader, "load", fake_reader_load)

    run_ocr(
        OcrRequest(dst=str(dst), ocr_dir=str(ocr_dir), report_dir=str(tmp_path / "r"))
    )
    assert probes == [None]
    assert given == ["cuda:7", "cuda:7"]


def test_the_ocr_request_declares_the_device_flag_like_every_stage():
    import dataclasses

    from anime_tools._device import DEVICE_HELP
    from anime_tools.stages.requests import OcrRequest

    device = next(f for f in dataclasses.fields(OcrRequest) if f.name == "device")
    assert device.metadata["help"] == DEVICE_HELP and device.default is None


def test_an_explicit_device_is_taken_as_given():
    """``--device`` set means no probe at all."""
    from anime_tools._device import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_reading_order_runs_across_a_row_before_down_the_page():
    from anime_tools.ocr.engine import reading_order

    lines = [
        line("bottom", box=(10, 200, 90, 230)),
        line("right", box=(300, 10, 380, 40)),
        line("left", box=(10, 12, 90, 42)),
    ]
    assert [ln.text for ln in reading_order(lines)] == ["left", "right", "bottom"]


# ---- the content floors and reading order ------------------------------
#
# The reader answers one box at a time, and a watermark and a page number are
# boxes too. What reaches the sidecar is decided here, on the strings and their
# boxes, with no weights in sight.


def _col(text: str, x0: int, *, y0: int = 20, y1: int = 120, w: int = 20, score=0.9):
    """One vertical column of a balloon."""
    return line(text, box=(x0, y0, x0 + w, y1), score=score)


def _row(text: str, y0: int, *, x0: int = 10, x1: int = 110, h: int = 20, score=0.9):
    """One horizontal line of a balloon."""
    return line(text, box=(x0, y0, x1, y0 + h), score=score)


def test_min_chars_drops_a_stray_glyph_and_skip_en_drops_the_page_number():
    from anime_tools.ocr._text import keep_line

    assert keep_line("こんにちは", min_chars=3, skip_en=True)
    assert not keep_line("あ", min_chars=3, skip_en=True)
    # Two glyphs is under the floor; whitespace is not a character.
    assert not keep_line("は い", min_chars=3, skip_en=True)
    assert not keep_line("12", min_chars=3, skip_en=True)
    # ASCII goes whatever its length: the page number, the URL, the romaji sfx.
    assert not keep_line("DOKAAAN", min_chars=3, skip_en=True)
    assert not keep_line("pixiv.net/en/users/1", min_chars=3, skip_en=True)
    assert keep_line("DOKAAAN", min_chars=3, skip_en=False)
    # Mixed is not English, and a script the filter was never written for stays.
    assert keep_line("Hello 世界", min_chars=3, skip_en=True)
    assert keep_line("안녕하세요", min_chars=3, skip_en=True)
    # Both floors off is every line.
    assert keep_line("a", min_chars=0, skip_en=False)


def test_a_page_set_in_columns_reads_right_to_left_then_down():
    """Manga columns start at the right edge; two balloons stacked at one x
    read top first. A row page keeps the across-then-down order."""
    from anime_tools.ocr._text import reading_order

    lines = [
        _col("left", 100),
        _col("right-lower", 200, y0=140, y1=240),
        _col("right-upper", 200),
        _col("middle", 150),
    ]
    assert [ln.text for ln in reading_order(lines)] == [
        "right-upper",
        "right-lower",
        "middle",
        "left",
    ]
    # A horizontal sfx on a column page falls in by its right edge.
    sfx = line("sfx", box=(120, 200, 180, 220))
    assert [ln.text for ln in reading_order([_col("a", 100), sfx, _col("b", 200)])] == [
        "b",
        "sfx",
        "a",
    ]


def test_tally_marks_are_a_count_and_never_a_line():
    from anime_tools.ocr._text import is_tally, keep_line

    assert is_tally("正T正正")
    assert is_tally("正正正 正一")
    assert not is_tally("正しい")
    assert not is_tally("TTT")  # no 正 at all: skip_en's job, not this one
    assert not keep_line("正T正正", min_chars=0, skip_en=False)
    assert keep_line("正解です", min_chars=0, skip_en=False)


# ---- attaching the record to a caption ----------------------------------


def test_the_clause_is_attached_after_the_position_clauses():
    from anime_tools.captions.ocr_sidecar import with_ocr_clause

    got = with_ocr_clause(
        "1girl, solo. On the left, cat.", [line("大丈夫"), line("本当に")]
    )
    assert (
        got
        == '1girl, solo. On the left, cat. Japanese text reads as "大丈夫", "本当に".'
    )


def test_combining_again_replaces_the_clause_and_no_lines_removes_it():
    from anime_tools.captions.ocr_sidecar import with_ocr_clause

    once = with_ocr_clause("1girl", [line("旧")])
    assert with_ocr_clause(once, [line("新")]) == '1girl. Japanese text reads as "新".'
    assert with_ocr_clause(once, []) == "1girl"
    assert with_ocr_clause("", [line("孤")]) == 'Japanese text reads as "孤".'


def test_the_attached_line_round_trips_through_the_parser():
    from anime_tools.captions.ocr_sidecar import with_ocr_clause
    from anime_tools.captions.position_clauses import parse_caption

    got = with_ocr_clause("1girl, solo", [line('He said "no", really. On the left')])
    parsed = parse_caption(got)
    assert parsed.flat_tags == ("1girl", "solo")
    (text,) = parsed.text_clauses
    assert text.tags == ('"He said ”no”, really. On the left"',)


def test_sfx_lines_get_their_own_clause_one_per_sound():
    from anime_tools.captions.ocr_sidecar import with_ocr_clause

    lines = [
        line("いいわよモデルになってあげる", det=0.9),
        line("パシャ", seq=2, det=0.9),
        line("パシャ", seq=3, det=0.9),
        line("ぱんぱん", seq=4, det=0.9),
        line("ぴちょっ", seq=5, det=0.9),
    ]
    assert with_ocr_clause("1girl", lines) == (
        '1girl. Japanese text reads as "いいわよモデルになってあげる". '
        'Japanese SFX reads as "パシャ", "ぱんぱん", "ぴちょっ".'
    )
    # a page of nothing but one sound is one SFX clause and no text clause
    assert with_ocr_clause(
        "1girl", [line("ガク", det=0.9), line("ガク", seq=2, det=0.9)]
    ) == ('1girl. Japanese SFX reads as "ガク".')


def test_the_speech_clause_says_a_repeated_line_once():
    from anime_tools.captions.ocr_sidecar import with_ocr_clause

    # 12971620: a page of panting, read as はあ seven times over.
    lines = [
        line("はあ", seq=1, det=0.9),
        line("はあ", seq=2, det=0.9),
        line("バットの使い方", seq=3, det=0.9),
        line("はあ", seq=4, det=0.9),
        line("はぁ", seq=5, det=0.9),
    ]
    assert with_ocr_clause("1girl", lines) == (
        '1girl. Japanese text reads as "はあ", "バットの使い方", "はぁ".'
    )


def test_speech_folds_nothing_the_sfx_key_would():
    from anime_tools.captions.ocr_sfx import dedupe_sfx, dedupe_speech

    # the SFX key would make these one sound; as dialogue they are three lines
    said = ["はっ", "はー", "はっ"]
    assert dedupe_speech(said) == ["はっ", "はー"]
    assert dedupe_sfx(said) == ["はっ"]


def test_the_glyph_floor_drops_a_box_too_small_for_what_was_read():
    from anime_tools.captions.ocr_sidecar import (
        DEFAULT_MIN_GLYPH,
        usable_lines,
        with_ocr_clause,
    )

    assert DEFAULT_MIN_GLYPH == 16.0
    # 13442495: a 39x22 shop sign read as four glyphs — 14.6 px each.
    sign = line("タンKロ", box=(784, 771, 823, 793), det=0.62)
    assert round(sign.glyph_px, 1) == 14.6
    # the same box with two glyphs in it is 20.7 px and stays
    short = line("ん♡", seq=2, box=(0, 0, 39, 22), det=0.62)
    # a 15 px narration strip: read correctly, too fine to train on
    strip = line("気さくな妹の友達", seq=3, box=(0, 0, 102, 16), det=0.65)
    assert usable_lines([sign, short, strip]) == [short]
    assert with_ocr_clause("1girl", [sign, short, strip]) == (
        '1girl. Japanese text reads as "ん♡".'
    )
    # the floor is a parameter, and 0 attaches everything
    assert with_ocr_clause("1girl", [sign, strip], min_glyph=0) == (
        '1girl. Japanese text reads as "気さくな妹の友達". Japanese SFX reads as "タンKロ".'
    )


def test_the_glyph_floor_holds_unscored_lines_too():
    from anime_tools.captions.ocr_sidecar import usable_lines

    # the det floor exempts an unscored line; the glyph floor cannot — its box
    # is as real as any other's.
    tiny = line("契約日", box=(0, 0, 26, 17), det=0.0)
    assert usable_lines([tiny]) == []


def test_the_det_floor_keeps_low_boxes_out_of_the_caption_but_not_unscored_lines():
    from anime_tools.captions.ocr_sidecar import (
        DEFAULT_MIN_DET,
        usable_lines,
        with_ocr_clause,
    )

    assert DEFAULT_MIN_DET == 0.5
    sure = line("大丈夫", det=0.91)
    shaky = line("シャ", seq=2, det=0.38)  # the nested fragment the detector doubled
    unscored = line("本当に", seq=3, det=0.0)  # a mask component, or a pre-det record
    assert usable_lines([sure, shaky, unscored]) == [sure, unscored]
    assert with_ocr_clause("1girl", [sure, shaky, unscored]) == (
        '1girl. Japanese text reads as "大丈夫", "本当に".'
    )
    # the floor is a parameter: 0 attaches everything, 0.95 attaches only the unscored
    assert with_ocr_clause("1girl", [sure, shaky], min_det=0) == (
        '1girl. Japanese text reads as "大丈夫". Japanese SFX reads as "シャ".'
    )
    assert with_ocr_clause("1girl", [sure, shaky, unscored], min_det=0.95) == (
        '1girl. Japanese text reads as "本当に".'
    )
    # every usable line gone → the clause goes too
    assert with_ocr_clause('1girl. Japanese text reads as "旧".', [shaky]) == "1girl"
