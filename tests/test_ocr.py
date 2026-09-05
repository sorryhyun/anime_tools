"""OCR: the sidecar, the engine's weights-free pieces, and what a run writes.

A run writes the recognized text to ``{stem}.ocr.txt`` in the OCR tree and reads
or writes no caption. Nothing here loads a model: the stage takes its reader as
an argument, and the CTC decode and the shape-holding tricks run against
hand-built arrays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import anime_tools
from anime_tools.captions.ocr_sidecar import (
    OCR_SIDECAR_SUFFIX,
    OcrLine,
    ocr_sidecar_path,
    read_ocr,
    write_ocr_for,
)
from anime_tools.stages.ocr import number_lines, run_ocr


def line(text: str, *, seq: int = 1, score: float = 0.95, box=(0, 0, 40, 20)):
    return OcrLine(seq=seq, box=box, score=score, text=text)


# ---- the sidecar ------------------------------------------------------


def test_the_sidecar_round_trips_through_a_tab_and_a_multi_dot_stem(tmp_path: Path):
    assert ocr_sidecar_path(Path("a/b.c.txt")).name == "b.c" + OCR_SIDECAR_SUFFIX
    # A tab inside the text is why the text field is last.
    lines = [line("こん\tにちは"), line("SALE", seq=2)]
    p = write_ocr_for(tmp_path, Path("a.b.txt"), lines)
    assert p.name == "a.b" + OCR_SIDECAR_SUFFIX
    assert read_ocr(p) == lines


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
        "5\t0,0,10,10\n",
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
    return run_ocr(
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

    one, stats_one = run_ocr(
        resized_dir=dst, ocr_dir=ocr, read_fn=lambda _p: list(lines), apply=False
    )
    many, stats_many = run_ocr(
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

    _, stats = run_ocr(
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

    run_ocr(
        resized_dir=dst,
        ocr_dir=ocr,
        read_fn=lambda _p: [],
        read_iter_fn=reader,
        apply=True,
    )
    # The third image is read only after the first two sidecars are on disk.
    assert seen == [[], ["a.ocr.txt"], ["a.ocr.txt", "b.ocr.txt"]]


# ---- the pieces of the engine that need no weights ---------------------


def test_ctc_decode_collapses_repeats_and_drops_blanks():
    """CTC decode: repeats collapse only across a blank."""
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
    """The vocabulary is sized off the graph's class count: dict + blank, or dict
    + blank + the appended space; anything else raises."""
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
    assert TextRecognizer._vocab(chars, FakeSession(5)) == [
        "<blank>",
        "a",
        "b",
        "c",
        " ",
    ]
    assert TextRecognizer._vocab(chars, FakeSession(4)) == ["<blank>", "a", "b", "c"]
    with pytest.raises(RuntimeError, match="not a pair"):
        TextRecognizer._vocab(chars, FakeSession(9))


def _fake_recognizer(widths: list[int], *, fixed_shape: bool):
    """A recognizer whose session records the width of every batch it is fed."""
    np = pytest.importorskip("numpy")
    from anime_tools.ocr._onnx import TextRecognizer

    class FakeInput:
        name = "x"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, _outputs, feed):
            batch = feed["x"]
            widths.append(batch.shape[-1])
            return [np.zeros((batch.shape[0], 4, 2), dtype="float32")]

    return TextRecognizer(
        session=FakeSession(),
        vocab=["<blank>", "a"],
        batch_size=8,
        fixed_shape=fixed_shape,
    )


def test_a_crop_is_never_padded_past_the_width_it_needs():
    """A batch is drawn from one width rung.

    Padding a crop past its own width is not merely wasted compute — the
    recognizer reads the zeros, and the same crop answers something different at
    320 than at 1280. Batching the aspect-sorted list by position let one wide
    crop drag seven narrow ones up to its width, which made what a line said
    depend on which lines happened to be queued beside it.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    widths: list[int] = []
    rec = _fake_recognizer(widths, fixed_shape=True)
    crops = [np.zeros((48, 48, 3), dtype="uint8") for _ in range(7)]
    crops.append(np.zeros((48, 2000, 3), dtype="uint8"))

    assert len(rec.recognize(crops)) == len(crops)
    # Two batches, not one: the wide crop is alone on its rung.
    assert sorted(widths) == [320, 2240]


def test_the_rung_batching_still_fills_a_batch():
    """Crops of one rung batch together up to ``batch_size``, so the fix costs
    partial batches only at a rung boundary."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    widths: list[int] = []
    rec = _fake_recognizer(widths, fixed_shape=True)
    rec.recognize([np.zeros((48, 60, 3), dtype="uint8") for _ in range(20)])
    assert widths == [320, 320, 320]  # 8 + 8 + 4, all on one rung


def test_a_cpu_batch_pads_to_its_own_need_not_the_rung():
    """``fixed_shape`` off, the batch keeps the exact width its widest member
    needs — the rung is the grouping key, never the tensor width."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    widths: list[int] = []
    rec = _fake_recognizer(widths, fixed_shape=False)
    # 48x400 needs ceil(48 * 400/48) = 400, inside the 640 rung.
    rec.recognize([np.zeros((48, 400, 3), dtype="uint8")])
    assert widths == [400]


def test_resolving_the_ocr_device_never_imports_torch():
    """Asking torch whether there is a GPU costs the run 1.8x.

    The probe initialises CUDA, and torch's context then time-shares the device
    with ORT's for the life of the process — measured on PP-OCRv6 at 23 ms an
    image against 40. This stage runs on onnxruntime, so onnxruntime is what it
    asks. A subprocess, because another test may already have imported torch.
    """
    import subprocess

    pytest.importorskip("onnxruntime")
    code = (
        "import sys;"
        "from anime_tools.ocr import resolve_onnx_device;"
        "d = resolve_onnx_device();"
        "print(d, 'torch' in sys.modules)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
    device, torch_seen = r.stdout.split()
    assert device in {"cuda", "cpu"}
    assert torch_seen == "False"


def test_an_explicit_device_is_taken_as_given():
    """``--device`` set means no probe at all, on either resolver."""
    from anime_tools.ocr import resolve_onnx_device

    assert resolve_onnx_device("cpu") == "cpu"
    assert resolve_onnx_device("cuda") == "cuda"


def test_the_ocr_stage_asks_onnxruntime_not_torch_for_its_device():
    """Pinned in the runner too: the torch resolver is the one every *torch*
    stage uses, and reaching for it here is the whole regression."""
    import inspect

    from anime_tools.stages import run

    src = inspect.getsource(run.run_ocr)
    assert "device = resolve_onnx_device(req.device)" in src
    assert "resolve_device(" not in src
    # The request declares the flag as every stage does — a `device` field
    # carrying `_device.DEVICE_HELP` — and neither it nor the CLI shell touches
    # the torch resolver.
    import dataclasses

    from anime_tools._device import DEVICE_HELP
    from anime_tools.stages.requests import OcrRequest

    device = next(f for f in dataclasses.fields(OcrRequest) if f.name == "device")
    assert device.metadata["help"] == DEVICE_HELP and device.default is None
    cli = (
        Path(anime_tools.__file__).parent / "stages" / "cli" / "ocr_captions.py"
    ).read_text(encoding="utf-8")
    assert "resolve_device" not in cli


# ---- one input shape per session ---------------------------------------
#
# ONNX Runtime's CUDA provider re-plans on every input-shape change (~70 ms
# behind a 12 ms detector forward), and a bucketed resized tree changes the shape
# on every image, so both sessions are held at one shape. When it works nothing
# is visible in the result; when it breaks, padding shows up as phantom boxes on
# the canvas seam and batch padding as text landing on the wrong crop.


def _detector(pad_to: int):
    from anime_tools.ocr._onnx import TextDetector

    return TextDetector(
        session=None,
        thresh=0.3,
        box_thresh=0.5,
        unclip_ratio=1.5,
        max_candidates=100,
        limit_side=960,
        pad_to=pad_to,
    )


def test_the_detector_sees_one_shape_whatever_the_image_was():
    """Differently-shaped images arrive as one tensor shape, each still reporting
    the live region so the pad can be cut back off."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    det = _detector(pad_to=960)
    shapes = set()
    for h, w in [(1200, 880), (1248, 832), (600, 600)]:
        x, sx, sy, live = det._preprocess(np.zeros((h, w, 3), dtype="uint8"))
        shapes.add(x.shape)
        assert live[0] <= x.shape[2] and live[1] <= x.shape[3]
        # The scale factors still map through the live region, not the canvas.
        assert sx == pytest.approx(w / live[1])
        assert sy == pytest.approx(h / live[0])
    assert shapes == {(1, 3, 960, 960)}


def test_a_cpu_detector_keeps_the_image_shape_and_pays_nothing_to_pad():
    """A CPU session gets `pad_to=0`: no re-planning penalty, so padding is loss."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    x, _, _, live = _detector(pad_to=0)._preprocess(np.zeros((1200, 880, 3), "uint8"))
    assert x.shape[2:] == live


def test_nothing_found_in_the_padded_band_survives():
    """`_boxes` crops the map to the live region first, so the replicated border
    cannot produce a box."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    det = _detector(pad_to=960)
    live = (64, 96)
    prob = np.zeros((160, 160), dtype="float32")
    prob[100:130, 110:140] = 1.0  # entirely inside the pad
    assert det._boxes(prob, (64, 96), 1.0, 1.0, live) == []

    prob[10:40, 20:60] = 1.0  # inside the live region
    assert len(det._boxes(prob, (64, 96), 1.0, 1.0, live)) == 1


def test_the_recognizer_quantizes_its_width_but_only_when_it_has_to():
    from anime_tools.ocr._onnx import REC_MIN_WIDTH, TextRecognizer

    fixed = TextRecognizer(session=None, vocab=[], fixed_shape=True)
    native = TextRecognizer(session=None, vocab=[], fixed_shape=False)
    # 48 * 8 = 384, which upstream would feed as-is and a GPU would re-plan for.
    assert fixed._width(8.0) == 2 * REC_MIN_WIDTH
    assert native._width(8.0) == 384
    # A ratio under the floor is the floor either way.
    assert fixed._width(1.0) == native._width(1.0) == REC_MIN_WIDTH


def test_a_short_final_batch_is_padded_out_and_its_filler_dropped():
    """Batch size is an input axis too: the tail is padded out, and the filler
    rows are dropped before the scatter."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    from anime_tools.ocr._onnx import TextRecognizer

    seen: list[tuple] = []

    class FakeSession:
        def get_inputs(self):
            class In:
                name = "x"

            return [In()]

        def run(self, _outputs, feed):
            batch = feed["x"]
            seen.append(batch.shape)
            logits = np.zeros((batch.shape[0], 3, 3), dtype="float32")
            logits[:, :, 1] = 1.0  # every row decodes to "a"
            return [logits]

    rec = TextRecognizer(
        session=FakeSession(),
        vocab=["<blank>", "a", "b"],
        batch_size=4,
        fixed_shape=True,
    )
    crops = [np.zeros((20, 60, 3), dtype="uint8") for _ in range(5)]
    out = rec.recognize(crops)
    assert len(out) == len(crops)
    assert all(text == "a" for text, _ in out)
    # Two forwards, both full width and both full batch; the second held 1 crop.
    assert seen == [(4, 3, 48, 320), (4, 3, 48, 320)]


def test_reading_order_runs_across_a_row_before_down_the_page():
    from anime_tools.ocr._onnx import reading_order

    lines = [
        line("bottom", box=(10, 200, 90, 230)),
        line("right", box=(300, 10, 380, 40)),
        line("left", box=(10, 12, 90, 42)),
    ]
    assert [ln.text for ln in reading_order(lines)] == ["left", "right", "bottom"]


def test_the_loader_and_the_download_catalog_name_one_directory():
    """The catalog row and the loader name one directory."""
    from anime_tools.downloads import (
        by_id,
        default_ppocr_det_dir,
        default_ppocr_rec_dir,
    )

    rows = by_id()
    assert rows["ppocr_det"].dest == default_ppocr_det_dir()
    assert rows["ppocr_rec"].dest == default_ppocr_rec_dir()
    # Two directories, not one: both repos ship a file called inference.onnx.
    assert default_ppocr_det_dir() != default_ppocr_rec_dir()
    for row in (rows["ppocr_det"], rows["ppocr_rec"]):
        assert row.stages == ("ocr",)
        assert set(row.files) == {"inference.onnx", "inference.yml"}


# ---- the content filters and the CJK join ------------------------------
#
# A comic page is where the two models are least like each other: the detector
# finds every balloon column, and the recognizer answers each one as its own
# string. What reaches the sidecar is decided here, on the strings and their
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


def test_a_balloon_of_vertical_columns_joins_right_to_left():
    """Japanese sets its columns right to left, so the rightmost box is the
    start of the sentence."""
    from anime_tools.ocr._text import join_cjk

    joined = join_cjk([_col("げんきです", 170, y1=110), _col("こんにちは", 200)])
    assert [ln.text for ln in joined] == ["こんにちは げんきです"]
    # The record covers the whole balloon.
    assert joined[0].box == (170, 20, 220, 120)


def test_a_balloon_of_horizontal_rows_joins_top_to_bottom():
    from anime_tools.ocr._text import join_cjk

    joined = join_cjk([_row("あるところに", 36, x1=90), _row("むかしむかし", 10)])
    assert [ln.text for ln in joined] == ["むかしむかし あるところに"]


def test_two_balloons_stay_two_records():
    """The gap between columns of one balloon is a fraction of a column; the gap
    to the next balloon is not."""
    from anime_tools.ocr._text import join_cjk

    joined = join_cjk(
        [_col("こんにちは", 200), _col("げんきです", 170), _col("さようなら", 100)]
    )
    assert sorted(ln.text for ln in joined) == ["こんにちは げんきです", "さようなら"]


def test_a_sfx_glyph_does_not_swallow_the_dialogue_beside_it():
    """Comparable thickness is part of being one block: a column three times as
    wide is a different piece of text however close it lands."""
    from anime_tools.ocr._text import join_cjk

    sfx = line("ドン", box=(60, 20, 160, 120))
    joined = join_cjk([sfx, _col("こんにちは", 170)])
    assert sorted(ln.text for ln in joined) == ["こんにちは", "ドン"]


def test_a_column_and_a_row_are_never_one_block():
    """Mixed orientations are a sfx over dialogue, not its continuation."""
    from anime_tools.ocr._text import join_cjk

    joined = join_cjk(
        [_col("こんにちは", 200), _row("むかしむかし", 20, x0=170, x1=270)]
    )
    assert len(joined) == 2


def test_english_lines_never_join():
    """English wraps for width, so two stacked boxes are two lines and stay two."""
    from anime_tools.ocr._text import join_cjk

    joined = join_cjk([_row("ONCE UPON", 10), _row("A TIME", 36, x1=90)])
    assert [ln.text for ln in joined] == ["ONCE UPON", "A TIME"]


def test_the_merged_score_is_the_per_character_mean_of_its_parts():
    """A long confident column is not outvoted by the two glyphs beside it."""
    from anime_tools.ocr._text import join_cjk

    joined = join_cjk(
        [_col("はい", 170, score=0.6), _col("こんにちは", 200, score=1.0)]
    )
    assert joined[0].score == pytest.approx((5 * 1.0 + 2 * 0.6) / 7)


def _engine(**kw):
    from anime_tools.ocr._onnx import OcrEngine

    return OcrEngine(detector=None, recognizer=None, **kw)


def _quad(box):
    np = pytest.importorskip("numpy")

    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32")


def _read(engine, *lines):
    """What `_lines` makes of a set of recognized boxes."""
    return engine._lines(
        [_quad(ln.box) for ln in lines], [(ln.text, ln.score) for ln in lines]
    )


def test_the_join_runs_before_the_floors_so_a_short_column_is_not_lost_first():
    """Order is the whole point: a two-glyph column is only short until the rest
    of its balloon is on it."""
    parts = (_col("はい", 200), _col("そうです", 170))

    joined = _read(_engine(min_chars=3, skip_en=True, join_cjk=True), *parts)
    assert [ln.text for ln in joined] == ["はい そうです"]
    # Without the join each column faces the floor alone, and one of them loses.
    apart = _read(_engine(min_chars=3, skip_en=True, join_cjk=False), *parts)
    assert [ln.text for ln in apart] == ["そうです"]


def test_a_scanned_page_keeps_its_dialogue_and_drops_its_furniture():
    """The defaults over one page: two balloon columns become one line, the page
    number and the watermark go, and what is left is numbered in reading order."""
    got = _read(
        _engine(min_score=0.6, min_chars=3, skip_en=True, join_cjk=True),
        _row("むかしむかし", 10),
        _row("あるところに", 36, x1=90),
        line("12", box=(300, 400, 320, 420)),
        line("pixiv.net/en/users/1", box=(10, 440, 200, 460)),
        line("ぼやけた", box=(10, 300, 90, 320), score=0.2),
    )
    assert [(ln.seq, ln.text) for ln in got] == [(1, "むかしむかし あるところに")]


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


def test_a_vertical_choon_read_as_a_digit_or_bar_is_put_back_after_kana():
    from anime_tools.ocr._text import normalize_ja, normalize_line

    assert (
        normalize_ja("おちんぼの時間だぞ1♡", vertical=True) == "おちんぼの時間だぞー♡"
    )
    assert normalize_ja("新メ|ューで1す", vertical=True) == "新メーューでーす"
    # A real ``ー`` is kana for the next glyph, so a misread after it is fixed;
    # two digits in a row are a number (``あと10``), never a double mark.
    assert normalize_ja("あー1", vertical=True) == "あーー"
    # After a kanji or a digit, ``1`` is a number — and so is one that counts
    # something: a kanji counter, a kana counter, more digits.
    assert normalize_ja("第1話", vertical=True) == "第1話"
    assert normalize_ja("もう1回", vertical=True) == "もう1回"
    assert normalize_ja("エナドリ１か月分", vertical=True) == "エナドリ１か月分"
    assert normalize_ja("あと1つ", vertical=True) == "あと1つ"
    assert normalize_ja("あと10", vertical=True) == "あと10"
    assert normalize_ja("2024年1月", vertical=True) == "2024年1月"
    # Horizontal text spells it as a dash, never a digit.
    assert normalize_ja("だぞ1", vertical=False) == "だぞ1"
    assert normalize_ja("だぞ-", vertical=False) == "だぞー"
    assert normalize_ja("スゴ—イ", vertical=False) == "スゴーイ"
    # ``一`` is a ``ー`` only between katakana, horizontally.
    assert normalize_ja("リ一チ", vertical=False) == "リーチ"
    assert normalize_ja("コーヒー一杯", vertical=False) == "コーヒー一杯"
    assert normalize_ja("の一部", vertical=False) == "の一部"
    assert normalize_ja("リ一チ", vertical=True) == "リ一チ"
    # ``=`` between katakana is ``ニ`` whichever way the line runs; elsewhere
    # it is an equals sign.
    assert normalize_ja("新メ=ューで1す", vertical=True) == "新メニューでーす"
    assert normalize_ja("メ＝ュー", vertical=False) == "メニュー"
    assert normalize_ja("x=1", vertical=False) == "x=1"
    assert normalize_ja("答え=正", vertical=False) == "答え=正"
    # The line form picks the axis off the box and keeps identity when unchanged.
    col = _col("だぞ1", 200)
    assert normalize_line(col).text == "だぞー"
    row = _row("だぞ1", 10)
    assert normalize_line(row) is row


def test_tally_marks_are_a_count_and_never_a_line():
    from anime_tools.ocr._text import is_tally, keep_line

    assert is_tally("正T正正")
    assert is_tally("正正正 正一")
    assert not is_tally("正しい")
    assert not is_tally("TTT")  # no 正 at all: skip_en's job, not this one
    assert not keep_line("正T正正", min_chars=0, skip_en=False)
    assert keep_line("正解です", min_chars=0, skip_en=False)
