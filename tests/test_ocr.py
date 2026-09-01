"""OCR: the sidecar, the engine's weights-free pieces, and what a run writes.

A run writes the recognized text to ``{stem}.ocr.txt`` in the OCR tree and reads
or writes no caption. Nothing here loads a model: the stage takes its reader as
an argument, and the CTC decode and the shape-holding tricks run against
hand-built arrays.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_every_recognized_line_reaches_the_sidecar_numbered_in_order():
    """There is no language filter: the score floor is the only one, and every
    line clearing it is renumbered in order."""
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
    """``read_many_fn`` is a pure substitution for ``read_fn``."""
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
        read_many_fn=lambda paths: [list(lines) for _ in paths],
        apply=False,
    )
    assert [r.to_row() for r in one] == [r.to_row() for r in many]
    assert stats_one.lines == stats_many.lines == 3


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
