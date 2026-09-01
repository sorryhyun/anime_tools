"""OCR: the sidecar, the engine's weights-free pieces, and what a run writes.

The split these pin is the whole design. A run answers **one** question and
writes **one** place: the recognized text goes into ``{stem}.ocr.txt`` in the OCR
tree, and no caption is read or written. The stage used to also infer a Danbooru
script tag from a guessed language and append it to the caption, and later to
drop a line whose guessed language was not asked for; both halves are gone, and
``test_the_stage_never_touches_a_caption`` plus
``test_every_recognized_line_reaches_the_sidecar_numbered_in_order`` are what
keep them gone.

None of this loads a model. The sidecar is stdlib, the stage takes its reader as
an argument, and the CTC decode and the two shape-holding tricks are exercised
against hand-built arrays — so the suite stays CPU-only and needs no 139 MB
download.
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
    """The OCR tree joins to the dataset by the same relative path every other
    root does, so a nested artist folder has to be created rather than assumed."""
    p = write_ocr_for(tmp_path, Path("artist/a.txt"), [line("SALE")])
    assert p == tmp_path / "artist" / ("a" + OCR_SIDECAR_SUFFIX)
    assert read_ocr(p)[0].text == "SALE"


def test_no_text_deletes_the_sidecar_rather_than_writing_an_empty_one(tmp_path: Path):
    p = write_ocr_for(tmp_path, Path("a.txt"), [line("SALE")])
    assert p.is_file()
    # A re-run over re-cropped pixels that no longer hold the text must not leave
    # the old claim standing.
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
    """There is no language filter, and the absence is the point.

    A ``--lang`` allowlist used to drop a line whose script was not asked for,
    with the script guessed back off the characters -- so ``01R`` was English,
    a lone ``心`` was Chinese, and ``!?`` was neither and was deleted. Guessing
    a language is not a way of deciding whether something is text. The score
    floor is the filter; everything that clears it is a readout.
    """
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
    # And nowhere near the resized tree, which is where it used to land.
    assert not (dst / "a.ocr.txt").exists()


def test_the_stage_never_touches_a_caption(tmp_path: Path):
    """The tag half is gone: a run over an English line used to append
    ``english text`` to the caption, and most of what that proposed rested on
    two-character fragments. Nothing in the resized tree may change."""
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
    """Unlike the caption stages: this one has nothing to look a caption up for,
    so having none is not a reason to skip an image."""
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
    """``read_many_fn`` is what lets the engine batch; it must be a pure
    substitution, so the two are run against the same images and compared."""
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
# ONNX Runtime's CUDA provider re-plans on every input-shape change, and a
# bucketed resized tree changes it on every image -- measured at 12 ms of
# detector forward behind 70 ms of re-planning, which was most of a run. Both
# sessions are therefore held at one shape. Neither half of that is visible in
# a result when it works: the padding would show up as phantom boxes along the
# canvas seam, and the batch padding as recognized text landing on the wrong
# crop. Hence pinning them here rather than trusting the timing.


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
    """The whole point: two differently-shaped images must arrive at the session
    as the same tensor shape, and each must still report the region it occupies
    so the pad can be cut back off."""
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
    """The CPU provider has no re-planning penalty, so padding it to a square
    would be pure loss -- `pad_to=0` is what a CPU session gets."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")

    x, _, _, live = _detector(pad_to=0)._preprocess(np.zeros((1200, 880, 3), "uint8"))
    assert x.shape[2:] == live


def test_nothing_found_in_the_padded_band_survives():
    """The replicated border is not image, and a box along the canvas seam is
    the failure mode padding would introduce. `_boxes` crops the map to the live
    region first, so the band cannot be looked at at all."""
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
    """Batch size is an input axis too, so the tail of a run must not shrink it.
    The filler rows have to disappear again before the scatter, or every string
    after the first short batch lands on the wrong crop."""
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
    # Two forwards, both full width and both full batch -- the second held 1 crop.
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
    """A download and a load that spell the same path twice can drift, and the
    symptom is a Download button that appears to do nothing."""
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
