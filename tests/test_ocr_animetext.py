"""The AnimeText text-block detector (``anime_tools.ocr.animetext``) without
its weights: the head decode and NMS, the nesting policy, the letterbox, the
engine's detect-only path over a fake detector, the reread seam over its
empty lines, the catalog row, the request, and the bounded CUDA arena every
ONNX session gets. Nothing here loads a model."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from anime_tools import downloads as DL
from anime_tools.captions.ocr_sidecar import OcrLine
from anime_tools.ocr import animetext, reread
from anime_tools.stages.requests import OcrRequest


def test_the_module_is_torch_free():
    code = (
        "import sys, anime_tools.ocr.animetext as a; a.denest([(0,0,1,1)]); "
        "a.containment((0,0,1,1),(0,0,2,2)); print('torch' in sys.modules)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


# ---- letterbox ------------------------------------------------------------


def test_the_letterbox_scales_the_long_side_and_pads_top_left():
    pytest.importorskip("cv2")
    page = np.zeros((1200, 800, 3), np.uint8)
    canvas, r = animetext.letterbox(page, 640)
    assert canvas.shape == (640, 640, 3)
    assert r == pytest.approx(640 / 1200)
    # the image sits top-left; the rest is the grey pad
    assert canvas[0, 0].tolist() == [0, 0, 0]
    assert canvas[0, 639].tolist() == [animetext.PAD_VALUE] * 3


def test_a_native_letterbox_pads_to_a_multiple_of_32_at_scale_one():
    pytest.importorskip("cv2")
    canvas, r = animetext.letterbox(np.zeros((1201, 799, 3), np.uint8), 0)
    assert r == 1.0 and canvas.shape == (1216, 800, 3)


def test_the_tensor_is_rgb_0_1_nchw():
    canvas = np.zeros((4, 4, 3), np.uint8)
    canvas[:, :, 0] = 255  # blue in BGR
    x = animetext.to_tensor(canvas)
    assert x.shape == (1, 3, 4, 4) and x.dtype == np.float32
    assert x[0, 2].max() == 1.0 and x[0, 0].max() == 0.0  # blue is the last RGB plane


# ---- decode + NMS ---------------------------------------------------------


def _head(rows):
    """``[(cx, cy, w, h, score), …]`` → the export's ``(5, N)``."""
    return np.array(rows, dtype=np.float32).T


def test_decode_keeps_the_best_of_overlapping_anchors_and_drops_the_floor():
    pytest.importorskip("cv2")
    head = _head(
        [
            (100, 100, 40, 40, 0.9),  # kept
            (102, 101, 40, 40, 0.8),  # the same box, suppressed
            (300, 300, 20, 60, 0.5),  # kept
            (500, 500, 20, 20, 0.1),  # under the floor
        ]
    )
    boxes = animetext.decode(head, 1.0, 640, 640, conf=0.25, nms=0.5)
    assert [b[:4] for b in boxes] == [(80, 80, 120, 120), (290, 270, 310, 330)]
    assert [round(b[4], 2) for b in boxes] == [0.9, 0.5]


def test_decode_maps_through_the_letterbox_scale_and_clips_to_the_image():
    pytest.importorskip("cv2")
    # A 1280-wide page letterboxed to 640: r = 0.5; the anchor sits at the edge.
    head = _head([(10, 10, 40, 40, 0.9)])
    boxes = animetext.decode(head, 0.5, 1280, 1280, conf=0.25, nms=0.5)
    assert boxes == [(0, 0, 60, 60, pytest.approx(0.9))]


def test_an_empty_head_decodes_to_nothing():
    pytest.importorskip("cv2")
    assert animetext.decode(_head([(1, 1, 1, 1, 0.01)]), 1.0, 10, 10) == []
    assert animetext.decode(np.zeros((5, 0), np.float32), 1.0, 10, 10) == []


# ---- nesting --------------------------------------------------------------

BLOCK = (100, 100, 300, 400)
COL_A = (110, 110, 170, 390)
COL_B = (200, 110, 260, 390)
ELSEWHERE = (500, 500, 540, 600)


def test_inner_drops_the_block_holding_its_columns():
    boxes = [BLOCK, COL_A, COL_B, ELSEWHERE]
    assert animetext.denest(boxes, "inner") == [COL_A, COL_B, ELSEWHERE]


def test_outer_drops_the_columns_inside_the_block():
    assert animetext.denest([BLOCK, COL_A, COL_B, ELSEWHERE], "outer") == [
        BLOCK,
        ELSEWHERE,
    ]


def test_raw_keeps_both_and_a_block_holding_one_box_is_not_nested():
    boxes = [BLOCK, COL_A, COL_B]
    assert animetext.denest(boxes, "raw") == boxes
    # One column inside a block is a detector doubling a line, not a balloon:
    # `inner` needs two.
    assert animetext.denest([BLOCK, COL_A], "inner") == [BLOCK, COL_A]


def test_denest_keeps_a_trailing_score_and_the_order():
    scored = [(*BLOCK, 0.9), (*COL_A, 0.8), (*COL_B, 0.7)]
    assert animetext.denest(scored, "inner") == scored[1:]
    with pytest.raises(ValueError, match="nest policy"):
        animetext.denest(scored, "middle")


def test_containment_is_the_share_of_the_inner_box():
    assert animetext.containment(COL_A, BLOCK) == 1.0
    assert animetext.containment(BLOCK, COL_A) == pytest.approx(
        (60 * 280) / (200 * 300)
    )
    assert animetext.containment(ELSEWHERE, BLOCK) == 0.0


def test_as_quad_is_the_engine_shape():
    q = animetext.as_quad((10, 20, 30, 40))
    assert q.shape == (4, 2) and q.dtype == np.float32
    assert q.tolist() == [[10, 20], [30, 20], [30, 40], [10, 40]]


# ---- the detector over a fake session ------------------------------------


class _Session:
    """Answers one anchor per call — a box at the canvas centre — and records
    every input shape it was fed."""

    def __init__(self):
        self.shapes = []

    def get_inputs(self):
        return [types.SimpleNamespace(name="images")]

    def run(self, _outputs, feeds):
        x = feeds["images"]
        self.shapes.append(x.shape)
        s = x.shape[-1]
        return [_head([(s / 2, s / 2, s / 4, s / 4, 0.9)])[None]]


def test_the_detector_maps_the_canvas_box_back_onto_the_page():
    pytest.importorskip("cv2")
    sess = _Session()
    det = animetext.AnimeTextDetector(session=sess, imgsz=640)
    page = np.zeros((1280, 1280, 3), np.uint8)  # r = 0.5: canvas (240..400)² → page
    assert det.detect(page) == [(480, 480, 800, 800)]
    assert det.detect_scored(page)[0][4] == pytest.approx(0.9)
    assert sess.shapes == [(1, 3, 640, 640)] * 2  # one forward per call


def test_every_page_reaches_the_session_at_one_shape():
    pytest.importorskip("cv2")
    sess = _Session()
    det = animetext.AnimeTextDetector(session=sess, imgsz=640)
    det.detect_batch(
        [
            np.zeros((h, w, 3), np.uint8)
            for h, w in [(1200, 880), (600, 600), (300, 900)]
        ]
    )
    assert set(sess.shapes) == {(1, 3, 640, 640)}


def test_the_protocol_halves_agree_with_detect():
    pytest.importorskip("cv2")
    det = animetext.AnimeTextDetector(session=_Session(), imgsz=640)
    page = np.zeros((640, 640, 3), np.uint8)
    prepared = det.prepare(page)
    raw = det.forward_batch([prepared])[0]
    pairs = det.boxes(raw, prepared, page.shape[:2])
    assert len(pairs) == 1
    quad, score = pairs[0]
    assert quad.shape == (4, 2)
    assert [int(v) for v in (quad[:, 0].min(), quad[:, 1].min())] == [240, 240]
    # the pair's score is the same one detect_scored reports
    assert score == pytest.approx(det.detect_scored(page)[0][4])


# ---- the detect-only engine -----------------------------------------------


class _Fake:
    """A detector that answers fixed quads per image, keyed on its height."""

    def __init__(self, by_height):
        self.by_height = by_height

    def prepare(self, bgr):
        return bgr.shape[0]

    def forward_batch(self, prepared):
        return list(prepared)

    def boxes(self, raw, prepared, shape):
        # (quad, score) pairs: the score is the box's own, kept as ``det``
        return [
            (animetext.as_quad(b), 0.5 + 0.01 * k)
            for k, b in enumerate(self.by_height.get(raw, []))
        ]


def test_the_engine_emits_every_box_as_an_empty_line_in_reading_order(
    tmp_path: Path,
):
    from PIL import Image

    from anime_tools.ocr._onnx import NO_TEXT_SCORE, OcrEngine

    for name, size in (("a.png", (200, 300)), ("b.png", (200, 100))):
        Image.new("RGB", size, "white").save(tmp_path / name)
    engine = OcrEngine(
        detector=_Fake({300: [(10, 10, 40, 200), (100, 10, 130, 200)], 100: []}),
        min_box_px=12,
    )
    pages = engine.read_many([tmp_path / "a.png", tmp_path / "b.png"])
    # two vertical columns read right to left; b has nothing
    assert [[ln.box for ln in pg] for pg in pages] == [
        [(100, 10, 130, 200), (10, 10, 40, 200)],
        [],
    ]
    assert all(ln.text == "" and ln.score == NO_TEXT_SCORE for ln in pages[0])
    assert [ln.seq for ln in pages[0]] == [1, 2]
    # the detector's confidence rides on each line, following its box through
    # the reorder (the second box was scored 0.51 and now reads first)
    assert [ln.det for ln in pages[0]] == pytest.approx([0.51, 0.50])


def test_the_engine_still_applies_the_size_filters(tmp_path: Path):
    from PIL import Image

    from anime_tools.ocr._onnx import OcrEngine

    Image.new("RGB", (300, 200), "white").save(tmp_path / "a.png")
    engine = OcrEngine(
        detector=_Fake({200: [(0, 0, 5, 5), (10, 10, 40, 100), (50, 10, 90, 100)]}),
        min_box_px=12,
        max_boxes=1,
    )
    (page,) = engine.read_many([tmp_path / "a.png"])
    # the 5 px box never becomes a line; max_boxes keeps the largest
    assert [ln.box for ln in page] == [(50, 10, 90, 100)]


# ---- the reread seam on lines with no text --------------------------------


def _page(h=400, w=300):
    return np.full((h, w, 3), 255, np.uint8)


def _empty(box):
    return OcrLine(seq=0, box=box, score=0.0, text="")


def test_an_empty_line_lives_by_its_read():
    lines = [
        _empty((10, 10, 40, 200)),
        _empty((60, 10, 90, 200)),
        _empty((200, 300, 260, 340)),
    ]
    reads = {0: "ぱんぱん", 1: None, 2: "♡"}

    def read_boxes(bgr, boxes):
        return [reads[i] for i in range(len(boxes))]

    out = reread.reread_lines(_page(), lines, read_boxes, min_chars=2)
    # the guard-rejected box and the decoration are gone; the read is the line
    assert [(ln.text, ln.box) for ln in out] == [("ぱんぱん", (10, 10, 40, 200))]
    assert out[0].seq == 1


# ---- the catalog row and the loader ----------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIME_TOOLS_HOME", str(tmp_path))
    monkeypatch.delenv("ANIME_TOOLS_MODELS", raising=False)
    return tmp_path


def test_the_row_lands_where_the_loader_looks_and_names_the_stage(home):
    row = DL.by_id()["animetext_det"]
    assert row.dest == DL.default_animetext_dir() == home / "models" / "animetext"
    assert row.subfolder == "yolo12l_animetext"
    assert DL.ANIMETEXT_ONNX in row.files and "threshold.json" in row.files
    # The stage's one detector, so the stage bar warns before a run; the VL
    # reader's two rows are the stage's too, and all three sit in the ocr pack.
    assert row.stages == ("ocr",) and row.pack == "ocr"
    assert {a.id for a in DL.by_pack()["ocr"]} == {
        "animetext_det",
        "vl16_base",
        "sfx_reader",
    }
    assert all(a.stages == ("ocr",) for a in DL.by_pack()["ocr"])
    assert "GPL" in row.notes


def test_load_without_weights_names_the_row(home):
    with pytest.raises(animetext.AnimeTextWeightsMissing, match="animetext_det"):
        animetext.AnimeTextDetector.load(fetch=False)
    with pytest.raises(animetext.AnimeTextWeightsMissing, match=str(home)):
        animetext.AnimeTextDetector.load(home / "elsewhere")
    with pytest.raises(ValueError, match="nest policy"):
        animetext.AnimeTextDetector.load(nest="middle", fetch=False)


def test_load_ocr_builds_the_detect_only_engine_over_the_animetext_detector(
    home, monkeypatch
):
    """The one entry point: no detector or recognizer to choose any more."""
    from anime_tools.ocr import OcrEngine, load_ocr

    built = {}

    def fake_load(cls, model_dir=None, *, device, conf, nest, **_):
        built.update(device=device, conf=conf, nest=nest)
        return cls(session=object())

    monkeypatch.setattr(animetext.AnimeTextDetector, "load", classmethod(fake_load))
    engine = load_ocr(device="cpu", det_conf=0.4, max_boxes=8)
    assert isinstance(engine, OcrEngine) and engine.max_boxes == 8
    assert built == {"device": "cpu", "conf": 0.4, "nest": "inner"}
    assert not hasattr(engine, "recognizer")
    load_ocr(device="cpu")
    assert built["conf"] == animetext.DEFAULT_CONF


# ---- the request ----------------------------------------------------------


def test_the_request_has_one_path_and_round_trips_its_knobs():
    # PP-OCRv6 was retired 2026-09-07: no detector / reader / join / score-floor /
    # recognizer-batch / detector-side flags; the AnimeText detector's score
    # floor is the one detector knob.
    req = OcrRequest()
    assert req.det_conf == 0.25
    assert "--det_conf" not in req.to_argv()
    for gone in ("detector", "reader", "min_score", "batch_size", "det_limit_side"):
        assert not hasattr(req, gone), gone
    req = OcrRequest(det_conf=0.426, mask_dir="m", max_boxes=8)
    assert "--det_conf" in req.to_argv() and "--mask_dir" in req.to_argv()
    assert OcrRequest.from_argv(OcrRequest.parser(), req.to_argv()) == req


def test_the_request_refuses_bad_values():
    with pytest.raises(ValueError, match="--det_conf"):
        OcrRequest(det_conf=1.5)
    with pytest.raises(ValueError, match="--min_chars"):
        OcrRequest(min_chars=-1)


# ---- the bounded CUDA arena -----------------------------------------------


def test_every_gpu_session_gets_the_bounded_arena(monkeypatch, tmp_path):
    from anime_tools import _onnx

    seen = {}

    class Ort:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def preload_dlls():
            pass

        @staticmethod
        def InferenceSession(path, providers):
            seen["providers"] = providers
            return "session"

    monkeypatch.setitem(sys.modules, "onnxruntime", Ort)
    monkeypatch.setenv("ANIME_TOOLS_ORT_GPU_MEM_GB", "2")
    assert _onnx.make_session(tmp_path / "m.onnx", "cuda", what="x") == "session"
    cuda, cpu = seen["providers"]
    assert cpu == "CPUExecutionProvider"
    assert cuda[0] == "CUDAExecutionProvider"
    assert cuda[1] == {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
        "gpu_mem_limit": 2 * 1024**3,
    }
    # The CPU build gets a plain provider list, as before.
    seen.clear()
    _onnx.make_session(tmp_path / "m.onnx", "cpu", what="x")
    assert seen["providers"] == ["CPUExecutionProvider"]


def test_the_detector_answers_scored_pairs_and_the_stage_floor_admits_two_glyphs():
    from anime_tools.stages.requests import OcrRequest

    fake = _Fake({100: [(1, 2, 3, 4)]})
    ((quad, score),) = fake.boxes(100, None, (100, 100))
    assert quad.shape == (4, 2) and score == pytest.approx(0.5)
    # ドン / ゴゴ are SFX, not screentone: the default floor keeps them
    assert OcrRequest().min_chars == 2
