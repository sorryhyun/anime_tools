"""The vendored YOLO12 graph (``anime_tools.vision.yolo12``) without its weights.

The detector this backs was an onnxruntime session until 2026-09-09. What can be
checked offline is the shape of the graph and the checkpoint contract: the layer
table matches ``yolov12l.yaml``, the head answers ``(B, 4 + nc, N)`` with N the
three levels' cells, the anchors are the cell centres at their strides, the
BatchNorm epsilon is ultralytics' and not torch's, and an ultralytics ``.pt``
loads without ultralytics installed.

The numbers themselves were checked against the ONNX graph this replaced, layer by
layer, at the swap: every ``model.N`` within 1e-04 and the head within 1.2e-03 on a
box coordinate in canvas pixels. That needs both a 106 MB graph and onnxruntime, so
it is not pinned here.
"""

from __future__ import annotations

import pickle
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from anime_tools.vision import yolo12 as Y


def test_the_layer_table_is_the_l_scale_of_the_yaml():
    """22 layers, the yaml's ``from`` wiring, and the three Detect inputs."""
    assert len(Y.LAYERS) == 22
    srcs = [src for src, _, _ in Y.LAYERS]
    # The four concats and their yaml partners; everything else is sequential.
    assert srcs[10] == (-1, 6)
    assert srcs[13] == (-1, 4)
    assert srcs[16] == (-1, 11)
    assert srcs[19] == (-1, 8)
    assert srcs[21] == (14, 17, 20)
    assert all(s == -1 for i, s in enumerate(srcs) if i not in {10, 13, 16, 19, 21})
    # Scale `l` caps channels at 512, so the yaml's 1024s land as 512.
    assert Y.LAYERS[8][2]["c2"] == 512
    # ...except where a Concat feeds one, which is a sum of two 512s.
    assert Y.LAYERS[11][2]["c1"] == 1024


def test_the_bn_epsilon_is_ultralytics_and_not_torchs():
    """1e-3 is folded into the published weights; torch's 1e-5 moves the first
    conv's activations by 271."""
    assert Y.BN_EPS == 1e-3
    assert Y.Conv(3, 8, 3).bn.eps == 1e-3


def test_attention_blocks_carry_the_residual_gamma_and_c3k_bodies_do_not():
    model = Y.Yolo12(nc=1)
    assert model.model[6].gamma is not None  # a2=True at the l scale
    assert model.model[11].gamma is None  # a2=False -> C3k bodies
    assert isinstance(model.model[11].m[0], Y.C3k)
    assert isinstance(model.model[6].m[0][0], Y.ABlock)
    # mlp_ratio 1.2 on 256 channels is the checkpoint's 307-wide MLP.
    assert model.model[6].m[0][0].mlp[0].conv.out_channels == 307
    # Area attention: 4 bands in the backbone's first stage, 1 in the second.
    assert model.model[6].m[0][0].attn.area == 4
    assert model.model[8].m[0][0].attn.area == 1


def test_the_head_answers_one_row_per_cell_across_the_three_strides():
    model = Y.Yolo12(nc=1)
    with torch.inference_mode():
        out = model(torch.zeros(2, 3, 64, 64))
    cells = sum((64 // int(s)) ** 2 for s in Y.STRIDES)
    assert out.shape == (2, 5, cells)  # cx cy w h + 1 class


def test_anchors_are_cell_centres_scaled_by_their_stride():
    feats = [torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2)]
    points, strides = Y.make_anchors(feats, Y.STRIDES)
    assert points.shape == (2, 64 + 16 + 4) and strides.shape == (1, 84)
    # First cell of each level is the half-cell offset; the stride column says
    # which level it came from.
    assert points[:, 0].tolist() == [0.5, 0.5]
    assert strides[0, 0] == 8 and strides[0, 64] == 16 and strides[0, 80] == 32


def test_dfl_reads_a_distribution_as_its_expected_bin():
    """All the mass on bin 3 is a distance of 3."""
    dfl = Y.DFL(Y.REG_MAX)
    x = torch.full((1, 4 * Y.REG_MAX, 2), -30.0)
    x[:, 3 :: Y.REG_MAX, :] = 30.0
    assert torch.allclose(dfl(x), torch.full((1, 4, 2), 3.0), atol=1e-4)


def test_an_ultralytics_checkpoint_loads_without_ultralytics(tmp_path):
    """The unpickler answers ``ultralytics.*`` itself, so nothing of it is
    imported — the package is AGPL-3.0 and this one is MIT."""

    class _Fake(torch.nn.Module):
        """Stands in for ``DetectionModel``, pickled by its module path."""

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)

    _Fake.__module__ = "ultralytics.nn.tasks"
    _Fake.__qualname__ = "DetectionModel"
    sys.modules.setdefault("ultralytics", type(sys)("ultralytics"))
    sys.modules.setdefault("ultralytics.nn", type(sys)("ultralytics.nn"))
    mod = type(sys)("ultralytics.nn.tasks")
    mod.DetectionModel = _Fake
    sys.modules["ultralytics.nn.tasks"] = mod
    ckpt = tmp_path / "model.pt"
    try:
        torch.save({"model": _Fake().half()}, ckpt)
    finally:
        for name in ("ultralytics.nn.tasks", "ultralytics.nn", "ultralytics"):
            sys.modules.pop(name, None)

    sd = Y.load_ultralytics_state_dict(ckpt)
    assert set(sd) == {"conv.weight", "conv.bias"}
    # Read back as float32 whatever the checkpoint stored (ultralytics saves fp16).
    assert all(v.dtype is torch.float32 for v in sd.values())
    assert "ultralytics" not in sys.modules


def test_loading_refuses_a_checkpoint_that_is_not_this_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(Y, "load_ultralytics_state_dict", lambda _p: {"nope": None})
    with pytest.raises(RuntimeError, match="not a YOLO12-l detection checkpoint"):
        Y.load_yolo12(tmp_path / "model.pt")


def test_reading_a_checkpoint_never_imports_ultralytics_even_when_installed():
    """A real ultralytics on the machine must not change what this reads.

    The unpickler matches on the module *name*, before any import is attempted, so
    the stub answers whether or not the package exists. A subprocess with a decoy
    ``ultralytics`` on ``sys.modules`` proves the decoy is never consulted.
    """
    code = (
        "import sys, types, torch, pickle;"
        "decoy = types.ModuleType('ultralytics');"
        "decoy.__getattr__ = lambda n: (_ for _ in ()).throw(AssertionError('touched'));"
        "sys.modules['ultralytics'] = decoy;"
        "from anime_tools.vision.yolo12 import load_ultralytics_state_dict as L;"
        "print(L.__module__)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "anime_tools.vision.yolo12"
    assert pickle  # the unpickler is stdlib pickle, not dill or torch's own
