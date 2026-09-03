"""The onnxruntime dbv4 backend and how a checkpoint picks it — offline.

No ONNX graph is built here (the backbone is gated and the export needs the
``export`` dependency group); the session is a stub returning canned arrays, which
is enough to pin everything this side owns.

1. ``dbv4_onnx_path`` is the one spelling of the graph's location.
2. Backend selection: ``auto`` follows the file, ``onnx`` / ``torch`` pin it, and
   ``$ANIMA_TAGGER_BACKEND`` answers for the call sites that take no argument.
3. :class:`Dbv4OnnxBackend` feeds the session float32 ``[0, 1]`` pixels under the
   exported input name and hands back the same :class:`Dbv4Output` the torch
   backend does.
4. A graph that is not ours fails at load — wrong IO names, or a class count that
   disagrees with the card — rather than at the first predict.
5. A missing graph names the command that builds it.
6. ``default_dtype`` is per device: bf16 only where there are kernels for it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from test_tagger_dbv4_backend import DBV4_NAMES, FakeBackend, _card, _write_ckpt

from anime_tools.tagger import dbv4_backend as db
from anime_tools.tagger import dbv4_onnx as dx
from anime_tools.tagger import tagger as at
from anime_tools.tagger.dbv4_meta import DBV4_ONNX_NAME, dbv4_onnx_path

N_CLASSES = len(DBV4_NAMES)
D_HIDDEN = 8


class FakeSession:
    """The two calls :class:`Dbv4OnnxBackend` makes on a session, plus a record
    of what it was fed."""

    def __init__(self, inputs=(dx.INPUT_NAME,), outputs=dx.OUTPUT_NAMES, n=N_CLASSES):
        self._in = [_IO(name, [None, 3, None, None]) for name in inputs]
        self._out = [_IO(outputs[0], [None, n]), _IO(outputs[1], [None, D_HIDDEN])]
        self.fed: dict[str, np.ndarray] = {}

    def get_inputs(self):
        return self._in

    def get_outputs(self):
        return self._out

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _outputs, feed):
        self.fed = feed
        b = feed[dx.INPUT_NAME].shape[0]
        return (
            np.full((b, N_CLASSES), 0.25, dtype=np.float32),
            np.arange(b * D_HIDDEN, dtype=np.float32).reshape(b, D_HIDDEN),
        )


class _IO:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


@pytest.fixture
def onnx_backend(monkeypatch):
    """A backend whose session is a stub and whose card needs no download."""

    def _build(session=None, **kwargs):
        session = session or FakeSession()
        monkeypatch.setattr(dx, "make_session", lambda *a, **kw: session)
        backend = dx.Dbv4OnnxBackend(
            "/nonexistent/dbv4.onnx", card=_card(), device="cpu", **kwargs
        )
        # The path check runs before make_session; the stub replaces the session,
        # not the probe, so short-circuit it.
        monkeypatch.setattr(dx.Path, "is_file", lambda self: True)
        return backend, session

    return _build


# --------------------------------------------------------------------------- #
# 1. where the graph lives
# --------------------------------------------------------------------------- #


def test_onnx_path_is_beside_the_checkpoint(tmp_path):
    assert dbv4_onnx_path(tmp_path) == tmp_path / DBV4_ONNX_NAME
    assert DBV4_ONNX_NAME.endswith(".onnx")


# --------------------------------------------------------------------------- #
# 2. selection
# --------------------------------------------------------------------------- #


def _tagger(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(at, "Dbv4Backend", lambda **kw: FakeBackend({"1girl": 0.9}))
    monkeypatch.setattr(
        at, "Dbv4OnnxBackend", lambda *a, **kw: FakeBackend({"1girl": 0.9})
    )
    return at.AnimaTagger(tmp_path, device="cpu", **kwargs)


def test_auto_picks_torch_without_a_graph(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    tagger = _tagger(tmp_path, monkeypatch)
    assert tagger._backend_choice == "auto"
    assert isinstance(tagger._dbv4, FakeBackend)  # the patched Dbv4Backend


def test_auto_picks_onnx_when_the_graph_is_there(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    dbv4_onnx_path(tmp_path).write_bytes(b"not really a graph")
    seen: list[tuple] = []
    monkeypatch.setattr(at, "Dbv4Backend", lambda **kw: FakeBackend({}))
    monkeypatch.setattr(
        at,
        "Dbv4OnnxBackend",
        lambda *a, **kw: (seen.append((a, kw)), FakeBackend({"1girl": 0.9}))[1],
    )
    at.AnimaTagger(tmp_path, device="cpu")
    ((args, kwargs),) = seen
    assert args[0] == dbv4_onnx_path(tmp_path)
    assert kwargs["repo"] == "fake/dbv4"


def test_explicit_torch_ignores_the_graph(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    dbv4_onnx_path(tmp_path).write_bytes(b"not really a graph")
    called: list[str] = []
    monkeypatch.setattr(
        at, "Dbv4Backend", lambda **kw: (called.append("torch"), FakeBackend({}))[1]
    )
    monkeypatch.setattr(
        at,
        "Dbv4OnnxBackend",
        lambda *a, **kw: pytest.fail("backend='torch' built the onnx backend"),
    )
    at.AnimaTagger(tmp_path, device="cpu", backend="torch")
    assert called == ["torch"]


def test_env_var_answers_for_call_sites_without_a_flag(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    dbv4_onnx_path(tmp_path).write_bytes(b"not really a graph")
    monkeypatch.setenv(at.BACKEND_ENV, "torch")
    monkeypatch.setattr(
        at,
        "Dbv4OnnxBackend",
        lambda *a, **kw: pytest.fail("$ANIMA_TAGGER_BACKEND=torch was ignored"),
    )
    tagger = _tagger(tmp_path, monkeypatch)
    assert tagger._backend_choice == "torch"
    assert tagger.dbv4_runtime == "torch"


def test_an_unknown_backend_is_a_valueerror(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    with pytest.raises(ValueError, match="backend must be one of"):
        _tagger(tmp_path, monkeypatch, backend="tensorrt")


# --------------------------------------------------------------------------- #
# 3. the forward contract
# --------------------------------------------------------------------------- #


def test_forward_tensor_feeds_float32_and_returns_torch(onnx_backend):
    backend, session = onnx_backend()
    x = torch.rand(2, 3, 8, 8, dtype=torch.float64)
    out = backend.forward_tensor(x)
    fed = session.fed[dx.INPUT_NAME]
    assert fed.dtype == np.float32 and fed.shape == (2, 3, 8, 8)
    assert isinstance(out.probs, torch.Tensor) and out.probs.dtype is torch.float32
    assert out.probs.shape == (2, N_CLASSES)
    assert out.hidden.shape == (2, D_HIDDEN)


def test_forward_preprocesses_like_the_torch_backend(onnx_backend):
    backend, session = onnx_backend(img_size=16)
    im = Image.new("RGB", (8, 32), (10, 20, 30))
    backend.forward([im])
    fed = session.fed[dx.INPUT_NAME]
    expected = db.preprocess_dbv4(im, 16).unsqueeze(0).numpy()
    # Same square pad + resize + [0, 1] scaling; normalisation is in the graph.
    assert np.allclose(fed, expected)
    assert 0.0 <= fed.min() and fed.max() <= 1.0


def test_d_hidden_is_read_off_the_graph(onnx_backend):
    backend, _ = onnx_backend()
    assert backend.d_hidden == D_HIDDEN


# --------------------------------------------------------------------------- #
# 4. a graph that is not ours
# --------------------------------------------------------------------------- #


def test_wrong_io_names_fail_at_load(onnx_backend):
    backend, _ = onnx_backend(session=FakeSession(inputs=("image",)))
    with pytest.raises(RuntimeError, match="not an Anima dbv4 graph"):
        backend.forward_tensor(torch.rand(1, 3, 8, 8))


def test_class_count_must_match_the_card(onnx_backend):
    backend, _ = onnx_backend(session=FakeSession(n=N_CLASSES + 1))
    with pytest.raises(RuntimeError, match="exported against a different backbone"):
        backend.forward_tensor(torch.rand(1, 3, 8, 8))


# --------------------------------------------------------------------------- #
# 5 & 6. missing graph, and the dtype rule
# --------------------------------------------------------------------------- #


def test_a_missing_graph_names_the_export_command(tmp_path):
    backend = dx.Dbv4OnnxBackend(tmp_path / DBV4_ONNX_NAME, card=_card(), device="cpu")
    with pytest.raises(FileNotFoundError, match="tagger.cli.export_onnx"):
        backend.forward_tensor(torch.rand(1, 3, 8, 8))


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", torch.float32), ("cuda", torch.bfloat16), ("cuda:1", torch.bfloat16)],
)
def test_default_dtype_is_per_device(device, expected):
    assert db.default_dtype(device) is expected
