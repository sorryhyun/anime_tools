"""``import sam3`` on a platform without triton goes through
``_sam3.stub_edt_kernel``, and a build without CUDA through
``_sam3.shim_sam3_for_cpu``."""

from __future__ import annotations

import importlib.util
import sys

import pytest

from anime_tools.masking import _sam3


def _find_spec_saying(installed: bool):
    real = importlib.util.find_spec

    def find_spec(name, *a, **k):
        if name == "triton":
            return object() if installed else None
        return real(name, *a, **k)

    return find_spec


@pytest.fixture
def no_triton(monkeypatch):
    """A process where triton is not installed and sam3's kernel module is not
    loaded."""
    monkeypatch.delitem(sys.modules, _sam3.EDT_MODULE, raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_saying(False))
    yield
    stub = sys.modules.get(_sam3.EDT_MODULE)
    if getattr(stub, "__anime_tools_stub__", False):
        del sys.modules[_sam3.EDT_MODULE]


def test_the_stand_in_imports_and_refuses_to_run(no_triton):
    assert _sam3.stub_edt_kernel() is True
    assert _sam3.stub_edt_kernel() is False, "installed twice"
    # ``triton`` itself is never faked: torch guards its own import of it.
    assert "triton" not in sys.modules or not getattr(
        sys.modules["triton"], "__anime_tools_stub__", False
    )

    from sam3.model.edt import edt_triton  # what sam3_tracker_utils does

    with pytest.raises(RuntimeError, match="triton is not installed"):
        edt_triton(None)


def test_a_real_triton_is_never_shadowed(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_saying(True))
    monkeypatch.delitem(sys.modules, _sam3.EDT_MODULE, raising=False)
    assert _sam3.stub_edt_kernel() is False
    assert _sam3.EDT_MODULE not in sys.modules


def test_the_cpu_shim_moves_the_build_time_caches_off_cuda(monkeypatch):
    _sam3.stub_edt_kernel()  # a Mac cannot import sam3 without it
    pytest.importorskip("sam3")
    import torch
    from sam3.model.decoder import TransformerDecoder
    from sam3.model.position_encoding import PositionEmbeddingSine

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _sam3.shim_sam3_for_cpu()  # idempotent: a second call is a no-op
    assert _sam3.shim_sam3_for_cpu() is False

    # The precompute (a literal "cuda" allocation) is skipped; the cache fills lazily.
    sine = PositionEmbeddingSine(64, precompute_resolution=64)
    assert sine.cache == {}
    h, w = TransformerDecoder._get_coords(4, 4, "cuda")
    assert h.device.type == "cpu" and w.device.type == "cpu"

    # The fused MLP op stays fp32 on CPU instead of casting to bf16.
    from sam3.model import vitdet

    linear = torch.nn.Linear(4, 3)
    x = torch.randn(2, 4)
    with torch.no_grad():
        y = vitdet.addmm_act(torch.nn.GELU, linear, x)
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, torch.nn.functional.gelu(linear(x)))

    # Pinning is for host→CUDA copies; without CUDA it hands the tensor back.
    assert x.pin_memory() is x
