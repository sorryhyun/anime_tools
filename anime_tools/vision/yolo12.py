"""Vendored YOLO12 detection graph — the one architecture the OCR detector runs.

The AnimeText text-block detector (:mod:`anime_tools.ocr.animetext`) is a YOLO12-l
published as an ultralytics checkpoint. Ultralytics itself is AGPL-3.0 and this
package is MIT, so its runtime is not a dependency here: the twelve module types
that checkpoint names are written out below, the ``model.N.…`` numbering is kept
exactly as ultralytics lays it out, and the tensors load with
:func:`load_ultralytics_state_dict` — a ``torch.load`` whose unpickler stubs the
``ultralytics.*`` classes it is asked for. Nothing of ultralytics is imported,
installed, or vendored; only the weight file's own layout is read.

Written against the published YOLO12 architecture and checked numerically against
the ONNX export the detector used to run, on the day it replaced it (2026-09-09):
every one of the 22 ``model.N`` outputs within 9.2e-05 at 320², the head within
1.2e-03 at 320² and 3.0e-03 at 640² — float32 rounding on a box coordinate that is
itself in canvas pixels. ``tests/test_vision_yolo12.py`` pins the graph's shape and
the checkpoint contract; the comparison itself needed onnxruntime and a 106 MB
graph, so it is not a test.

The graph mirrors ``yolov12l.yaml``: scale ``l`` is ``depth 1.0 / width 1.0 /
max_channels 512``, which is why every ``1024`` in the yaml lands as 512 here, and
why the ``C3k2`` blocks take ``C3k`` bodies and the ``A2C2f`` blocks a residual
gamma at ``mlp_ratio`` 1.2 — all three are what ultralytics' parser does for the
m/l/x scales. :data:`LAYERS` is that yaml, resolved.

Torch and torchvision only. Checkpoint: ``deepghs/AnimeText_yolo`` (model card
GPL-3.0, dataset CC-BY-NC-SA-4.0), fetched on first use and never bundled.
"""

from __future__ import annotations

import pickle
import types
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

REG_MAX = 16
"""DFL bins per box side. ``4 * REG_MAX`` is the box branch's channel count."""

STRIDES = (8.0, 16.0, 32.0)
"""The three detection levels' strides, in canvas pixels per cell."""


# The twelve module types the checkpoint names.


def _autopad(k: int, p: int | None = None) -> int:
    return k // 2 if p is None else p


BN_EPS = 1e-3
"""Ultralytics' BatchNorm epsilon, not torch's 1e-5. It is folded into the
published weights, so the wrong one is a detector that reads plausible nonsense
rather than one that fails — the first conv's activations move by 271 at 1e-5."""


class Conv(nn.Module):
    """conv-bn-act, ultralytics' spelling of it — ``act=False`` drops the SiLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=BN_EPS)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depthwise :class:`Conv` — one group per input channel."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, act: bool = True):
        import math

        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    """Two convs and an optional identity, the residual unit under every C3k."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    """CSP block: ``n`` bottlenecks down one branch, a plain conv down the other."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    """The C2f split-and-accumulate block, with :class:`C3k` bodies at this scale."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class AAttn(nn.Module):
    """YOLO12's area attention: the map split into ``area`` horizontal bands, each
    attended over on its own, plus a depthwise positional term on the values.

    ``area`` is what keeps this affordable at detector resolutions — a 80x80 map
    is 6400 tokens, and four bands make that four 1600-token problems.
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        super().__init__()
        self.area = area
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * num_heads
        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(b * self.area, n // self.area, c * 3)
            b, n, _ = qkv.shape

        q, k, v = (
            qkv.view(b, n, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = ((q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)).softmax(dim=-1)
        y = (v @ attn.transpose(-2, -1)).permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            y = y.reshape(b // self.area, n * self.area, c)
            v = v.reshape(b // self.area, n * self.area, c)
            b, n, _ = y.shape

        y = y.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return self.proj(y + self.pe(v))


class ABlock(nn.Module):
    """Area attention and an MLP, each added residually."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        super().__init__()
        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, hidden, 1), Conv(hidden, dim, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """The C2f accumulate pattern over attention pairs (``a2``) or :class:`C3k`
    bodies, with a learned residual scale when it is attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__()
        c_ = int(c2 * e)
        num_heads = c_ // 32
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        self.gamma = nn.Parameter(0.01 * torch.ones(c2)) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        if self.gamma is None:
            return out
        return x + self.gamma.view(1, -1, 1, 1) * out


class Concat(nn.Module):
    """``torch.cat`` as a graph node, so the layer table can name it."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.d = dim

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(xs), self.d)


class DFL(nn.Module):
    """Distribution Focal Loss decode: each box side's ``REG_MAX`` logits softmaxed
    and read as the expectation over bin indices."""

    def __init__(self, c1: int = REG_MAX):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.conv.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )


class Detect(nn.Module):
    """The three-level decoupled head: DFL box distances and per-class scores,
    decoded onto the canvas in one ``(B, 4 + nc, N)`` tensor.

    The output is exactly what the ONNX export emitted — ``cx cy w h`` in canvas
    pixels followed by sigmoid scores — which is why
    :func:`anime_tools.ocr.animetext.decode` did not change when the runtime did.
    """

    def __init__(self, nc: int, ch: Sequence[int]):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = REG_MAX
        self.no = nc + self.reg_max * 4
        c2 = max(16, ch[0] // 4, self.reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        heads = [
            torch.cat((self.cv2[i](f), self.cv3[i](f)), 1) for i, f in enumerate(feats)
        ]
        anchors, strides = make_anchors(heads, STRIDES)
        cat = torch.cat([h.view(h.shape[0], self.no, -1) for h in heads], 2)
        box, cls = cat.split((self.reg_max * 4, self.nc), 1)
        lt, rb = self.dfl(box).chunk(2, 1)
        x1y1 = anchors.unsqueeze(0) - lt
        x2y2 = anchors.unsqueeze(0) + rb
        xywh = torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), 1) * strides
        return torch.cat((xywh, cls.sigmoid()), 1)


def make_anchors(
    feats: Sequence[torch.Tensor], strides: Sequence[float], offset: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cell centres and their strides for the three levels, as ``(2, N)`` and
    ``(1, N)`` — built per call, since the canvas may be any size."""
    points, scales = [], []
    for f, stride in zip(feats, strides, strict=True):
        h, w = f.shape[2], f.shape[3]
        sy, sx = torch.meshgrid(
            torch.arange(h, device=f.device, dtype=f.dtype) + offset,
            torch.arange(w, device=f.device, dtype=f.dtype) + offset,
            indexing="ij",
        )
        points.append(torch.stack((sx, sy), -1).view(-1, 2))
        scales.append(torch.full((h * w, 1), stride, device=f.device, dtype=f.dtype))
    return torch.cat(points).transpose(0, 1), torch.cat(scales).transpose(0, 1)


# The graph.

LAYERS: tuple[tuple, ...] = (
    # (from, module, kwargs) — `from` is -1 for the previous layer, an int for an
    # earlier one, or a tuple for a Concat. Index == the checkpoint's `model.N`.
    (-1, Conv, {"c1": 3, "c2": 64, "k": 3, "s": 2}),
    (-1, Conv, {"c1": 64, "c2": 128, "k": 3, "s": 2}),
    (-1, C3k2, {"c1": 128, "c2": 256, "n": 2, "c3k": True, "e": 0.25}),
    (-1, Conv, {"c1": 256, "c2": 256, "k": 3, "s": 2}),
    (-1, C3k2, {"c1": 256, "c2": 512, "n": 2, "c3k": True, "e": 0.25}),
    (-1, Conv, {"c1": 512, "c2": 512, "k": 3, "s": 2}),
    (-1, A2C2f, {"c1": 512, "c2": 512, "n": 4, "a2": True, "area": 4}),
    (-1, Conv, {"c1": 512, "c2": 512, "k": 3, "s": 2}),
    (-1, A2C2f, {"c1": 512, "c2": 512, "n": 4, "a2": True, "area": 1}),
    (-1, nn.Upsample, {"scale_factor": 2, "mode": "nearest"}),
    ((-1, 6), Concat, {"dim": 1}),
    (-1, A2C2f, {"c1": 1024, "c2": 512, "n": 2, "a2": False}),
    (-1, nn.Upsample, {"scale_factor": 2, "mode": "nearest"}),
    ((-1, 4), Concat, {"dim": 1}),
    (-1, A2C2f, {"c1": 1024, "c2": 256, "n": 2, "a2": False}),
    (-1, Conv, {"c1": 256, "c2": 256, "k": 3, "s": 2}),
    ((-1, 11), Concat, {"dim": 1}),
    (-1, A2C2f, {"c1": 768, "c2": 512, "n": 2, "a2": False}),
    (-1, Conv, {"c1": 512, "c2": 512, "k": 3, "s": 2}),
    ((-1, 8), Concat, {"dim": 1}),
    (-1, C3k2, {"c1": 1024, "c2": 512, "n": 2, "c3k": True}),
    ((14, 17, 20), Detect, {"ch": (256, 512, 512)}),
)
"""``yolov12l.yaml`` with scale ``l`` resolved. The `l`/`x` scales carry a
residual attention block at ``mlp_ratio`` 1.2, which :class:`Yolo12` applies to
every ``a2=True`` entry rather than repeating it in the table."""

RESIDUAL_MLP_RATIO = 1.2
"""``A2C2f``'s residual scale and MLP width at the l/x scales."""


class Yolo12(nn.Module):
    """The detection graph, laid out so ``model.N.…`` matches the checkpoint.

    ``forward`` takes the ``(B, 3, H, W)`` float RGB canvas in [0, 1] that
    :func:`anime_tools.ocr.animetext.to_tensor` builds and answers the
    ``(B, 4 + nc, N)`` head :func:`~anime_tools.ocr.animetext.decode` reads. ``H``
    and ``W`` must be multiples of 32.
    """

    def __init__(self, nc: int = 1):
        super().__init__()
        self.nc = nc
        layers, self.froms = [], []
        for src, cls, kwargs in LAYERS:
            if cls is A2C2f and kwargs.get("a2"):
                kwargs = {**kwargs, "residual": True, "mlp_ratio": RESIDUAL_MLP_RATIO}
            if cls is Detect:
                kwargs = {**kwargs, "nc": nc}
            layers.append(cls(**kwargs))
            self.froms.append(src)
        self.model = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs: list[torch.Tensor] = []
        for layer, src in zip(self.model, self.froms, strict=True):
            if isinstance(src, tuple):
                arg = [x if i == -1 else outs[i] for i in src]
                x = layer(arg)
            else:
                x = layer(x if src == -1 else outs[src])
            outs.append(x)
        return x


# Weights.


def load_ultralytics_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """The float32 tensors inside an ultralytics ``.pt``, without ultralytics.

    The checkpoint pickles a live ``DetectionModel``, so unpickling it names
    ``ultralytics.nn.*`` classes. The unpickler below answers each of those with a
    bare :class:`~torch.nn.Module` subclass, which is enough to rebuild the module
    tree and read ``state_dict()`` off it — the class bodies are never needed,
    since nothing is ever called on the result. Ultralytics is not imported and a
    real installation of it is not consulted either, so this reads the same on any
    machine.

    Ultralytics saves in half precision; the tensors come back as float32, which is
    what the graph runs in on every device.
    """

    class _Stub(nn.Module):
        def __setstate__(self, state: dict) -> None:
            self.__dict__.update(state)

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.split(".")[0] == "ultralytics":
                return type(name, (_Stub,), {"__module__": module})
            return super().find_class(module, name)

    shim = types.ModuleType("anime_tools.vision.yolo12._pickle")
    shim.Unpickler = _Unpickler  # type: ignore[attr-defined]
    shim.load = pickle.load  # type: ignore[attr-defined]

    ckpt = torch.load(
        str(path), map_location="cpu", pickle_module=shim, weights_only=False
    )
    model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    return {
        k: v.float()
        for k, v in model.state_dict().items()
        if not k.endswith("num_batches_tracked")
    }


def load_yolo12(
    weights: str | Path,
    *,
    nc: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Yolo12:
    """The graph in eval mode on ``device``, built from an ultralytics ``.pt``.

    Anything the checkpoint carries that this graph does not — ultralytics keeps a
    couple of registered buffers on the head — is refused rather than ignored: a
    silent mismatch here is a detector that returns plausible nonsense.
    """
    model = Yolo12(nc=nc)
    sd = load_ultralytics_state_dict(weights)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("dfl.conv.weight")]
    unexpected = [k for k in unexpected if not k.endswith((".anchors", ".strides"))]
    if missing or unexpected:
        raise RuntimeError(
            f"{weights} is not a YOLO12-l detection checkpoint: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return model.eval().to(device=device, dtype=dtype).requires_grad_(False)


__all__ = [
    "BN_EPS",
    "DFL",
    "LAYERS",
    "REG_MAX",
    "STRIDES",
    "A2C2f",
    "AAttn",
    "ABlock",
    "Bottleneck",
    "C3k",
    "C3k2",
    "Concat",
    "Conv",
    "DWConv",
    "Detect",
    "Yolo12",
    "load_ultralytics_state_dict",
    "load_yolo12",
    "make_anchors",
]
