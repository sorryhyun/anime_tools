"""Export the dbv4 backbone to a single ONNX graph — a one-time local build.

The upstream repo ships a ``model.onnx`` of its own, and it is the wrong graph for
us: its ``embedding`` output is the 768-d pooled feature from *before* the head,
while the sidecar (``sidecar.json['d_in']``) trains on the 3072-d ``mlp_hidden``
that :meth:`Dbv4Backend.forward_tensor` takes out of the middle of timm's
``MlpHead``. So the graph is exported here, with both outputs the tagger actually
reads, and the sidecar / feature cache / trainer stay untouched.

What comes out is one file with the same contract as
:meth:`Dbv4Backend.forward_tensor`:

* input ``pixel_values`` ``[batch, 3, S, S]`` float32 in ``[0, 1]`` — ImageNet
  normalisation is folded into the graph, so the caller's preprocessing stays
  :func:`~anime_tools.tagger.dbv4_backend.preprocess_dbv4` and nothing else.
* outputs ``probs`` ``[batch, n_classes]`` (sigmoid, not logits) and ``hidden``
  ``[batch, d_hidden]``.

torch, timm and the ``export`` dependency group are needed *here*; running the
result needs only onnxruntime. The weights are gated and GPL-3.0, so the file is
never redistributed — every user exports their own.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from anime_tools.tagger.dbv4_backend import Dbv4Backend
from anime_tools.tagger.dbv4_meta import (
    DEFAULT_DBV4_ARCH,
    DEFAULT_DBV4_IMG_SIZE,
    DEFAULT_DBV4_REPO,
    dbv4_onnx_path,
)

logger = logging.getLogger(__name__)

DEFAULT_OPSET = 18
MAX_EXPORT_BATCH = 256
"""Upper bound declared for the dynamic batch dimension. torch.export wants a
*range*, not "anything"; nothing here feeds the backbone more than a handful of
crops at a time, and the bound is only a promise to the shape solver."""


class _ExportWrapper(nn.Module):
    """:meth:`Dbv4Backend.forward_tensor` as a module, so the export traces it.

    The two heads come out of one pass exactly the way the torch backend splits
    them (``fc1 → act → norm`` is the sidecar's feature, ``fc2`` the tag logits);
    keeping the split *inside* the graph is what makes the exported file a drop-in
    for the backend rather than a second thing to keep in step.
    """

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, pixel_values: torch.Tensor):
        x = (pixel_values - self.mean) / self.std
        feats = self.model.forward_features(x)
        pooled = self.model.forward_head(feats, pre_logits=True)
        fc = self.model.head.fc
        hidden = fc.norm(fc.act(fc.fc1(pooled)))
        return fc.fc2(hidden).sigmoid(), hidden


def export_dbv4_onnx(
    out_path: str | Path,
    *,
    repo: str = DEFAULT_DBV4_REPO,
    arch: str = DEFAULT_DBV4_ARCH,
    img_size: int = DEFAULT_DBV4_IMG_SIZE,
    revision: str | None = None,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Build ``out_path`` from the backbone ``repo`` and return it.

    Exported in float32 whatever the runtime will use: ORT picks its own kernels,
    and a bf16 graph would only pin the CPU path to the slowest of them.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend = Dbv4Backend(
        repo=repo,
        arch=arch,
        img_size=img_size,
        device="cpu",
        dtype=torch.float32,
        revision=revision,
    )
    mean, std = backend.normalization
    wrapper = _ExportWrapper(backend.model, mean, std).eval()
    # Batch 2, not 1: a dynamic dimension whose example value is 1 reads as a
    # degenerate range and torch.export refuses it ("Invalid ranges [2:1]").
    example = torch.rand(2, 3, img_size, img_size)
    batch = torch.export.Dim("batch", min=1, max=MAX_EXPORT_BATCH)

    logger.info("exporting %s (%s, %dpx) → %s", repo, arch, img_size, out_path)
    torch.onnx.export(
        wrapper,
        (example,),
        str(out_path),
        dynamo=True,
        # One file, not a graph plus a sidecar blob: `dbv4.onnx` alone is what the
        # backend probes for, and a half-copied pair would pass that probe and then
        # fail inside the session.
        external_data=False,
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["probs", "hidden"],
        dynamic_shapes={"pixel_values": {0: batch}},
    )
    return out_path


def export_for_checkpoint(
    ckpt_dir: str | Path,
    *,
    opset: int = DEFAULT_OPSET,
    overwrite: bool = False,
) -> Path:
    """Export the backbone *that checkpoint was built against* into it.

    The repo / arch / image size come from ``config.json['dbv4']``, the same block
    :meth:`AnimaTagger._init_dbv4_backend` reads, so the graph beside a checkpoint
    can never be another checkpoint's backbone.
    """
    from anime_tools._json import read_json

    ckpt_dir = Path(ckpt_dir)
    out = dbv4_onnx_path(ckpt_dir)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists — pass --overwrite to rebuild it")
    d = dict(read_json(ckpt_dir / "config.json").get("dbv4") or {})
    return export_dbv4_onnx(
        out,
        repo=d.get("repo", DEFAULT_DBV4_REPO),
        arch=d.get("arch", DEFAULT_DBV4_ARCH),
        img_size=int(d.get("img_size", DEFAULT_DBV4_IMG_SIZE)),
        revision=d.get("revision"),
        opset=opset,
    )
