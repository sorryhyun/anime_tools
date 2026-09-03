"""The dbv4 backbone on onnxruntime — same contract as :class:`Dbv4Backend`.

:class:`Dbv4OnnxBackend` answers ``forward`` / ``forward_tensor`` /
``card`` / ``d_hidden`` exactly as the torch backend does, so
:class:`~anime_tools.tagger.tagger.AnimaTagger` holds one or the other and nothing
downstream of the score vector can tell which. It is picked automatically when
``<ckpt_dir>/dbv4.onnx`` exists — :mod:`anime_tools.tagger.onnx_export` writes it.

Why bother, given torch is a plain dependency either way (SAM3, PE-Spatial): speed.
Measured on this backbone at 384px, batch 1, an Apple CPU — torch bfloat16 (the
default dtype) 2.21 s/img, torch float32 1.54, onnxruntime 0.49. timm also drops
out of the tagging path, which is what lets the ComfyUI node tag without building
a second timm model inside ComfyUI's own torch.

The outputs are still torch tensors: :class:`Dbv4Output` is the seam with the
sidecar head and the threshold / group post-processing, all of which are torch and
all of which stay. ``torch.from_numpy`` over a 12k-float row is free next to the
forward pass.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anime_tools._onnx import make_session, resolve_onnx_device
from anime_tools.tagger.dbv4_backend import (
    Dbv4Card,
    Dbv4Output,
    load_dbv4_card,
    preprocess_dbv4,
)
from anime_tools.tagger.dbv4_meta import (
    DEFAULT_DBV4_IMG_SIZE,
    DEFAULT_DBV4_REPO,
)

logger = logging.getLogger(__name__)

INPUT_NAME = "pixel_values"
OUTPUT_NAMES = ("probs", "hidden")
"""The graph's IO, as :mod:`anime_tools.tagger.onnx_export` names it. Read back off
the session rather than trusted, so a graph exported by an older build with other
names fails at load with something readable instead of at the first predict."""


class Dbv4OnnxBackend:
    """Lazy-loading onnxruntime wrapper around an exported dbv4 graph.

    ``device`` follows the ORT probe (:func:`anime_tools._onnx.resolve_onnx_device`),
    not torch's: this backend is the one thing in the tagger that does not need a
    torch CUDA context, and initialising one just to ask would hand ORT a device it
    has to time-share.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        repo: str = DEFAULT_DBV4_REPO,
        img_size: int = DEFAULT_DBV4_IMG_SIZE,
        device: str | None = None,
        revision: str | None = None,
        card: Dbv4Card | None = None,
    ):
        self.onnx_path = Path(onnx_path)
        self.repo = repo
        self.img_size = int(img_size)
        self.device = resolve_onnx_device(device)
        self.revision = revision
        self._card = card
        self._session = None

    @property
    def card(self) -> Dbv4Card:
        if self._card is None:
            self._card = load_dbv4_card(self.repo, revision=self.revision)
        return self._card

    @property
    def session(self):
        if self._session is None:
            self._session = self._load_session()
        return self._session

    @property
    def d_hidden(self) -> int:
        return int(self.session.get_outputs()[1].shape[-1])

    def _load_session(self):
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"no exported dbv4 graph at {self.onnx_path} — run "
                f"`python -m anime_tools.tagger.cli.export_onnx "
                f"--ckpt_dir {self.onnx_path.parent}`"
            )
        session = make_session(self.onnx_path, self.device, what="the dbv4 tagger")
        got_in = [i.name for i in session.get_inputs()]
        got_out = [o.name for o in session.get_outputs()]
        if got_in != [INPUT_NAME] or tuple(got_out) != OUTPUT_NAMES:
            raise RuntimeError(
                f"{self.onnx_path} is not an Anima dbv4 graph: expected inputs "
                f"[{INPUT_NAME}] / outputs {list(OUTPUT_NAMES)}, got {got_in} / "
                f"{got_out}. Re-export it with "
                "`python -m anime_tools.tagger.cli.export_onnx --overwrite`."
            )
        n_classes = int(session.get_outputs()[0].shape[-1])
        if n_classes != self.card.n_classes:
            raise RuntimeError(
                f"{self.onnx_path} emits {n_classes} classes but the card for "
                f"{self.repo} has {self.card.n_classes} — the graph was exported "
                "against a different backbone. Re-export it."
            )
        logger.info(
            "Dbv4OnnxBackend: %s (%d classes, %dpx) on %s via %s",
            self.onnx_path,
            n_classes,
            self.img_size,
            self.device,
            session.get_providers()[0],
        )
        return session

    def forward_tensor(self, x01: torch.Tensor) -> Dbv4Output:
        """``[B, 3, S, S]`` in [0, 1] → probs + hidden.

        Normalisation lives inside the graph, so what the session is fed is exactly
        what :func:`preprocess_dbv4` returns — the same tensor the torch backend
        takes.
        """
        x = np.ascontiguousarray(x01.detach().cpu().numpy(), dtype=np.float32)
        probs, hidden = self.session.run(None, {INPUT_NAME: x})
        return Dbv4Output(
            probs=torch.from_numpy(probs.astype(np.float32, copy=False)),
            hidden=torch.from_numpy(hidden.astype(np.float32, copy=False)),
        )

    def forward(self, images: Sequence[Image.Image]) -> Dbv4Output:
        x = torch.stack([preprocess_dbv4(im, self.img_size) for im in images])
        return self.forward_tensor(x)
