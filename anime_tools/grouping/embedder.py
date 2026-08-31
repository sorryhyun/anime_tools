"""PE-Spatial embedder for dataset grouping / near-twin mining.

The feature cache (:func:`anime_tools.grouping.features.embed_members`) takes
any :class:`~anime_tools.grouping.features.Embedder`; this is the package's own
PE-Spatial-B16-512 implementation (via :mod:`anime_tools.vision.pe`) and the
default ``build_groups`` uses when no embedder is injected.

``pe_spatial_embedder`` is a dotted-path factory (``module:callable``) so the
``build_groups`` CLI can be pointed at it — or at any other embedder — with
``--embedder``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from anime_tools.grouping.features import GRID_CACHE, GRID_NATIVE


class PESpatialEmbedder:
    """Wraps a PE-Spatial tower as a grouping ``Embedder``: ``[B,3,512,512]``
    device batch in ``[-1, 1]`` → ``(cls [B,768] L2-normed f32, grid16
    [B,16,16,768] f16)``."""

    name = "pe_spatial"

    def __init__(self, model, device: torch.device, dtype: torch.dtype):
        self.model = model
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        feats, _pooled = self.model.encode(batch.to(self.device, self.dtype))
        lhs = feats.float()  # [B, 1+1024, 768]
        cls = F.normalize(lhs[:, 0], dim=-1)  # global descriptor
        grid = lhs[:, 1:].reshape(lhs.shape[0], GRID_NATIVE, GRID_NATIVE, -1)
        g = grid.permute(0, 3, 1, 2)  # [B, 768, 32, 32]
        g16 = F.adaptive_avg_pool2d(g, GRID_CACHE).permute(0, 2, 3, 1)
        return cls.cpu().numpy(), g16.cpu().numpy().astype(np.float16)


def pe_spatial_embedder(
    device: torch.device | str | None = None,
    *,
    model_path=None,
    dtype: torch.dtype = torch.bfloat16,
) -> PESpatialEmbedder:
    """Factory: load PE-Spatial-B16-512 on ``device`` (auto when None).

    bf16 by default, which is what existing ``$NEAR_TWIN_CACHE`` entries were
    written with."""
    from anime_tools._device import resolve_device
    from anime_tools.vision.pe import load_pe_spatial

    dev = torch.device(resolve_device(device))
    model = load_pe_spatial(dev, model_path=model_path, dtype=dtype)
    return PESpatialEmbedder(model, dev, dtype)
