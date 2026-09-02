"""Text masks: the UNet++ stroke segmenter behind comictextdetector's text-block gate,
SAM3 grounded on prompts, or both — :func:`run_mit_masks` over a
:class:`~anime_tools.masking.requests.MitMaskRequest`.

The CLI (``cli/generate_masks_mit.py``) is a shell over this module.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from torch import nn

from anime_tools._device import resolve_device
from anime_tools._env import resolve_path

# Both nets are catalog rows so the GUI can pre-fetch them. The UNet++ is read out of the
# hub cache; the CTD gate is a path, and it is the catalog's rather than a flag's.
from anime_tools.downloads import MIT_TEXT_FILENAME as _HF_FILENAME
from anime_tools.downloads import MIT_TEXT_REPO as _HF_REPO
from anime_tools.downloads import default_ctd_onnx_path
from anime_tools.masking._masks import (
    MaskRun,
    coverage_pct,
    mask_run,
    write_ignore_mask,
)

# Importing _sam3 also installs the `np.bool` alias sam3 needs before it loads.
from anime_tools.masking._sam3 import autocast, detect_union, load_sam3
from anime_tools.masking.requests import MitMaskRequest

_ENCODER = "tu-efficientnetv2_rw_m"


def _convert_batchnorm_to_groupnorm(module: nn.Module) -> None:
    """Replace BatchNorm2d with GroupNorm in decoder (matches training setup)."""
    from torch import nn

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = 8
            if num_channels < num_groups or num_channels % num_groups != 0:
                for i in range(min(num_channels, 8), 1, -1):
                    if num_channels % i == 0:
                        num_groups = i
                        break
                else:
                    num_groups = 1
            setattr(
                module,
                name,
                nn.GroupNorm(num_groups=num_groups, num_channels=num_channels),
            )
        else:
            _convert_batchnorm_to_groupnorm(child)


def _load_model(model_path: str | None = None, device: str = "cuda") -> nn.Module:
    # smp costs ~2s to import (timm + torchvision eagerly), and `--help` / the GUI schema
    # dump import this module just for its parser.
    import segmentation_models_pytorch as smp
    import torch

    model = smp.UnetPlusPlus(
        encoder_name=_ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse",
    )
    _convert_batchnorm_to_groupnorm(model.decoder)

    if model_path is None:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@cache
def _transform():
    """Albumentations is another second of import; build it on first use."""
    from albumentations import Compose, Normalize
    from albumentations.pytorch import ToTensorV2

    return Compose(
        [
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _detect_mask(
    model: nn.Module,
    image: np.ndarray,
    device: str = "cuda",
    text_threshold: float | None = None,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    h, w = image.shape[:2]

    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    with torch.no_grad():
        tensor = _transform()(image=image)["image"].unsqueeze(0).to(device)

        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

        is_cuda = device == "cuda" or (
            isinstance(device, torch.device) and device.type == "cuda"
        )
        if is_cuda:
            with torch.amp.autocast("cuda"):
                logits = model(tensor)
        else:
            logits = model(tensor)

        prob_map = logits.sigmoid()[0, 0, :h, :w].cpu().numpy()

    if text_threshold is not None:
        prob_map = (prob_map > text_threshold).astype(np.float32)

    mask = (prob_map * 255).astype(np.uint8)
    return mask


def _load_ctd(onnx_path: str, device: str = "cuda"):
    """Return forward(canvas_1024_rgb) -> raw output list.

    onnxruntime CUDAExecutionProvider when available (~17 ms/forward vs seconds
    on cv2.dnn CPU), cv2.dnn CPU as fallback.
    """
    if device != "cpu":
        try:
            import onnxruntime as ort

            ort.preload_dlls()  # resolve cudnn/cublas from the venv's nvidia-* wheels
            sess = ort.InferenceSession(
                str(onnx_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            if "CUDAExecutionProvider" in sess.get_providers():
                inp = sess.get_inputs()[0].name

                def forward(canvas: np.ndarray) -> list[np.ndarray]:
                    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
                    return sess.run(None, {inp: blob})

                return forward
            print(
                "WARNING: onnxruntime CUDAExecutionProvider unavailable — CTD gate falls back to cv2.dnn CPU"
            )
        except Exception as e:  # noqa: BLE001 — any ORT init failure degrades to CPU
            print(
                f"WARNING: onnxruntime CUDA init failed ({e}) — CTD gate falls back to cv2.dnn CPU"
            )

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    uoln = net.getUnconnectedOutLayersNames()

    def forward(canvas: np.ndarray) -> list[np.ndarray]:
        net.setInput(
            cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, size=(1024, 1024))
        )
        return list(net.forward(uoln))

    return forward


def _ctd_text_boxes(
    ctd_forward, img: np.ndarray, conf_th=0.4, nms_th=0.35, seg_th=0.3, seg_cov=0.03
) -> list[tuple[int, int, int, int]]:
    """Text-block boxes (yolo blk head + stroke-coverage cross-check) in img coords.

    The blk head is required: the seg head alone false-positives on halos and
    ornaments.
    """
    h0, w0 = img.shape[:2]
    r = min(1024 / h0, 1024 / w0)
    nw, nh = round(w0 * r), round(h0 * r)
    canvas = np.zeros((1024, 1024, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    outs = ctd_forward(canvas)
    blk = next(o for o in outs if o.ndim == 3)[0]  # (N,7)
    seg = next(o for o in outs if o.ndim == 4 and o.shape[1] == 1)[0, 0]  # stroke prob
    conf = blk[:, 4] * blk[:, 5:].max(axis=1)
    keep = conf > conf_th
    if not keep.any():
        return []
    b, c = blk[keep], conf[keep]
    xywh = np.concatenate([b[:, :2] - b[:, 2:4] / 2, b[:, 2:4]], axis=1)
    boxes = []
    for i in np.array(
        cv2.dnn.NMSBoxes(xywh.tolist(), c.tolist(), conf_th, nms_th)
    ).flatten():
        x, y, w, h = xywh[i]
        cx0, cy0 = max(int(x), 0), max(int(y), 0)
        cx1, cy1 = min(int(x + w), 1024), min(int(y + h), 1024)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        if (seg[cy0:cy1, cx0:cx1] > seg_th).mean() < seg_cov:
            continue
        boxes.append(
            (
                max(int(x / r), 0),
                max(int(y / r), 0),
                min(int((x + w) / r), w0),
                min(int((y + h) / r), h0),
            )
        )
    return boxes


def _keep_text_blocks(ctd_forward, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``mask`` minus every component no comictextdetector text block overlaps.

    Component-wise rather than box-wise: the blk head decides *which* strokes are text,
    the UNet++ still decides where each ends, so a kept letter keeps its own outline
    instead of the rectangle around it.
    """
    boxmask = np.zeros(mask.shape, bool)
    for x0, y0, x1, y1 in _ctd_text_boxes(ctd_forward, img):
        boxmask[y0:y1, x0:x1] = True
    _, lab = cv2.connectedComponents(mask)
    keep_ids = np.unique(lab[boxmask])
    return np.isin(lab, keep_ids[keep_ids != 0]).astype(np.uint8)


def run_mit_masks(req: MitMaskRequest) -> MaskRun:
    """Write ``{stem}_mask.png`` under ``req.mask_dir`` for every image the walk
    plans; returns the run (its ``items`` are what was planned)."""
    device = resolve_device(req.device)
    sam_prompts = req.active_sam_prompts

    dilate_kernel = (
        np.ones((req.dilate, req.dilate), dtype=np.uint8) if req.dilate > 0 else None
    )

    model = None
    ctd = None
    if req.use_mit:
        print("Loading text segmentation model...")
        model = _load_model(req.model_path, device=device)
        if req.ctd_gate:
            ctd_onnx = default_ctd_onnx_path()
            if ctd_onnx.exists():
                ctd = _load_ctd(str(ctd_onnx), device=device)
            else:
                print(
                    f"WARNING: --ctd-gate on but {ctd_onnx} missing — gating "
                    f"disabled (get it with `python -m anime_tools.downloads "
                    f"ctd_onnx`, or pass --no-ctd-gate)"
                )

    sam_model = None
    processor = None
    if sam_prompts:
        print("Loading SAM3 model...")
        sam_model, processor = load_sam3(
            resolve_path(req.checkpoint) if req.checkpoint else None, device
        )
        print(f"SAM3 prompts: {', '.join(sam_prompts)}")

    amp = autocast(device)

    with mask_run(req) as run:
        for image_path, mask_path in run.items:
            pil_image = Image.open(image_path).convert("RGB")
            img_np = np.array(pil_image)
            h, w = img_np.shape[:2]

            run.advance()

            text_mask = np.zeros((h, w), dtype=np.uint8)
            # Distinguished in the progress line: a page with no text and a page whose
            # gate threw everything away both mask nothing, and only the second is a knob
            # to reconsider.
            gated_away = False

            if model is not None:
                unet = _detect_mask(
                    model,
                    img_np,
                    device=device,
                    text_threshold=req.text_threshold,
                )
                text_mask = (unet > 127).astype(np.uint8)
                if ctd is not None and text_mask.any():
                    text_mask = _keep_text_blocks(ctd, img_np, text_mask)
                    gated_away = not text_mask.any()

            if processor is not None:
                with amp:
                    state = processor.set_image(pil_image)
                    text_mask = np.maximum(
                        text_mask,
                        detect_union(
                            processor,
                            sam_model,
                            state,
                            sam_prompts,
                            (h, w),
                            req.sam_threshold,
                        ),
                    )

            if not text_mask.any():
                why = " (ctd-gated)" if gated_away else ""
                run.note(image_path, f"skipped{why}")
                continue

            # One dilation over the union, not one per detector: the two overlap on a
            # lettered balloon, and dilating twice would grow that seam twice.
            if dilate_kernel is not None:
                text_mask = cv2.dilate(text_mask, dilate_kernel, iterations=1)

            write_ignore_mask(mask_path, text_mask, pool=run.pool)
            run.note(image_path, f"{coverage_pct(text_mask):.1f}%")
    return run
