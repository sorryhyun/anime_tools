#!/usr/bin/env python3
"""Generate text segmentation masks for training images.

Model: https://huggingface.co/a-b-c-x-y-z/Manga-Text-Segmentation-2025

By default each mask is gated by comictextdetector's text-BLOCK head (--ctd-gate):
only UNet++ mask components overlapping a detected text block survive. The UNet++
alone has a small but systematic false-positive habit on decorative line art
(measured 2026-07-04: 0.07-0.63% of pixels on halo/ornament-only images — i.e. it
excludes exactly the decorative elements a style LoRA should train on), while its
real-text recall is solid. The blk head supplies the precision; the UNet++ supplies
the stroke-accurate mask. Trade-off: sfx text the blk head misses is no longer
masked — use --no-ctd-gate to restore raw UNet++ behavior.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from torch import nn

from anime_tools._device import resolve_device
from anime_tools._walk import walk_images

# The repo/filename live in the download catalog so the GUI can pre-fetch this
# net; the loader below reads it straight out of the hub cache either way.
from anime_tools.downloads import MIT_TEXT_FILENAME as _HF_FILENAME
from anime_tools.downloads import MIT_TEXT_REPO as _HF_REPO

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
    # segmentation_models_pytorch costs ~2s to import (it pulls timm + torchvision
    # eagerly). Only the weight-loading path needs it, and `--help` / the GUI's
    # schema dump import this module just for its parser -- keep it out of both.
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


def save_mask(path: Path, alpha_mask: np.ndarray) -> None:
    Image.fromarray(alpha_mask, mode="L").save(path)


def _load_ctd(onnx_path: str, device: str = "cuda"):
    """Return forward(canvas_1024_rgb) -> raw output list.

    onnxruntime CUDAExecutionProvider when available (~17 ms/forward vs seconds
    on cv2.dnn CPU — same lever as project/finished/sr/scripts/detect_text_boxes.py), cv2.dnn
    CPU as fallback.
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

    Mirrors project/finished/sr/scripts/detect_text_boxes.py::_detect — see there for why the blk
    head is required (the seg head alone false-positives on halos/ornaments).
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument(
        "--mask-dir", type=str, required=True, help="Output mask directory"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model.pth (downloads from HuggingFace if not specified)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing masks"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="cuda|cpu (default: auto)"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.8,
        help="Text segmentation threshold (default: 0.7)",
    )
    parser.add_argument(
        "--dilate", type=int, default=3, help="Mask dilation in pixels (default: 5)"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="I/O workers (default: 4)"
    )
    parser.add_argument(
        "--ctd-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep only mask components overlapping a comictextdetector text "
            "block — drops UNet++ false positives on halos/decorative line art "
            "(--no-ctd-gate = raw UNet++ masks, restores pre-2026-07 behavior)"
        ),
    )
    parser.add_argument(
        "--ctd-onnx",
        type=str,
        default="models/mit/comictextdetector.pt.onnx",
        help="comictextdetector onnx for --ctd-gate (from make download-models)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Walk subfolders under --image-dir. Mask output mirrors the source "
            "subdir structure under --mask-dir."
        ),
    )
    parser.add_argument(
        "--path-pattern",
        type=str,
        default=None,
        help=(
            "fnmatch glob (| to OR-combine) on each image's path relative to "
            "--image-dir, restricting which images get masked. Same semantics "
            "as the training path_pattern."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.device = resolve_device(args.device)

    dilate_kernel = (
        np.ones((args.dilate, args.dilate), dtype=np.uint8) if args.dilate > 0 else None
    )

    print("Loading text segmentation model...")
    model = _load_model(args.model_path, device=args.device)

    ctd = None
    if args.ctd_gate:
        if Path(args.ctd_onnx).exists():
            ctd = _load_ctd(args.ctd_onnx, device=args.device)
        else:
            print(
                f"WARNING: --ctd-gate on but {args.ctd_onnx} missing — gating "
                f"disabled (run `make download-models` or pass --no-ctd-gate)"
            )

    image_dir = Path(args.image_dir)
    masks_dir = Path(args.mask_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # walk_images enforces per-subfolder stem uniqueness (same-folder stem
    # collisions would overwrite each other's mask); same stem across folders
    # is fine — the nested output layout disambiguates by subdir.
    image_files = walk_images(
        image_dir,
        recursive=args.recursive,
        pattern=args.path_pattern,
    )

    work_items = []
    for image_path in image_files:
        try:
            rel = image_path.parent.relative_to(image_dir)
        except ValueError:
            rel = Path("")
        rel_str = str(rel)
        target_dir = masks_dir / rel if rel_str not in ("", ".") else masks_dir
        mask_path = target_dir / f"{image_path.stem}_mask.png"
        if mask_path.exists() and not args.force:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        work_items.append((image_path, mask_path))

    total = len(work_items)
    if total == 0:
        print("No images to process.")
        return

    pool = ThreadPoolExecutor(max_workers=args.workers)

    pbar = tqdm(total=total, desc="Generating masks")
    for image_path, mask_path in work_items:
        pil_image = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_image)

        mask = _detect_mask(
            model,
            img_np,
            device=args.device,
            text_threshold=args.text_threshold,
        )

        pbar.update(1)

        if mask is None or not mask.any():
            pbar.set_postfix_str(f"{image_path.name}: skipped")
            continue

        combined_mask = (mask > 127).astype(np.uint8)

        if ctd is not None and combined_mask.any():
            boxmask = np.zeros(combined_mask.shape, bool)
            for x0, y0, x1, y1 in _ctd_text_boxes(ctd, img_np):
                boxmask[y0:y1, x0:x1] = True
            _, lab = cv2.connectedComponents(combined_mask)
            keep_ids = np.unique(lab[boxmask])
            combined_mask = np.isin(lab, keep_ids[keep_ids != 0]).astype(np.uint8)
            if not combined_mask.any():  # everything was decorative-line FP
                pbar.set_postfix_str(f"{image_path.name}: skipped (ctd-gated)")
                continue

        if dilate_kernel is not None:
            combined_mask = cv2.dilate(combined_mask, dilate_kernel, iterations=1)

        # Invert: detected=1 → alpha=0 (ignore), no detection → alpha=255 (train)
        alpha_mask = ((1 - combined_mask) * 255).astype(np.uint8)

        pool.submit(save_mask, mask_path, alpha_mask)

        h, w = img_np.shape[:2]
        masked_pct = 100 * np.count_nonzero(combined_mask) / (w * h)
        pbar.set_postfix_str(f"{image_path.name}: {masked_pct:.1f}%")

    pbar.close()
    pool.shutdown(wait=True)
    print(f"Masks saved to {masks_dir}/")


if __name__ == "__main__":
    main()
