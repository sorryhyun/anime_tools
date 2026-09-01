"""Text masks: SAM3 prompts, a per-stroke UNet++, or both.

Two detectors over one walk, each behind its own switch, unioned before the
dilation and the write — they answer different questions about the same thing.
``--use-sam`` grounds SAM3 on ``--sam-prompts`` (`speech bubble` by default),
which is a *shape*: the balloon, its tail and the white inside it, all of it
off the loss. ``--use-mit`` runs the UNet++ text segmenter
(https://huggingface.co/a-b-c-x-y-z/Manga-Text-Segmentation-2025), which is a
*stroke*: the lettering itself, wherever it sits, sfx and bare captions
included.

The UNet++ half is gated by comictextdetector's text-BLOCK head (--ctd-gate):
only mask components overlapping a detected text block survive. The UNet++
alone systematically false-positives on decorative line art — exactly the elements
a style LoRA should train on — while its real-text recall is solid, so the blk head
supplies the precision and the UNet++ the stroke-accurate mask. Trade-off: sfx text
the blk head misses is no longer masked; --no-ctd-gate restores raw UNet++ output.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    from torch import nn

from anime_tools import workspace as WS
from anime_tools._device import resolve_device
from anime_tools._env import resolve_path

# Both nets are catalog rows so the GUI can pre-fetch them. The UNet++ is read
# out of the hub cache either way; the CTD gate is a *path*, and it is the
# catalog's, not a flag's — a stage that let you point --ctd-onnx elsewhere is a
# stage whose Download button can disagree with its loader.
from anime_tools.downloads import MIT_TEXT_FILENAME as _HF_FILENAME
from anime_tools.downloads import MIT_TEXT_REPO as _HF_REPO
from anime_tools.downloads import default_ctd_onnx_path
from anime_tools.masking._masks import (
    add_device_arg,
    add_force_arg,
    add_mask_dir_args,
    add_walk_args,
    add_workers_arg,
    gated_group,
    plan_mask_jobs,
    write_ignore_mask,
)

# Importing _sam3 also installs the `np.bool` alias sam3 needs before it loads.
from anime_tools.masking._sam3 import (
    add_checkpoint_arg,
    autocast,
    detect_union,
    load_sam3,
    prompt_list,
)

_ENCODER = "tu-efficientnetv2_rw_m"

DEFAULT_SAM_PROMPTS = "speech bubble"
"""What ``--use-sam`` means by *text* until told otherwise. A balloon is the one
text region SAM3 reads better than a stroke segmenter does — it is a closed
shape, and its interior is as untrainable as the lettering in it."""


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
    # smp costs ~2s to import (timm + torchvision eagerly), and `--help` / the
    # GUI schema dump import this module just for its parser.
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

    Component-wise rather than box-wise on purpose: the blk head decides *which*
    strokes are text, the UNet++ still decides where each of them ends, so a
    kept letter keeps its own outline instead of the rectangle around it.
    """
    boxmask = np.zeros(mask.shape, bool)
    for x0, y0, x1, y1 in _ctd_text_boxes(ctd_forward, img):
        boxmask[y0:y1, x0:x1] = True
    _, lab = cv2.connectedComponents(mask)
    keep_ids = np.unique(lab[boxmask])
    return np.isin(lab, keep_ids[keep_ids != 0]).astype(np.uint8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_mask_dir_args(parser, mask_default=WS.MASKS_MIT)

    sam = gated_group(
        parser,
        "SAM3 prompts",
        gate="use_sam",
        default=False,
        help="Ground SAM3 on --sam-prompts and mask what it finds. Off by "
        "default: it is a second set of weights to load, and it answers a "
        "different question than the segmenter below — turn it on for "
        "balloons, which are a shape rather than a stroke",
    )
    sam.add_argument(
        "--sam-prompts",
        dest="sam_prompts",
        type=str,
        default=DEFAULT_SAM_PROMPTS,
        help=f"Comma-separated SAM3 text prompts for the regions to mask OUT "
        f"(default `{DEFAULT_SAM_PROMPTS}`; e.g. `speech bubble,sign,"
        f"watermark`). Same polarity as `generate_masks --prompts`: everything "
        f"named here is ignored in the loss",
    )
    sam.add_argument(
        "--sam-threshold",
        dest="sam_threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence floor for a detection (default: 0.5)",
    )
    add_checkpoint_arg(sam)

    mit = gated_group(
        parser,
        "MIT text segmentation",
        gate="use_mit",
        default=True,
        help="Run the UNet++ text segmenter — the stroke-accurate half, and "
        "the only one that finds lettering outside a balloon",
    )
    mit.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model.pth (downloads from HuggingFace if not specified)",
    )
    mit.add_argument(
        "--text-threshold",
        type=float,
        default=0.8,
        help="Text segmentation threshold (default: %(default)s)",
    )
    mit.add_argument(
        "--ctd-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep only mask components overlapping a comictextdetector text "
            "block — drops UNet++ false positives on halos/decorative line art "
            "(--no-ctd-gate = raw UNet++ masks, restores pre-2026-07 behavior). "
            "The net is the download catalog's `ctd_onnx` row; a missing one "
            "warns and leaves the masks ungated"
        ),
    )

    add_force_arg(parser)
    add_device_arg(parser)
    parser.add_argument(
        "--dilate",
        type=int,
        default=3,
        help="Mask dilation in pixels, applied once to the union (default: %(default)s)",
    )
    add_workers_arg(parser)
    add_walk_args(parser)
    return parser


def detectors(parser: argparse.ArgumentParser, args) -> tuple[bool, tuple[str, ...]]:
    """``(run the segmenter, the SAM3 prompts)`` — or exit saying why not.

    A shut drawer contributes nothing, so both shut is a run that would load no
    model, walk the whole tree and write nothing. Said here, before the first
    weight is read, rather than discovered from an empty mask directory.
    """
    sam_prompts = prompt_list(args.sam_prompts) if args.use_sam else ()
    if args.use_sam and not sam_prompts:
        parser.error("--use-sam with no --sam-prompts: nothing for SAM3 to ground on")
    if not args.use_mit and not sam_prompts:
        parser.error("nothing to detect: pass --use-mit and/or --use-sam")
    return bool(args.use_mit), sam_prompts


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    use_mit, sam_prompts = detectors(parser, args)
    args.device = resolve_device(args.device)

    dilate_kernel = (
        np.ones((args.dilate, args.dilate), dtype=np.uint8) if args.dilate > 0 else None
    )

    model = None
    ctd = None
    if use_mit:
        print("Loading text segmentation model...")
        model = _load_model(args.model_path, device=args.device)
        if args.ctd_gate:
            ctd_onnx = default_ctd_onnx_path()
            if ctd_onnx.exists():
                ctd = _load_ctd(str(ctd_onnx), device=args.device)
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
            resolve_path(args.checkpoint) if args.checkpoint else None, args.device
        )
        print(f"SAM3 prompts: {', '.join(sam_prompts)}")

    # Home-anchored like the SAM3 generator's — see its note.
    image_dir = resolve_path(args.image_dir)
    masks_dir = resolve_path(args.mask_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    work_items = plan_mask_jobs(
        image_dir,
        masks_dir,
        recursive=args.recursive,
        pattern=args.path_pattern,
        force=args.force,
    )

    total = len(work_items)
    if total == 0:
        print("No images to process.")
        return

    pool = ThreadPoolExecutor(max_workers=args.workers)
    amp = autocast(args.device)

    pbar = tqdm(total=total, desc="Generating masks")
    for image_path, mask_path in work_items:
        pil_image = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_image)
        h, w = img_np.shape[:2]

        pbar.update(1)

        text_mask = np.zeros((h, w), dtype=np.uint8)
        # Worth distinguishing in the progress line: nothing masked because the
        # page has no text reads the same as nothing masked because the gate
        # threw all of it away, and only the second is a knob to reconsider.
        gated_away = False

        if model is not None:
            unet = _detect_mask(
                model,
                img_np,
                device=args.device,
                text_threshold=args.text_threshold,
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
                        args.sam_threshold,
                    ),
                )

        if not text_mask.any():
            why = " (ctd-gated)" if gated_away else ""
            pbar.set_postfix_str(f"{image_path.name}: skipped{why}")
            continue

        # One dilation over the union, not one per detector: the two overlap on
        # a lettered balloon, and dilating twice would grow that seam twice.
        if dilate_kernel is not None:
            text_mask = cv2.dilate(text_mask, dilate_kernel, iterations=1)

        write_ignore_mask(mask_path, text_mask, pool=pool)

        masked_pct = 100 * np.count_nonzero(text_mask) / (w * h)
        pbar.set_postfix_str(f"{image_path.name}: {masked_pct:.1f}%")

    pbar.close()
    pool.shutdown(wait=True)
    print(f"Masks saved to {masks_dir}/")


if __name__ == "__main__":
    main()
