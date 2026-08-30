"""Shared SAM3 plumbing for the soft-prompt (textual inversion) bench.

The learnable object is the **post-tower prompt**: SAM3 encodes a text prompt
into ``language_features`` ``(L=32, 1, 256)`` + ``language_mask`` ``(1, 32)``
and the rest of the model only ever sees that pair (`SAM3Image._encode_prompt`
concatenates it in front of the geometric prompt). So a soft prompt is just a
``(32, 1, 256)`` tensor initialised from a real phrase and moved by gradient —
the trunk, fusion encoder, decoder and scoring stay frozen.

Soft prompts are saved as safetensors ``{"language_features", "language_mask",
"language_embeds"}`` plus string metadata (init phrase, steps); load with
:func:`load_soft_prompt` and install with :func:`install_prompt`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# sam3 pins numpy<2 and still spells np.bool
if not hasattr(np, "bool"):
    np.bool = np.bool_

import torch
from safetensors.torch import save_file

# Canonical home is library/ — bench/ is in scripts/update.py's PRESERVE_DIRS,
# so anything the shipped preprocess path needs cannot live only here.
from anime_tools.stages.instance_detection import (
    SOFT_PROMPT_KEYS as PROMPT_KEYS,
)
from anime_tools.stages.instance_detection import load_soft_prompt  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
MASK_RES = 288  # native `pred_masks` resolution


def load_sam3(checkpoint: str | Path, device: str = "cuda"):
    """Frozen SAM3 image model + processor (processor built at a low floor —
    the score gate is re-applied by the caller, see `build_detect_fn`)."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    # SAM3 leaves two activation-checkpoint paths on even in eval mode (the
    # maskformer pixel decoder + the MHA in `model_misc`); they recompute the
    # forward inside our backward for nothing — the trunk is frozen and VRAM
    # is not the bottleneck at the batch sizes the prompt trainer uses.
    n = 0
    for m in model.modules():
        for attr in ("act_ckpt", "use_act_checkpoint"):
            if getattr(m, attr, False) is True:
                setattr(m, attr, False)
                n += 1
    if n:
        print(f"sam3: disabled activation checkpointing on {n} modules", flush=True)
    processor = Sam3Processor(model, confidence_threshold=0.0)
    return model, processor


@torch.no_grad()
def encode_image(model, processor, image) -> dict:
    """`Sam3Processor.set_image` minus `inference_mode` — inference tensors can't
    take part in an autograd graph, and the prompt gradient has to flow through
    the fusion encoder that consumes these features."""
    from torchvision.transforms import v2

    x = v2.functional.to_image(image).to(processor.device)
    x = processor.transform(x).unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.backbone.forward_image(x)


@torch.no_grad()
def encode_text(model, prompt: str, device: str = "cuda") -> dict[str, torch.Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.backbone.forward_text([prompt], device=device)
    return {k: out[k] for k in PROMPT_KEYS}


def install_prompt(backbone_out: dict, prompt: dict[str, torch.Tensor]) -> dict:
    backbone_out.update({k: prompt[k] for k in PROMPT_KEYS})
    return backbone_out


def preprocess_image(processor, image) -> torch.Tensor:
    """CPU side of `Sam3Processor.set_image`: PIL → transformed `(3, S, S)` tensor.
    Safe to run in DataLoader workers; `encode_images` stacks and moves them."""
    from torchvision.transforms import v2

    return processor.transform(v2.functional.to_image(image))


@torch.no_grad()
def encode_images(model, processor, batch: torch.Tensor) -> dict:
    """One trunk forward over a pre-transformed `(B, 3, S, S)` batch."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.backbone.forward_image(
            batch.to(processor.device, non_blocking=True)
        )


def _find_stage(processor, n: int):
    """`processor.find_stage` for a batch of `n` images sharing prompt 0."""
    if n == 1:
        return processor.find_stage
    from sam3.model.data_misc import FindStage

    dev = processor.device
    return FindStage(
        img_ids=torch.arange(n, device=dev, dtype=torch.long),
        text_ids=torch.zeros(n, device=dev, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def ground(model, processor, backbone_out: dict) -> dict:
    """Raw DETR outputs for the prompt already installed in ``backbone_out``.
    Differentiable w.r.t. ``language_features`` (no `inference_mode` here).
    Batched: every image in ``vision_features`` is grounded against prompt 0."""
    n = backbone_out["vision_features"].shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.forward_grounding(
            backbone_out=backbone_out,
            find_input=_find_stage(processor, n),
            geometric_prompt=model._get_dummy_prompt(num_prompts=n),
            find_target=None,
        )


def slice_out(out: dict, i: int) -> dict:
    """Image ``i`` of a batched grounding output as a batch-1 output."""
    keys = ("pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec")
    return {k: out[k][i : i + 1] for k in keys}


def scores_from(out: dict) -> torch.Tensor:
    """`(200,)` final probabilities — objectness × presence, as the processor does."""
    probs = out["pred_logits"].float().sigmoid().squeeze(-1)
    presence = out["presence_logit_dec"].float().sigmoid()
    return (probs * presence).squeeze(0)


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def box_xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], -1)


@torch.no_grad()
def proposals(out: dict, floor: float) -> list[dict]:
    """Postprocess one grounding pass into a list of proposals at ``floor``
    with normalized xyxy boxes and low-res mask logits (288²). Adds the box
    fill (mask ∩ box / box area) and area fraction the NMS audit uses."""
    scores = scores_from(out)
    keep = (scores > floor).nonzero().flatten()
    boxes = box_cxcywh_to_xyxy(out["pred_boxes"][0].float())[keep].clamp(0, 1)
    masks = out["pred_masks"][0].float()[keep]
    rows = []
    for i in range(len(keep)):
        x0, y0, x1, y1 = boxes[i].tolist()
        m = masks[i] > 0
        px = (
            max(0, int(x0 * MASK_RES)),
            max(0, int(y0 * MASK_RES)),
            min(MASK_RES, int(x1 * MASK_RES)),
            min(MASK_RES, int(y1 * MASK_RES)),
        )
        window = m[px[1] : px[3], px[0] : px[2]]
        rows.append(
            {
                "query": int(keep[i]),
                "score": float(scores[keep[i]]),
                "box": [x0, y0, x1, y1],
                "fill": float(window.float().mean()) if window.numel() else 0.0,
                "area_frac": (x1 - x0) * (y1 - y0),
                "mask": m,
            }
        )
    return rows


def iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(rows: list[dict], iou_threshold: float = 0.65) -> list[dict]:
    """Greedy score-ranked NMS mirroring `dedupe_detections`' IoU rule."""
    keep: list[dict] = []
    for r in sorted(rows, key=lambda r: -r["score"]):
        if all(iou_xyxy(r["box"], k["box"]) < iou_threshold for k in keep):
            keep.append(r)
    return keep


def save_soft_prompt(path: Path, prompt: dict[str, torch.Tensor], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: prompt[k].detach().cpu().contiguous() for k in PROMPT_KEYS}
    save_file(tensors, str(path), metadata={k: json.dumps(v) for k, v in meta.items()})


def soft_prompt_meta(path: str | Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as f:
        return {k: json.loads(v) for k, v in (f.metadata() or {}).items()}
