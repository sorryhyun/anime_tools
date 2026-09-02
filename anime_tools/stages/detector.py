"""The SAM3 subject detector both SAM3 stages run, built from a
:class:`~anime_tools.stages.requests.DetectionRequest`.

Torch and sam3 are imported inside :func:`build_detect_fn`; importing this
module only installs ``_sam3``'s ``np.bool`` alias.
"""

from __future__ import annotations

import numpy as np

from anime_tools._device import resolve_device
from anime_tools._env import resolve_path

# Importing _sam3 also installs the `np.bool` alias sam3 needs before it loads.
from anime_tools.masking._sam3 import (
    autocast,
    ground_with_soft_prompt,
    load_sam3,
    make_processor,
)
from anime_tools.stages.instance_detection import (
    Detection,
    load_soft_prompt,
    resolve_prompt_embed,
)
from anime_tools.stages.requests import DetectionRequest


def build_detect_fn(
    detection: DetectionRequest,
    *,
    device: str | None = None,
    model=None,
    processor=None,
):
    """SAM3 text-prompt detector returning per-instance boxes + masks.

    Pass ``model``/``processor`` from a previous call to build a second detector
    (different prompt) on the same loaded SAM3; ``load_sam3`` also caches per
    process, so a second stage in the same interpreter reuses the weights.

    GOTCHA 1: ``Sam3Processor`` applies its own ``confidence_threshold`` before
    the caller sees the boxes, so it must be built at the *lowest* threshold any
    retry might ask for (``detection.floor``); the score gate is applied on top
    in ``detect``.

    GOTCHA 2: ``detect_subjects`` calls back per retry and per part prompt on the
    same image, so encoding and raw detections are memoised per image/prompt.

    Returns ``(detect, part_detect, model, processor)``.
    """
    import torch

    floor = detection.floor
    device = resolve_device(device)
    if model is None:
        print("Loading SAM3...", flush=True)
        model, fresh = load_sam3(
            resolve_path(detection.checkpoint), device, confidence_threshold=floor
        )
        if processor is None:
            processor = fresh
    if processor is None or processor.confidence_threshold > floor:
        processor = make_processor(model, floor)
    subject = detection.prompt
    soft_prompt = None
    embed_path = resolve_prompt_embed(detection.prompt_embed)
    if embed_path is not None:
        soft_prompt = load_soft_prompt(embed_path, device)
        print(f"soft prompt: {embed_path} (replaces {subject!r})", flush=True)
    cache: dict[str, object] = {"key": None, "state": None, "dets": {}}
    amp = autocast(device)

    def _ground(image, prompt: str) -> list[Detection]:
        """Raw detections for one prompt, reusing this image's encoded state."""
        if cache["key"] is not image:
            with amp:
                cache["state"] = processor.set_image(image)
            cache["key"] = image
            cache["dets"] = {}
        memo: dict = cache["dets"]  # type: ignore[assignment]
        if prompt in memo:
            return memo[prompt]
        with amp:
            if soft_prompt is not None and prompt == subject:
                # Learned prompt tensor stands in for the subject phrase, so the
                # text encode is skipped.
                out = ground_with_soft_prompt(
                    processor, model, cache["state"], soft_prompt
                )
            else:
                out = processor.set_text_prompt(prompt=prompt, state=cache["state"])
        masks = out.get("masks")
        source = "subject" if prompt == subject else prompt
        dets: list[Detection] = []
        for i, (box, score) in enumerate(zip(out["boxes"], out["scores"])):
            coords = box.tolist() if torch.is_tensor(box) else list(box)
            mask = None
            if masks is not None and i < len(masks):
                m = masks[i]
                mask = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
            dets.append(
                Detection(
                    box=tuple(float(v) for v in coords),
                    score=float(score),
                    mask=mask,
                    source=source,
                )
            )
        memo[prompt] = dets
        return dets

    def detect(image, score_threshold: float) -> list[Detection]:
        return [d for d in _ground(image, subject) if d.score >= score_threshold]

    def part_detect(image, prompt: str, score_threshold: float) -> list[Detection]:
        return [d for d in _ground(image, prompt) if d.score >= score_threshold]

    return detect, part_detect, model, processor
