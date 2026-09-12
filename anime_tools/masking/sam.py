"""SAM3 subject masks — :func:`run_sam_masks` over a
:class:`~anime_tools.masking.requests.SamMaskRequest`.

The CLI (``cli/generate_masks.py``) is a shell over this module.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from anime_tools._device import resolve_device
from anime_tools._env import resolve_path
from anime_tools.masking._masks import (
    MaskRun,
    coverage_pct,
    mask_run,
    write_ignore_mask,
    write_mask,
)

# Torch-free at import (the safetensors read is deferred); the same prompt
# vocabulary the position stage resolves its --prompt_embed through.
from anime_tools.masking._prompts import (
    SUBJECT_PROMPT,
    load_soft_prompt,
    resolve_prompt_embed,
)

# Importing _sam3 also installs the `np.bool` alias sam3 needs before it loads.
from anime_tools.masking._sam3 import autocast, detect_union, load_sam3
from anime_tools.masking.requests import MaskPrompt, SamMaskRequest


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _resolve_prompts(req: SamMaskRequest, device: str) -> tuple[list, list]:
    """``(keep, ignore)`` as ``detect_union`` takes them: text as the string, soft
    as its loaded tensors. A soft entry naming the shipped default that is not
    downloaded falls back to the text it was learned from, as ``--prompt_embed``
    always has; any other missing file is an error before a model loads."""
    loaded: dict = {}

    def resolve(m: MaskPrompt):
        if m.kind == "text":
            return m.value
        path = resolve_prompt_embed(m.value)
        if path is None:
            return SUBJECT_PROMPT
        if path not in loaded:
            loaded[path] = load_soft_prompt(path, device)
            print(f"soft prompt ({m.role}): {path}")
        return loaded[path]

    return [resolve(m) for m in req.keep], [resolve(m) for m in req.ignore]


def run_sam_masks(req: SamMaskRequest) -> MaskRun:
    """Write ``{stem}_mask.png`` under ``req.mask_dir`` for every image the walk
    plans; returns the run (its ``items`` are what was planned)."""
    device = resolve_device(req.device)
    kernel = np.ones((req.dilate,) * 2, dtype=np.uint8) if req.dilate > 0 else None

    print("Loading SAM3 model...")
    model, processor = load_sam3(
        resolve_path(req.checkpoint) if req.checkpoint else None, device
    )
    focus_prompts, ignore_prompts = _resolve_prompts(req, device)

    def detect(state, prompts, shape) -> np.ndarray:
        """This run's SAM3 pass: the shared union at this run's threshold."""
        return detect_union(processor, model, state, prompts, shape, req.threshold)

    with mask_run(req) as run, autocast(device):
        # Prefetch images ahead of GPU to keep it saturated. ``run.workers`` is
        # the clamped pool size, so the depth is never zero.
        prefetch = min(run.workers, run.total)
        load_futures = [
            run.pool.submit(load_image, run.items[j][0]) for j in range(prefetch)
        ]
        save_futures = []

        # One image at a time: ``processor.set_image`` is a single-image encode,
        # so an outer batch only held every inference state resident until its
        # detect loop drained — memory for no throughput. That is why the stage
        # has no ``--batch-size``; the I/O pool is what keeps the GPU fed.
        for i in range(run.total):
            image = load_futures[i].result()
            # A ``Future`` keeps its result: holding the whole list would pin
            # every decoded RGB image for the length of the run, which on a
            # large tree is tens of gigabytes. The slot stays so the indexing
            # above still lines up with ``run.items``.
            load_futures[i] = None
            if i + prefetch < run.total:
                load_futures.append(
                    run.pool.submit(load_image, run.items[i + prefetch][0])
                )

            image_path, mask_path = run.items[i]
            w, h = image.size
            run.advance()
            inference_state = processor.set_image(image)

            ignore_mask = np.zeros((h, w), dtype=np.uint8)
            if ignore_prompts:
                ignore_mask = detect(inference_state, ignore_prompts, (h, w))
                if kernel is not None and ignore_mask.any():
                    ignore_mask = cv2.dilate(ignore_mask, kernel, iterations=1)

            if focus_prompts:
                focus_mask = detect(inference_state, focus_prompts, (h, w))
                if kernel is not None and focus_mask.any():
                    focus_mask = cv2.dilate(focus_mask, kernel, iterations=1)
                if not focus_mask.any():
                    # Subject not found — leave unmasked (train fully) rather
                    # than zeroing out the whole loss.
                    run.note(image_path, "focus not found")
                    continue
                trainable = focus_mask * (1 - ignore_mask)
                save_futures.append(write_mask(mask_path, trainable, pool=run.pool))
                run.note(image_path, f"train {coverage_pct(trainable):.1f}%")
                continue

            if not ignore_mask.any():
                run.note(image_path, "skipped")
                continue

            save_futures.append(
                write_ignore_mask(mask_path, ignore_mask, pool=run.pool)
            )
            run.note(image_path, f"{coverage_pct(ignore_mask):.1f}%")

        # Inside the `with`, before the pool is shut down: a save that raised is a mask
        # that is not there, and this is the only place it can be seen.
        for f in save_futures:
            f.result()

    return run
