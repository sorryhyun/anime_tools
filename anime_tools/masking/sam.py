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

# Importing _sam3 also installs the `np.bool` alias sam3 needs before it loads.
from anime_tools.masking._sam3 import (
    SUBJECT_PROMPT,
    autocast,
    detect_union,
    load_sam3,
)
from anime_tools.masking.requests import SamMaskRequest

# Torch-free at import (the safetensors read is deferred); the same two helpers the
# position stage resolves its --prompt_embed through.
from anime_tools.stages.instance_detection import load_soft_prompt, resolve_prompt_embed


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def run_sam_masks(req: SamMaskRequest) -> MaskRun:
    """Write ``{stem}_mask.png`` under ``req.mask_dir`` for every image the walk
    plans; returns the run (its ``items`` are what was planned)."""
    device = resolve_device(req.device)
    ignore_prompts = req.prompts
    focus_prompts = req.focus_prompts
    kernel = np.ones((req.dilate,) * 2, dtype=np.uint8) if req.dilate > 0 else None

    print("Loading SAM3 model...")
    model, processor = load_sam3(
        resolve_path(req.checkpoint) if req.checkpoint else None, device
    )

    # The soft prompt stands in for the subject phrase only; every other prompt still
    # goes through the text encoder.
    soft_prompt = None
    embed_path = resolve_prompt_embed(req.prompt_embed)
    if embed_path is not None:
        if SUBJECT_PROMPT in ignore_prompts + focus_prompts:
            soft_prompt = load_soft_prompt(embed_path, device)
            print(f"soft prompt: {embed_path} (replaces {SUBJECT_PROMPT!r})")
        else:
            print(
                f"NOTE: --prompt_embed is the {SUBJECT_PROMPT!r} prompt, which "
                f"neither --prompts nor --focus-prompts asks for — every prompt "
                f"here is textual"
            )

    def detect(state, prompts, shape) -> np.ndarray:
        """This run's SAM3 pass: the shared union, with the soft prompt bound."""
        return detect_union(
            processor,
            model,
            state,
            prompts,
            shape,
            req.threshold,
            soft_prompt=soft_prompt,
        )

    batch_size = req.batch_size
    amp = autocast(device)

    with mask_run(req) as run:
        # Prefetch images ahead of GPU to keep it saturated.
        prefetch = min(req.workers, run.total)
        load_futures = [
            run.pool.submit(load_image, run.items[j][0]) for j in range(prefetch)
        ]
        save_futures = []

        for batch_start in range(0, run.total, batch_size):
            batch_end = min(batch_start + batch_size, run.total)
            batch = []
            for i in range(batch_start, batch_end):
                image = load_futures[i].result()
                if i + prefetch < run.total:
                    load_futures.append(
                        run.pool.submit(load_image, run.items[i + prefetch][0])
                    )
                batch.append((run.items[i], image))

            with amp:
                states = []
                for (image_path, mask_path), image in batch:
                    states.append(
                        (image_path, mask_path, image, processor.set_image(image))
                    )

                for image_path, mask_path, image, inference_state in states:
                    w, h = image.size
                    run.advance()

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
                        save_futures.append(
                            write_mask(mask_path, trainable, pool=run.pool)
                        )
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
