"""SAM3 subject masks: mask a prompt OUT, or keep ONLY a prompt.

Two polarities over the same detector. ``--prompts`` names what is masked out
(ignored in the loss — speech bubbles, a watermark); ``--focus-prompts`` names
what is kept, everything else being masked out, which is how the subject is
isolated from a background. Give both and the focus region is what survives
minus the ignore regions.

The subject prompt is served by a learned soft prompt by default
(``--prompt_embed``, the textual inversion of ``anime girl``): same recall,
markedly less junk than the bare word. Pass ``none`` for the plain text prompt.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from anime_tools import workspace as WS
from anime_tools._device import resolve_device
from anime_tools._env import resolve_path
from anime_tools.masking._masks import (
    add_device_arg,
    add_force_arg,
    add_mask_dir_args,
    add_walk_args,
    add_workers_arg,
    plan_mask_jobs,
    write_ignore_mask,
    write_mask,
)

# Importing _sam3 also installs the `np.bool` alias sam3 needs before it loads.
from anime_tools.masking._sam3 import (
    SUBJECT_PROMPT,
    add_checkpoint_arg,
    add_prompt_embed_arg,
    ground_with_soft_prompt,
    load_sam3,
)

# Torch-free at import (the safetensors read is deferred), and the same two
# helpers the position stage resolves its --prompt_embed through, so `none`,
# a missing shipped default and an explicit bad path mean one thing everywhere.
from anime_tools.stages.instance_detection import load_soft_prompt, resolve_prompt_embed


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


_NO_PROMPTS = {"none", "off"}


def prompt_list(spec: str) -> tuple[str, ...]:
    """A comma-separated prompt flag as the tuple of prompts it names.

    ``none`` / ``off`` mean *no prompts*, the word ``--prompt_embed`` already
    takes for the same job. Emptying the field is not enough to say it: the GUI
    omits a flag whose value is blank, so a cleared ``--focus-prompts`` would
    come back as its default rather than as "focus on nothing".
    """
    if spec.strip().lower() in _NO_PROMPTS:
        return ()
    return tuple(t.strip() for t in spec.split(",") if t.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_mask_dir_args(parser, mask_default=WS.MASKS_SAM)
    parser.add_argument(
        "--prompts",
        type=str,
        default="",
        help="Comma-separated SAM3 text prompts to mask OUT — these regions are "
        "ignored in the loss (e.g. `speech bubble,text`)",
    )
    parser.add_argument(
        "--focus-prompts",
        dest="focus_prompts",
        type=str,
        default=SUBJECT_PROMPT,
        help="Comma-separated prompts to keep ONLY: everything outside them is "
        f"masked out. Default `{SUBJECT_PROMPT}` (the subject), so a bare run "
        "isolates the subject from her background; pass `none` to keep nothing "
        "in and use --prompts alone",
    )
    add_prompt_embed_arg(parser)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence floor for a detection (default: 0.5)",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=5,
        help="Mask dilation in pixels, 0 = off (default: 5)",
    )
    add_force_arg(parser)
    add_checkpoint_arg(parser)
    add_device_arg(parser)
    add_workers_arg(parser, help="I/O workers for loading/saving (default: 4)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Images to process in parallel (default: 1)",
    )
    add_walk_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    ignore_prompts = prompt_list(args.prompts)
    focus_prompts = prompt_list(args.focus_prompts)
    if not ignore_prompts and not focus_prompts:
        parser.error("nothing to mask: pass --prompts and/or --focus-prompts")
    kernel = np.ones((args.dilate,) * 2, dtype=np.uint8) if args.dilate > 0 else None

    import torch

    # Anchored, like every stage's roots: the defaults above are written
    # home-relative, so a run from another directory has to mean the same
    # tree the GUI and the merge do.
    image_dir = resolve_path(args.image_dir)
    masks_dir = resolve_path(args.mask_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SAM3 model...")
    model, processor = load_sam3(
        resolve_path(args.checkpoint) if args.checkpoint else None, args.device
    )

    # The soft prompt stands in for the subject phrase wherever it was asked
    # for; a prompt it does not name still goes through the text encoder.
    soft_prompt = None
    embed_path = resolve_prompt_embed(args.prompt_embed)
    if embed_path is not None:
        if SUBJECT_PROMPT in ignore_prompts + focus_prompts:
            soft_prompt = load_soft_prompt(embed_path, args.device)
            print(f"soft prompt: {embed_path} (replaces {SUBJECT_PROMPT!r})")
        else:
            print(
                f"NOTE: --prompt_embed is the {SUBJECT_PROMPT!r} prompt, which "
                f"neither --prompts nor --focus-prompts asks for — every prompt "
                f"here is textual"
            )

    def detect_union(state, prompts, shape, threshold) -> np.ndarray:
        """OR-combine SAM3 detections for every prompt into one binary mask."""
        h, w = shape
        out = np.zeros((h, w), dtype=np.uint8)
        for prompt in prompts:
            if soft_prompt is not None and prompt == SUBJECT_PROMPT:
                output = ground_with_soft_prompt(processor, model, state, soft_prompt)
            else:
                output = processor.set_text_prompt(state=state, prompt=prompt)
            for mask, score in zip(output["masks"], output["scores"]):
                if score < threshold:
                    continue
                mask_np = (
                    mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
                )
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                out = np.maximum(out, (mask_np > 0.5).astype(np.uint8))
        return out

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

    batch_size = args.batch_size
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    pool = ThreadPoolExecutor(max_workers=args.workers)

    # Prefetch images ahead of GPU to keep it saturated.
    prefetch = min(args.workers, total)
    load_futures = [pool.submit(load_image, work_items[j][0]) for j in range(prefetch)]
    save_futures = []

    pbar = tqdm(total=total, desc="Generating masks")
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = []
        for i in range(batch_start, batch_end):
            image = load_futures[i].result()
            if i + prefetch < total:
                load_futures.append(
                    pool.submit(load_image, work_items[i + prefetch][0])
                )
            batch.append((work_items[i], image))

        with autocast:
            states = []
            for (image_path, mask_path), image in batch:
                states.append(
                    (image_path, mask_path, image, processor.set_image(image))
                )

            for image_path, mask_path, image, inference_state in states:
                w, h = image.size
                pbar.update(1)

                ignore_mask = np.zeros((h, w), dtype=np.uint8)
                if ignore_prompts:
                    ignore_mask = detect_union(
                        inference_state, ignore_prompts, (h, w), args.threshold
                    )
                    if kernel is not None and ignore_mask.any():
                        ignore_mask = cv2.dilate(ignore_mask, kernel, iterations=1)

                if focus_prompts:
                    focus_mask = detect_union(
                        inference_state, focus_prompts, (h, w), args.threshold
                    )
                    if kernel is not None and focus_mask.any():
                        focus_mask = cv2.dilate(focus_mask, kernel, iterations=1)
                    if not focus_mask.any():
                        # Subject not found — leave unmasked (train fully) rather
                        # than zeroing out the whole loss.
                        pbar.set_postfix_str(f"{image_path.name}: focus not found")
                        continue
                    # ONLY the focus subject, minus any ignore-prompt regions.
                    trainable = focus_mask * (1 - ignore_mask)
                    save_futures.append(write_mask(mask_path, trainable, pool=pool))
                    train_pct = 100 * np.count_nonzero(trainable) / (w * h)
                    pbar.set_postfix_str(f"{image_path.name}: train {train_pct:.1f}%")
                    continue

                if not ignore_mask.any():
                    pbar.set_postfix_str(f"{image_path.name}: skipped")
                    continue

                save_futures.append(
                    write_ignore_mask(mask_path, ignore_mask, pool=pool)
                )
                masked_pct = 100 * np.count_nonzero(ignore_mask) / (w * h)
                pbar.set_postfix_str(f"{image_path.name}: {masked_pct:.1f}%")

    pbar.close()

    for f in save_futures:
        f.result()
    pool.shutdown()

    print(f"Masks saved to {masks_dir}/")


if __name__ == "__main__":
    main()
