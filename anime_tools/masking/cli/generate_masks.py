"""SAM3 subject masks, written to ``workspace/masks_sam/``.

``--prompts`` names what is masked OUT (ignored in the loss); ``--focus-prompts`` names
what is kept, everything else masked out. Give both and the focus region survives minus
the ignore regions. The subject prompt is served by a learned soft prompt by default
(``--prompt_embed``); pass ``none`` for the plain text prompt.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from anime_tools import workspace as WS
from anime_tools._device import add_device_arg, resolve_device
from anime_tools._env import resolve_path
from anime_tools.masking._masks import (
    add_force_arg,
    add_mask_dir_args,
    add_walk_args,
    add_workers_arg,
    coverage_pct,
    mask_run,
    write_ignore_mask,
    write_mask,
)

# Importing _sam3 also installs the `np.bool` alias sam3 needs before it loads.
from anime_tools.masking._sam3 import (
    SUBJECT_PROMPT,
    add_checkpoint_arg,
    add_prompt_embed_arg,
    autocast,
    detect_union,
    load_sam3,
    prompt_list,
)

# Torch-free at import (the safetensors read is deferred); the same two helpers the
# position stage resolves its --prompt_embed through.
from anime_tools.stages.instance_detection import load_soft_prompt, resolve_prompt_embed


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


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

    print("Loading SAM3 model...")
    model, processor = load_sam3(
        resolve_path(args.checkpoint) if args.checkpoint else None, args.device
    )

    # The soft prompt stands in for the subject phrase only; every other prompt still
    # goes through the text encoder.
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

    def detect(state, prompts, shape) -> np.ndarray:
        """This run's SAM3 pass: the shared union, with the soft prompt bound."""
        return detect_union(
            processor,
            model,
            state,
            prompts,
            shape,
            args.threshold,
            soft_prompt=soft_prompt,
        )

    batch_size = args.batch_size
    amp = autocast(args.device)

    with mask_run(args) as run:
        # Prefetch images ahead of GPU to keep it saturated.
        prefetch = min(args.workers, run.total)
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


if __name__ == "__main__":
    main()
