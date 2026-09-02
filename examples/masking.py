"""Training masks: SAM3 subject masks, text masks, and their merge.

    python examples/masking.py --home ~/data           # print the three requests
    python examples/masking.py --home ~/data --run     # generate + merge

Three request objects in ``anime_tools.masking``, each a CLI
(``python -m anime_tools.masking.cli.{generate_masks,generate_masks_mit,merge_masks}``,
hyphenated flags). A mask is an 8-bit L PNG named ``{stem}_mask.png`` at the
image's relative path under the mask dir: 255 keeps a pixel in the loss, 0
ignores it. Each generator writes its **own** tree (``workspace/masks_sam/``,
``workspace/masks_mit/``) because both name the file identically; the merge
takes the pixel-wise minimum into ``workspace/masks/``, which Export publishes.

Weights: ``python -m anime_tools.downloads sam3 soft_prompt mit_text ctd_onnx``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--home")
    p.add_argument("--image-dir", default="workspace/resized")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.home:
        os.environ["ANIME_TOOLS_HOME"] = str(Path(args.home).expanduser().resolve())

    from anime_tools.masking import MergeMasksRequest, MitMaskRequest, SamMaskRequest

    # --- subject masks -------------------------------------------------------
    # focus_prompts = keep ONLY these (default: the subject, served by a learned
    # soft prompt); prompts = mask OUT these. Both: focus minus ignore.
    sam = SamMaskRequest(
        image_dir=args.image_dir,
        recursive=True,
        focus_prompts=("girl",),
        prompts=("speech bubble",),
        dilate=5,
    )
    print("$ python -m anime_tools.masking.cli.generate_masks", *sam.to_argv())

    # --- text masks: two detectors behind two switches -------------------------
    # use_mit: the UNet++ stroke segmenter (lettering, gated by a
    # comictextdetector text-block net); use_sam: SAM3 on sam_prompts (balloons —
    # a shape, not a stroke). Both off is the one request the stage refuses.
    mit = MitMaskRequest(
        image_dir=args.image_dir,
        recursive=True,
        use_mit=True,
        text_threshold=0.8,
        use_sam=True,
        sam_prompts=("speech bubble", "sign"),
        dilate=3,
    )
    print("$ python -m anime_tools.masking.cli.generate_masks_mit", *mit.to_argv())
    try:
        MitMaskRequest(image_dir=args.image_dir, use_mit=False, use_sam=False)
    except ValueError as e:
        print("  refused:", e)

    # --- merge ---------------------------------------------------------------
    # Positional inputs default to the two generators' trees; a missing one is
    # skipped, so running one generator is a valid half of this.
    merge = MergeMasksRequest()
    print("$ python -m anime_tools.masking.cli.merge_masks", *merge.to_argv())

    # --- where a mask lives (no models) ------------------------------------------
    from anime_tools._env import resolve_path
    from anime_tools.masking._masks import iter_masks, mask_path_for

    image = resolve_path(args.image_dir) / "char_a" / "001.png"
    print(
        "\nmask for",
        image.name,
        "→",
        mask_path_for(
            image, resolve_path(args.image_dir), resolve_path(merge.output_dir)
        ),
    )
    if not args.run:
        return

    from anime_tools.masking import run_merge_masks, run_mit_masks, run_sam_masks

    run_sam_masks(sam)  # load_sam3 is cached per process on its arguments,
    run_mit_masks(mit)  # so the text pass reuses the subject pass's model
    n = run_merge_masks(merge)
    print(f"\n{n} merged mask(s):")
    from PIL import Image

    for rel_dir, path in iter_masks(resolve_path(merge.output_dir)):
        mask = Image.open(path)  # mode L
        kept = sum(1 for v in mask.getdata() if v) / (mask.width * mask.height)
        print(f"  {rel_dir or '.'}/{path.name}: {mask.size}, {kept:.0%} kept")


if __name__ == "__main__":
    main()
