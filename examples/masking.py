"""Training masks: SAM3 subject masks and their merge.

    python examples/masking.py --home ~/data           # print the two requests
    python examples/masking.py --home ~/data --run     # generate + merge

Two request objects in ``anime_tools.masking``, each a CLI
(``python -m anime_tools.masking.cli.{generate_masks,merge_masks}``, hyphenated
flags). A mask is an 8-bit L PNG named ``{stem}_mask.png`` at the image's
relative path under the mask dir: 255 keeps a pixel in the loss, 0 ignores it.
The generator writes its **own** tree (``workspace/masks_sam/``), never the
``masks`` root, so a hand-made tree listed beside it at merge time cannot
overwrite a SAM3 mask at the same relative path; the merge takes the pixel-wise
minimum into ``workspace/masks/``, which Export publishes.

Weights: ``python -m anime_tools.downloads sam3 soft_prompt`` (or the ``masking``
pack).
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

    from anime_tools.masking import MergeMasksRequest, SamMaskRequest
    from anime_tools.masking.requests import MaskPrompt

    # --- SAM3 masks ----------------------------------------------------------
    # One list of regions: `keep` trains only inside them (default: the subject,
    # served by a learned soft prompt), `ignore` masks its region out. Both: keep
    # minus ignore. Balloons and lettering are ordinary ignore prompts here.
    sam = SamMaskRequest(
        image_dir=args.image_dir,
        recursive=True,
        masks=(
            MaskPrompt("keep", "text", "girl"),
            MaskPrompt("ignore", "text", "speech bubble"),
        ),
        dilate=5,
    )
    print("$ python -m anime_tools.masking.cli.generate_masks", *sam.to_argv())
    try:
        SamMaskRequest(image_dir=args.image_dir, masks=())
    except ValueError as e:
        print("  refused:", e)

    # --- merge ---------------------------------------------------------------
    # The positional input defaults to the generator's tree; a missing one is
    # skipped, and a second tree (hand-painted masks) is simply listed beside it.
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

    from anime_tools.masking import run_merge_masks, run_sam_masks

    run_sam_masks(sam)
    n = run_merge_masks(merge)
    print(f"\n{n} merged mask(s):")
    from PIL import Image

    for rel_dir, path in iter_masks(resolve_path(merge.output_dir)):
        mask = Image.open(path)  # mode L
        kept = sum(1 for v in mask.getdata() if v) / (mask.width * mask.height)
        print(f"  {rel_dir or '.'}/{path.name}: {mask.size}, {kept:.0%} kept")


if __name__ == "__main__":
    main()
