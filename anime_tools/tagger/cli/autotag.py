"""Single-image autotagger entry for the GUI "Autotag" button.

Resolves (auto-downloading on first use) the Anima Tagger checkpoint, runs it
on one image, and prints the predicted caption on a single sentinel-prefixed
stdout line so the caller can parse it back out of the daemon's stdout log:

    ANIMA_AUTOTAG_RESULT\t<comma-separated caption>

Kept separate from ``anime_tools.tagger.cli.main`` so the GUI has a thin entry
whose only stdout contract is the sentinel line above.
"""

from __future__ import annotations

import argparse
import logging

from anime_tools._device import add_device_arg, resolve_device
from anime_tools._env import resolve_path, setup_logging
from anime_tools.tagger.tagger import (
    DEFAULT_TAGGER_DIR,
    AnimaTagger,
    ensure_tagger_checkpoint,
)

# Sentinel prefix the GUI greps for in the job's stdout. Tab-separated so the
# caption (which contains commas + spaces) survives intact on one line.
RESULT_PREFIX = "ANIMA_AUTOTAG_RESULT\t"

setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="Anima single-image autotagger")
    p.add_argument("--image", required=True, help="Path to the image to tag.")
    p.add_argument(
        "--tagger_dir",
        default=DEFAULT_TAGGER_DIR,
        help="Tagger checkpoint dir (repo-relative; auto-downloaded if missing).",
    )
    add_device_arg(p)
    p.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
        help="Extra probability floor (0-1) on top of the per-tag F1 thresholds.",
    )
    args = p.parse_args()

    from PIL import Image

    image_path = resolve_path(args.image)
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")

    ckpt_dir = ensure_tagger_checkpoint(resolve_path(args.tagger_dir))
    device = resolve_device(args.device)
    tagger = AnimaTagger(ckpt_dir, device=device)
    caption = tagger.predict_caption(
        Image.open(image_path), min_confidence=args.min_confidence
    )

    print(RESULT_PREFIX + caption, flush=True)


if __name__ == "__main__":
    main()
