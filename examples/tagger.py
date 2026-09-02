"""The Anima Tagger on one image: raw scores, kept tags, and the caption form.

    python examples/tagger.py path/to/image.png [--tagger_dir models/captioners/anima-tagger-dbv4]

CLI equivalents (single image, stdout only)::

    python -m anime_tools.tagger.cli.autotag --image img.png       # ANIMA_AUTOTAG_RESULT<TAB>caption
    python -m anime_tools.tagger.cli --mode predict --image img.png --show_scores

Batch tagging over a dataset is the **autotag stage** (``examples/stages.py``),
not this module. The checkpoint (vocab, thresholds, rules, groups, sidecar head)
is fetched from the Hub on first use; the GPL dbv4 backbone underneath it is
gated upstream and needs a Hugging Face token (``hf auth login``).
"""

from __future__ import annotations

import argparse

from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("image")
    p.add_argument(
        "--tagger_dir", default=None, help="checkpoint dir (default: models/…)"
    )
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    p.add_argument("--min_confidence", type=float, default=0.0)
    args = p.parse_args()

    from anime_tools._env import resolve_path
    from anime_tools.tagger.tagger import (
        DEFAULT_TAGGER_DIR,
        AnimaTagger,
        ensure_tagger_checkpoint,
    )

    # ensure_tagger_checkpoint fetches any missing checkpoint file (and the
    # backbone) into the dir; AnimaTagger then loads it. ``stages/_models.py``
    # caches one per (dir, device) when several stages share a process.
    ckpt_dir = ensure_tagger_checkpoint(
        resolve_path(args.tagger_dir or DEFAULT_TAGGER_DIR)
    )
    tagger = AnimaTagger(ckpt_dir, device=args.device)
    image = Image.open(args.image)

    # predict(): everything the heads say. ``kept`` is the thresholded set (per-tag
    # F1 thresholds, softmax argmax inside a typed group such as hair_color);
    # ``scores`` is every in-vocab probability; ``groups`` the per-group winner.
    out = tagger.predict(image)
    print(
        "rating:",
        out["rating"],
        {k: round(v, 3) for k, v in out["rating_scores"].items()},
    )
    if "people_count" in out:
        print("people:", out["people_count"])
    kept = sorted(out["kept"].items(), key=lambda kv: -kv[1])
    print(f"kept {len(kept)} tags:")
    for name, prob in kept[:25]:
        print(
            f"  {prob:.3f}  {name}   (threshold {out['thresholds'].get(name, 0.5):.2f})"
        )
    if out.get("groups"):
        print("groups:", {g: t for g, t in out["groups"].items() if t})

    # predict_caption(): the Anima caption string — rating first, then the
    # slotted tags (count, characters, copyrights, @artists, generals) with the
    # checkpoint's tag rules applied. This is what the autotag stage writes.
    print(
        "\ncaption:", tagger.predict_caption(image, min_confidence=args.min_confidence)
    )


if __name__ == "__main__":
    main()
