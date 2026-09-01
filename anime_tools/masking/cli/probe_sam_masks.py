"""Dump SAM3's raw per-instance masks, colour-coded, over one or many images.

Bypasses the audit to answer "is the mask broken, or are we mishandling it?": prints each
mask's shape / dtype / value range / fill and renders every instance in its own colour.
Several prompts sweep several images in one pass, the encoding computed once per image and
re-grounded per prompt. Renders land at ``<out>/<stem>/prompt_<label>.png``, the numbers
in a ``probe.json``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from anime_tools import workspace as WS
from anime_tools._device import add_device_arg, resolve_device
from anime_tools._env import resolve_path
from anime_tools._json import write_json
from anime_tools._walk import walk_images

# Importing _sam3 also installs the `np.bool` alias sam3 needs before it loads.
from anime_tools.masking._sam3 import add_checkpoint_arg, load_sam3
from anime_tools.stages.instance_detection import Detection, mask_box_fill

COLORS = [
    (255, 60, 60),
    (60, 140, 255),
    (255, 210, 60),
    (170, 100, 255),
    (60, 230, 190),
    (255, 120, 200),
    (150, 255, 90),
    (255, 165, 80),
]


def parse_prompts(spec: str) -> list[tuple[str, str]]:
    """``"girl,rear=buttocks"`` -> ``[("girl", "girl"), ("rear", "buttocks")]``.

    The label names the output file; without an explicit ``label=`` the prompt is
    slugified into one.
    """
    out: list[tuple[str, str]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        label, _, prompt = item.partition("=")
        if not prompt:
            label, prompt = re.sub(r"\W+", "_", label).strip("_"), label
        out.append((label, prompt))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "images",
        nargs="*",
        help="Image paths (repo-relative ok). Omit and pass --path_pattern to "
        "sweep the resized tree instead",
    )
    p.add_argument("--dst", default=WS.RESIZED)
    p.add_argument(
        "--path_pattern",
        dest="path_pattern",
        default=None,
        help="Glob over --dst, used when no image paths are given",
    )
    p.add_argument(
        "--prompts",
        default="girl",
        help="Comma-separated prompts, each optionally ``label=prompt``. The "
        'body-part fallback ships as "buttocks,hips,thighs"',
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Processor confidence floor — low on purpose, we want to see every "
        "proposal including the ones the audit filters out",
    )
    add_checkpoint_arg(p)
    add_device_arg(p)
    p.add_argument("--out", default=f"{WS.REPORTS}/mask_probe")
    p.add_argument(
        "--summary",
        default=None,
        help="Where the JSON lands (default <out>/probe.json)",
    )
    p.add_argument(
        "--dump_masks",
        action="store_true",
        help="Also write every binary mask as its own PNG (first prompt only)",
    )
    args = p.parse_args()
    args.device = resolve_device(args.device)
    return args


def resolve_images(args: argparse.Namespace) -> list[Path]:
    if args.images:
        return [resolve_path(i) for i in args.images]
    if not args.path_pattern:
        raise SystemExit("pass image paths or --path_pattern")
    dst = resolve_path(args.dst)
    return sorted(walk_images(dst, recursive=True, pattern=args.path_pattern))


def out_names(paths: list[Path]) -> list[str]:
    """One output dir per image, keyed on the stem.

    A bare stem is only unique within one folder, so the whole selection falls back to
    `<parent>__<stem>` the moment any two collide — all-or-nothing, so the naming does
    not depend on which images happened to be selected.
    """
    stems = [p.stem for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    return [f"{p.parent.name}__{p.stem}" for p in paths]


def probe_one(
    processor,
    state,
    image: Image.Image,
    prompt: str,
    *,
    verbose: bool,
    dump_dir: Path | None = None,
) -> tuple[list[dict], np.ndarray, list]:
    """Ground one prompt against an already-encoded image.

    Returns the per-proposal records, the colour overlay, and the raw boxes (the caller
    draws them, so the overlay stays a pure mask render).
    """
    out = processor.set_text_prompt(prompt=prompt, state=state)
    boxes, scores, masks = out["boxes"], out["scores"], out.get("masks")
    if verbose:
        print(f"    returned keys: {sorted(out.keys())}")
    print(f"    instances: {len(boxes)}   masks present: {masks is not None}")

    overlay = np.asarray(image).astype(np.float32)
    records: list[dict] = []
    canvas = image.width * image.height

    for i, (box, score) in enumerate(zip(boxes, scores)):
        import torch

        coords = [float(v) for v in (box.tolist() if torch.is_tensor(box) else box)]
        rec = {
            "score": round(float(score), 4),
            "box": [round(v, 1) for v in coords],
            "area_frac": round(
                max(0.0, coords[2] - coords[0])
                * max(0.0, coords[3] - coords[1])
                / canvas,
                4,
            ),
            "box_fill": None,
            "aligned": None,
        }
        if masks is None or i >= len(masks):
            print(f"    #{i} score={float(score):.4f} box={rec['box']}  mask=MISSING")
            records.append(rec)
            continue

        m = masks[i]
        raw = m.float().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
        flat = raw[0] if raw.ndim == 3 else raw
        aligned = flat.shape == (image.height, image.width)
        binary = flat > 0.5
        # Same fill the pipeline's NMS tie-break sees, i.e. the number
        # `dedupe_detections` would have ranked on. It clamps the box to the *mask's*
        # shape, which is the point when `aligned` is False.
        fill = mask_box_fill(
            Detection(box=tuple(coords), score=float(score), mask=flat)
        )
        rec["box_fill"] = round(fill or 0.0, 4)
        rec["aligned"] = bool(aligned)
        records.append(rec)

        if verbose:
            print(f"    mask shape={raw.shape} dtype={raw.dtype}")
            print(
                f"    values min={raw.min():.4f} max={raw.max():.4f} "
                f"mean={raw.mean():.4f}"
            )
            print(
                f"    2D shape={flat.shape} vs image (H,W)="
                f"({image.height},{image.width})  "
                f"{'ALIGNED' if aligned else '*** MISMATCH ***'}"
            )
        print(
            f"    #{i} score={float(score):.4f} box={rec['box']} "
            f"area={rec['area_frac']:.3f} fill={rec['box_fill']:.3f}"
            f"{'' if aligned else '  *** MASK MISALIGNED ***'}"
        )
        if dump_dir is not None:
            Image.fromarray((binary * 255).astype(np.uint8)).save(
                dump_dir / f"mask_{i}_raw.png"
            )
        if aligned:
            overlay[binary] = (
                overlay[binary] * 0.45
                + np.array(COLORS[i % len(COLORS)], np.float32) * 0.55
            )

    return records, overlay, boxes


def render(overlay: np.ndarray, boxes, path: Path) -> None:
    import torch

    composite = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(composite)
    for i, box in enumerate(boxes):
        color = COLORS[i % len(COLORS)]
        coords = [float(v) for v in (box.tolist() if torch.is_tensor(box) else box)]
        draw.rectangle(coords, outline=color, width=6)
        draw.rectangle(
            [coords[0], coords[1], coords[0] + 44, coords[1] + 44], fill=color
        )
        draw.text((coords[0] + 14, coords[1] + 12), str(i), fill=(0, 0, 0))
    composite.save(path)


def main() -> None:
    args = parse_args()
    import torch

    prompts = parse_prompts(args.prompts)
    if not prompts:
        raise SystemExit("--prompts is empty")
    paths = resolve_images(args)
    out_root = resolve_path(args.out)
    # A lone image with a lone prompt is chatty (dtype / value range / alignment); a
    # sweep stays terse.
    verbose = len(paths) == 1 and len(prompts) == 1
    print(
        f"{len(paths)} image(s) x {len(prompts)} prompt(s): {[p for _, p in prompts]}"
    )

    _model, processor = load_sam3(
        resolve_path(args.checkpoint),
        args.device,
        confidence_threshold=args.threshold,
    )

    summary: list[dict] = []
    for path, name in zip(paths, out_names(paths)):
        image = Image.open(path).convert("RGB")
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {path}  size={image.size} (W,H)")

        entry: dict = {"image": str(path), "size": list(image.size), "prompts": {}}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            for idx, (label, prompt) in enumerate(prompts):
                print(f"  -- {label}: {prompt!r}")
                records, overlay, boxes = probe_one(
                    processor,
                    state,
                    image,
                    prompt,
                    verbose=verbose,
                    dump_dir=out_dir if (args.dump_masks and idx == 0) else None,
                )
                entry["prompts"][label] = {
                    "prompt": prompt,
                    "n": len(records),
                    "proposals": records,
                }
                render(overlay, boxes, out_dir / f"prompt_{label}.png")
                if idx == 0:
                    # The first prompt's render, also under the name older references use.
                    render(overlay, boxes, out_dir / "overlay.png")
        summary.append(entry)
        print(f"  wrote: {out_dir}")

    summary_path = (
        resolve_path(args.summary) if args.summary else out_root / "probe.json"
    )
    write_json(summary_path, {"prompts": dict(prompts), "images": summary})
    print(f"\nwrote: {summary_path}")


if __name__ == "__main__":
    main()
