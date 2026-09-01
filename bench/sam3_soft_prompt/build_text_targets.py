"""Part B "free text / SFX" row — MIT-labelled targets for a SAM3 text prompt.

Unlike the girl row, this concept has an *independent* detector: MIT
(UNet++ stroke segmentation gated by comictextdetector's text-block head —
exactly what `make mask` runs today via `generate_masks_mit.py`). Its output
is the label, so the question "can a SAM3 soft prompt replace MIT" has a real
held-out answer instead of a self-labelled one.

Per image:

- **positives** — every CTD text block that contains gated stroke pixels
  becomes one instance: box = tight bbox of the strokes inside the block
  (normalized xyxy), mask = strokes inside the block morphologically closed
  into a glyph blob (so the 288² mask target is a region, not 1-px strokes),
  and the full-res gated stroke mask is kept (bit-packed) for the pixel-level
  eval.
- **negatives** — images where neither the UNet++ nor the CTD head fires at
  all (train rows with zero targets teach the presence head to say "no").
- **ambiguous** (CTD fires but no strokes, or strokes with no block) — skipped.

Split is a stable hash on the relative path: ``--holdout_mod`` of every N
images go to the holdout, which `eval_text_prompt.py` scores. Output manifest
is the same shape `train_soft_prompt.py --targets` reads.

    make daemon-run ARGS="bench/sam3_soft_prompt/build_text_targets.py"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from anime_tools._env import resolve_path as resolve_under_home
from anime_tools._walk import walk_images
from anime_tools.downloads import default_ctd_onnx_path
from bench._common import start_heartbeat
from bench.sam3_soft_prompt.common import MASK_RES


def _mit_module():
    return importlib.import_module("anime_tools.masking.cli.generate_masks_mit")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dst", default="post_image_dataset/resized")
    p.add_argument("--path_pattern", default="*")
    p.add_argument(
        "--out", default="post_image_dataset/captions/sam3_soft_prompt/text_targets"
    )
    p.add_argument("--mit_model", default="models/mit/model.pth")
    # The stage has no such flag any more (the catalog owns the path); the bench
    # keeps one so an ablation can point at a different net, defaulted from the
    # same row so the two cannot drift.
    p.add_argument("--ctd_onnx", default=str(default_ctd_onnx_path()))
    p.add_argument("--text_threshold", type=float, default=0.8)
    p.add_argument(
        "--close_frac",
        type=float,
        default=0.06,
        help="closing kernel as a fraction of the block's shorter side",
    )
    p.add_argument(
        "--min_stroke_px", type=int, default=30, help="drop blocks with fewer strokes"
    )
    p.add_argument("--holdout_mod", type=int, default=8, help="1/N images held out")
    p.add_argument(
        "--neg_ratio",
        type=float,
        default=1.0,
        help="negatives per positive kept in the train split (0 = none)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def is_holdout(rel: str, mod: int) -> bool:
    h = int(hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8], 16)
    return h % mod == 0


def instances(
    strokes: np.ndarray, boxes: list[tuple[int, int, int, int]], a
) -> tuple[list[list[float]], list[np.ndarray], np.ndarray]:
    """Return (norm xyxy boxes, 288² bool masks, gated full-res stroke mask)."""
    h, w = strokes.shape
    gated = np.zeros_like(strokes)
    out_boxes: list[list[float]] = []
    out_masks: list[np.ndarray] = []
    for x0, y0, x1, y1 in boxes:
        window = strokes[y0:y1, x0:x1]
        if window.sum() < a.min_stroke_px:
            continue
        ys, xs = np.nonzero(window)
        bx0, by0 = x0 + int(xs.min()), y0 + int(ys.min())
        bx1, by1 = x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1
        k = max(3, int(a.close_frac * min(bx1 - bx0, by1 - by0)) | 1)
        blob = np.zeros((h, w), np.uint8)
        blob[y0:y1, x0:x1] = window
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        gated[y0:y1, x0:x1] |= window
        out_boxes.append([bx0 / w, by0 / h, bx1 / w, by1 / h])
        out_masks.append(
            cv2.resize(blob, (MASK_RES, MASK_RES), interpolation=cv2.INTER_AREA) > 0
        )
    return out_boxes, out_masks, gated


def main() -> None:
    a = parse_args()
    start_heartbeat(label="build_text_targets")
    dst = resolve_under_home(a.dst)
    out = resolve_under_home(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images = walk_images(dst, recursive=True, pattern=a.path_pattern)
    if a.limit:
        images = images[: a.limit]
    print(f"{len(images)} images", flush=True)

    mit = _mit_module()
    model = mit._load_model(str(resolve_under_home(a.mit_model)), device=a.device)
    ctd = mit._load_ctd(str(resolve_under_home(a.ctd_onnx)), device=a.device)

    rows: list[dict] = []
    pos = neg = amb = 0
    for n, path in enumerate(images, 1):
        if n % 100 == 0:
            print(f"  [{n}/{len(images)}] pos={pos} neg={neg} amb={amb}", flush=True)
        rel = str(path.relative_to(dst))
        with Image.open(path) as h:
            img = np.array(h.convert("RGB"))
        prob = mit._detect_mask(model, img, device=a.device, text_threshold=None)
        strokes = (prob > int(a.text_threshold * 255)).astype(np.uint8)
        boxes = mit._ctd_text_boxes(ctd, img)
        tboxes, tmasks, gated = instances(strokes, boxes, a)
        holdout = is_holdout(rel, a.holdout_mod)
        row = {"image": rel, "holdout": holdout, "n": len(tboxes)}
        if tboxes:
            kind = "pos"
            pos += 1
        elif not strokes.any() and not boxes:
            kind = "neg"
            neg += 1
        else:
            kind = "amb"
            amb += 1
            row["kind"] = kind
            rows.append(row)
            continue
        row["kind"] = kind
        tpath = out / (rel.replace("/", "__") + ".npz")
        np.savez_compressed(
            tpath,
            boxes=np.asarray(tboxes, dtype=np.float32).reshape(-1, 4),
            masks=np.stack(tmasks).astype(np.bool_)
            if tmasks
            else np.zeros((0, MASK_RES, MASK_RES), np.bool_),
            scores=np.ones(len(tboxes), np.float32),
            mit_shape=np.asarray(gated.shape, np.int32),
            mit_packed=np.packbits(gated.astype(np.bool_)),
        )
        row["target"] = str(tpath.relative_to(out))
        row["mit_px"] = int(gated.sum())
        rows.append(row)

    # train flags: all holdout=False positives, plus a capped negative sample.
    train_pos = [r for r in rows if r["kind"] == "pos" and not r["holdout"]]
    train_neg = [r for r in rows if r["kind"] == "neg" and not r["holdout"]]
    train_neg.sort(key=lambda r: hashlib.sha1(r["image"].encode()).hexdigest())
    train_neg = train_neg[: int(len(train_pos) * a.neg_ratio)]
    for r in train_pos + train_neg:
        r["train"] = True
    summary = {
        "images": len(rows),
        "pos": pos,
        "neg": neg,
        "amb": amb,
        "train": len(train_pos) + len(train_neg),
        "train_pos": len(train_pos),
        "train_neg": len(train_neg),
        "holdout_pos": sum(r["kind"] == "pos" and r["holdout"] for r in rows),
        "holdout_neg": sum(r["kind"] == "neg" and r["holdout"] for r in rows),
        "instances": sum(r["n"] for r in rows),
        "args": vars(a),
    }
    (out / "manifest.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
