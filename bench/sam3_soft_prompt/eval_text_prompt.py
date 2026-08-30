"""Score SAM3 prompts (text or learned) against MIT labels on the text holdout.

The "can SAM3 replace MIT" gate from `sam3_soft_prompt_expansion.md`:
**MIT recall ≥ 0.9 at ≤ 0.1 FP/img**. Both are box-level (IoU ≥ 0.5 against
the CTD-block targets from `build_text_targets.py`); FP counts survivors on
holdout images that match no target, negatives included. Because the actual
consumer is a training-ignore mask, the pixel view is reported too: recall
of MIT's gated stroke pixels under the union of survivor masks, and the
over-mask ratio (SAM pixels / MIT pixels, after both sides get the shipped
`dilate: 3`). Every prompt is scored at several floors so the readout is a
curve, not one operating point.

A spec ending in ``.safetensors`` is a soft prompt; anything else is text.
``--sheets N`` also writes the N worst-recall positives per prompt as
side-by-side panels (MIT strokes | SAM masks) under the run dir.

    make daemon-run ARGS="bench/sam3_soft_prompt/eval_text_prompt.py \\
        --prompts text 'sound effect text' 'handwritten text' --sheets 24"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from anime_tools._env import resolve_path as resolve_under_home
from bench._common import make_run_dir, start_heartbeat, write_result
from bench.sam3_soft_prompt.common import (
    encode_image,
    encode_text,
    ground,
    install_prompt,
    iou_xyxy,
    load_sam3,
    load_soft_prompt,
    nms,
    proposals,
)

FLOORS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--targets", default="post_image_dataset/captions/sam3_soft_prompt/text_targets"
    )
    p.add_argument("--dst", default="post_image_dataset/resized")
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--which", choices=("holdout", "train", "all"), default="holdout")
    p.add_argument("--iou_threshold", type=float, default=0.65, help="NMS")
    p.add_argument("--match_iou", type=float, default=0.5)
    p.add_argument("--dilate", type=int, default=3)
    p.add_argument("--sheets", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--label", default=None)
    return p.parse_args()


def prompt_spec(model, spec: str, device: str) -> dict:
    if spec.endswith(".safetensors"):
        return load_soft_prompt(resolve_under_home(spec), device)
    return encode_text(model, spec, device)


def load_target(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    h, w = (int(v) for v in z["mit_shape"])
    mit = np.unpackbits(z["mit_packed"])[: h * w].reshape(h, w).astype(np.uint8)
    return z["boxes"], mit


def union_mask(rows: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if not rows:
        return np.zeros((h, w), np.uint8)
    m = torch.stack([r["mask"] for r in rows]).any(0).float()[None, None]
    m = torch.nn.functional.interpolate(m, size=(h, w), mode="bilinear")
    return (m[0, 0] > 0.5).cpu().numpy().astype(np.uint8)


def score_image(
    props: list[dict], tboxes: np.ndarray, mit: np.ndarray, a
) -> dict[str, dict]:
    """Per floor: matched targets, survivors, unmatched survivors, pixel stats."""
    h, w = mit.shape
    k = np.ones((a.dilate, a.dilate), np.uint8) if a.dilate > 0 else None
    mit_d = cv2.dilate(mit, k) if k is not None and mit.any() else mit
    mit_px = int(mit_d.sum())
    out: dict[str, dict] = {}
    for floor in FLOORS:
        surv = nms(
            [r for r in props if r["score"] >= floor and r["area_frac"] >= 0.0005],
            a.iou_threshold,
        )
        matched = set()
        fp = 0
        for r in surv:
            best = -1
            for ti, tb in enumerate(tboxes):
                if iou_xyxy(r["box"], tb.tolist()) >= a.match_iou:
                    best = ti
                    matched.add(ti)
            if best < 0:
                fp += 1
        sam = union_mask(surv, (h, w))
        if k is not None and sam.any():
            sam = cv2.dilate(sam, k)
        out[str(floor)] = {
            "targets": len(tboxes),
            "hit": len(matched),
            "survivors": len(surv),
            "fp": fp,
            "mit_px": mit_px,
            "hit_px": int((sam & mit_d).sum()) if mit_px else 0,
            "sam_px": int(sam.sum()),
            "canvas": h * w,
        }
    return out


def sheet(image: Image.Image, mit: np.ndarray, sam: np.ndarray, title: str):
    img = np.array(image)
    left, right = img.copy(), img.copy()
    left[mit > 0] = (left[mit > 0] * 0.3 + np.array([255, 60, 60]) * 0.7).astype(
        np.uint8
    )
    right[sam > 0] = (right[sam > 0] * 0.3 + np.array([60, 120, 255]) * 0.7).astype(
        np.uint8
    )
    panel = np.concatenate([left, right], axis=1)
    scale = 1280 / panel.shape[1]
    panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return Image.fromarray(panel)


def main() -> None:
    a = parse_args()
    start_heartbeat(label="eval_text_prompt")
    dst = resolve_under_home(a.dst)
    tdir = resolve_under_home(a.targets)
    manifest = json.loads((tdir / "manifest.json").read_text(encoding="utf-8"))
    rows = [r for r in manifest["rows"] if r["kind"] in ("pos", "neg")]
    if a.which == "holdout":
        rows = [r for r in rows if r["holdout"]]
    elif a.which == "train":
        rows = [r for r in rows if r.get("train")]
    if a.limit:
        rows = rows[: a.limit]
    print(
        f"{len(rows)} images ({sum(r['kind'] == 'pos' for r in rows)} pos)", flush=True
    )

    run_dir = make_run_dir("sam3_soft_prompt", a.label or "eval-text")
    model, processor = load_sam3(resolve_under_home(a.checkpoint), a.device)
    prompts = {s: prompt_spec(model, s, a.device) for s in a.prompts}
    per_prompt: dict[str, list[dict]] = {s: [] for s in a.prompts}
    worst: dict[str, list] = {s: [] for s in a.prompts}

    for n, row in enumerate(rows, 1):
        if n % 50 == 0:
            print(f"  [{n}/{len(rows)}]", flush=True)
        with Image.open(dst / row["image"]) as h:
            image = h.convert("RGB")
        tboxes, mit = load_target(tdir / row["target"])
        bo = encode_image(model, processor, image)
        for spec, prompt in prompts.items():
            install_prompt(bo, prompt)
            props = proposals(ground(model, processor, bo), min(FLOORS))
            s = score_image(props, tboxes, mit, a)
            per_prompt[spec].append({"image": row["image"], "kind": row["kind"], **s})
            if a.sheets and row["kind"] == "pos":
                s5 = s["0.5"]
                rec = s5["hit_px"] / max(1, s5["mit_px"])
                surv = nms(
                    [
                        r
                        for r in props
                        if r["score"] >= 0.5 and r["area_frac"] >= 0.0005
                    ],
                    a.iou_threshold,
                )
                worst[spec].append(
                    (rec, row["image"], image, mit, union_mask(surv, mit.shape))
                )

    def agg(items: list[dict]) -> dict[str, dict]:
        out = {}
        for floor in FLOORS:
            f = str(floor)
            tg = sum(i[f]["targets"] for i in items)
            hit = sum(i[f]["hit"] for i in items)
            fp = sum(i[f]["fp"] for i in items)
            mit_px = sum(i[f]["mit_px"] for i in items)
            hit_px = sum(i[f]["hit_px"] for i in items)
            sam_px = sum(i[f]["sam_px"] for i in items)
            neg = [i for i in items if i["kind"] == "neg"]
            out[f] = {
                "box_recall": round(hit / max(1, tg), 4),
                "fp_per_img": round(fp / max(1, len(items)), 4),
                "neg_img_fp_rate": round(
                    sum(i[f]["survivors"] > 0 for i in neg) / max(1, len(neg)), 4
                ),
                "px_recall": round(hit_px / max(1, mit_px), 4),
                "overmask_ratio": round(sam_px / max(1, mit_px), 3),
                "sam_canvas_frac": round(
                    sam_px / max(1, sum(i[f]["canvas"] for i in items)), 4
                ),
                "gate": hit / max(1, tg) >= 0.9 and fp / max(1, len(items)) <= 0.1,
            }
        return out

    metrics = {spec: agg(items) for spec, items in per_prompt.items()}
    for spec, m in metrics.items():
        print(f"\n== {spec}")
        for f, v in m.items():
            print(f"  floor {f}: {json.dumps(v)}")

    if a.sheets:
        for spec, lst in worst.items():
            sdir = run_dir / "sheets" / spec.replace("/", "_").replace(" ", "_")
            sdir.mkdir(parents=True, exist_ok=True)
            for rec, rel, image, mit, sam in sorted(lst, key=lambda t: t[0])[
                : a.sheets
            ]:
                name = rel.replace("/", "__") + ".jpg"
                sheet(image, mit, sam, f"{spec} px_recall={rec:.2f} {rel}").save(
                    sdir / name, quality=85
                )

    (run_dir / "per_image.json").write_text(
        json.dumps(per_prompt, indent=1), encoding="utf-8"
    )
    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics={"images": len(rows), "prompts": metrics},
        label=a.label,
        artifacts=[run_dir / "per_image.json"],
    )
    print(f"wrote: {run_dir}")


if __name__ == "__main__":
    main()
