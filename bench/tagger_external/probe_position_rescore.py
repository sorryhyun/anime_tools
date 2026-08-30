#!/usr/bin/env python
"""Re-score the saved position-caption probe crops with an external tagger.

Takes the artifacts of ``bench/position_captions/probe_autocaption.py`` (SAM3
instance crops + hand-written clause GT in ``per_image.json``) and
``probe_binding.py`` (counterbalanced left/right renders) and asks an external
timm tagger the same two questions the pipeline asks the Anima Tagger per
crop:

  * hair-color winner — argmax over the ``hair_color`` group (names from the
    anima-tagger vocab) vs the GT clause's hair color;
  * character — does the crop keep the GT clause's character name (external:
    prob >= its per-tag ``best_threshold``)?

plus the binding probe's side test (p(want hair) > p(other hair) per half).
Ours is re-read from the saved artifacts so both arms score the same crops.

    make daemon-run ARGS="../anime_tools/bench/tagger_external/probe_position_rescore.py --label dbv4"

(run from the trainer checkout: the probe reads the trainer's
``bench/position_captions/results/`` artifacts, resolved against the curation
home = ``ANIMA_HOME`` / CWD.) ``load_external`` / ``collect_external`` were
inlined from the archived ``run_bench.py`` (curation split Phase 3b).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from anime_tools._env import resolve_path
from bench._common import make_run_dir, write_result

log = logging.getLogger("probe_position_rescore")


def _pad_square_white(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = max(w, h)
    if w == h:
        return im
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def load_external(args, device: torch.device):
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as st_load

    repo = args.external_repo
    weights = hf_hub_download(repo, "model.safetensors")
    tags_csv = hf_hub_download(repo, "selected_tags.csv")
    meta_p = hf_hub_download(repo, "meta.json")
    with open(meta_p) as f:
        meta = json.load(f)
    margs = meta.get("model_args", {})
    kwargs = {}
    if "act_layer" in margs:
        kwargs["act_layer"] = margs["act_layer"]
    with open(tags_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    model = timm.create_model(
        args.external_arch, pretrained=False, num_classes=len(rows), **kwargs
    )
    sd = st_load(weights)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"external state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    model.to(device).eval()
    pcfg = model.pretrained_cfg
    mean = torch.tensor(pcfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
    std = torch.tensor(pcfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
    return model, rows, mean, std


def collect_external(
    args, model, mean, std, image_paths: list[str], device: torch.device
) -> torch.Tensor:
    import numpy as np

    S = args.external_img_size
    probs: list[torch.Tensor] = []
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    for i in range(0, len(image_paths), args.external_batch_size):
        chunk = image_paths[i : i + args.external_batch_size]
        arrs = []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            im = _pad_square_white(im).resize((S, S), Image.BICUBIC)
            arrs.append(
                torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(
                    2, 0, 1
                )
            )
        x = (torch.stack(arrs) - mean) / std
        with torch.no_grad(), autocast:
            out = model(x.to(device, non_blocking=True))
        probs.append(out.float().sigmoid().cpu())
        if (i // args.external_batch_size) % 20 == 0:
            log.info("external: %d/%d", i + len(chunk), len(image_paths))
    return torch.cat(probs)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--external_repo", default="animetimm/convnextv2_huge.dbv4-full")
    p.add_argument("--external_arch", default="convnextv2_huge")
    p.add_argument("--external_img_size", type=int, default=512)
    p.add_argument("--external_batch_size", type=int, default=8)
    p.add_argument(
        "--model_dir",
        default=str(
            REPO_ROOT / "_archive/anima_tagger_training/checkpoints/anima-tagger-v5"
        ),
        help="tagger vocab.json dir (the archived checkpoint moved with the package)",
    )
    p.add_argument(
        "--autocaption_run",
        default="bench/position_captions/results/20260817-1122-autocaption",
    )
    p.add_argument(
        "--binding_run", default="bench/position_captions/results/20260817-1123-binding"
    )
    p.add_argument("--device", default=None)
    p.add_argument("--label", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    vocab = json.loads(
        (resolve_path(args.model_dir) / "vocab.json").read_text(encoding="utf-8")
    )
    hair_group = next(g for g in vocab["groups"] if g["name"] == "hair_color")
    hair_names = set(hair_group["tag_names"])
    char_names = {t["name"] for t in vocab["tags"] if t["category"] == "character"}

    model, rows, mean, std = load_external(args, device)
    col = {r["name"].replace("_", " "): j for j, r in enumerate(rows)}
    thr = {r["name"].replace("_", " "): float(r["best_threshold"]) for r in rows}
    hair_cols = {n: col[n] for n in hair_names if n in col}

    # ---- autocaption crops ----
    auto_dir = resolve_path(args.autocaption_run)
    per_image = json.loads(
        Path(auto_dir / "per_image.json").read_text(encoding="utf-8")
    )
    crop_paths, jobs = [], []
    for img in per_image:
        gt_by_pos = {c["pos"]: c["tags"] for c in img.get("gt_clauses", [])}
        for inst in img["instances"]:
            gt = gt_by_pos.get(inst["pos"])
            if not gt or "crop" not in inst:
                continue
            gt_hair = [t for t in gt if t in hair_names]
            gt_chars = [t for t in gt if t in char_names]
            if not gt_hair and not gt_chars:
                continue
            crop_paths.append(str(auto_dir / inst["crop"]))
            jobs.append((img["image"], inst, gt_hair, gt_chars))
    probs = collect_external(args, model, mean, std, crop_paths, device)

    hair_tot = hair_hit_ours = hair_hit_ext = 0
    char_tot = char_hit_ours = char_hit_ext = 0
    detail = []
    for (image, inst, gt_hair, gt_chars), pr in zip(jobs, probs):
        ext_hair = max(hair_cols, key=lambda n: float(pr[hair_cols[n]]))
        rec = {
            "image": image,
            "pos": inst["pos"],
            "gt_hair": gt_hair,
            "gt_chars": gt_chars,
            "ours_hair": inst.get("hair_color"),
            "ext_hair": ext_hair,
            "ext_hair_p": round(float(pr[hair_cols[ext_hair]]), 3),
        }
        if gt_hair:
            hair_tot += 1
            hair_hit_ours += inst.get("hair_color") in gt_hair
            hair_hit_ext += ext_hair in gt_hair
        for c in gt_chars:
            char_tot += 1
            ours_kept = c in inst.get("kept", {})
            ext_p = float(pr[col[c]]) if c in col else float("nan")
            ext_kept = c in col and ext_p >= thr[c]
            char_hit_ours += ours_kept
            char_hit_ext += ext_kept
            rec.setdefault("chars", []).append(
                {
                    "gt": c,
                    "ours_kept": ours_kept,
                    "ext_p": round(ext_p, 3),
                    "ext_kept": ext_kept,
                }
            )
        detail.append(rec)

    # ---- binding renders (left/right halves) ----
    bind_dir = resolve_path(args.binding_run)
    brows = json.loads(Path(bind_dir / "per_image.json").read_text(encoding="utf-8"))
    halves, bjobs = [], []
    for r in brows:
        img = Image.open(bind_dir / "renders" / f"{r['case']}.png").convert("RGB")
        w, h = img.size
        for side, box in (("left", (0, 0, w // 2, h)), ("right", (w // 2, 0, w, h))):
            want = r[side]
            other = r["right" if side == "left" else "left"]
            halves.append(img.crop(box))
            bjobs.append((r["case"], side, want, other, r[f"{side}_correct"]))
    tmp = []
    for i, hp in enumerate(halves):
        p = bind_dir / f"_half_{i}.png"
        hp.save(p)
        tmp.append(str(p))
    bprobs = collect_external(args, model, mean, std, tmp, device)
    for p in tmp:
        Path(p).unlink()
    side_ours = side_ext = 0
    for (case, side, want, other, ours_ok), pr in zip(bjobs, bprobs):
        side_ours += bool(ours_ok)
        side_ext += float(pr[col[f"{want} hair"]]) > float(pr[col[f"{other} hair"]])

    metrics = {
        "external_repo": args.external_repo,
        "hair_position": {
            "n": hair_tot,
            "ours": hair_hit_ours / max(hair_tot, 1),
            "external": hair_hit_ext / max(hair_tot, 1),
        },
        "char_position": {
            "n": char_tot,
            "ours": char_hit_ours / max(char_tot, 1),
            "external": char_hit_ext / max(char_tot, 1),
        },
        "binding_side": {
            "n": len(bjobs),
            "ours": side_ours / len(bjobs),
            "external": side_ext / len(bjobs),
        },
    }
    print(json.dumps(metrics, indent=1))
    for d in detail:
        print(d)
    run_dir = make_run_dir("tagger_external", (args.label or "") + "-position")
    (run_dir / "detail.json").write_text(
        json.dumps(detail, indent=1, ensure_ascii=False)
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["detail.json"],
        device=device,
    )


if __name__ == "__main__":
    main()
