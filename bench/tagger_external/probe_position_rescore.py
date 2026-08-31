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
home = ``ANIMA_HOME`` / CWD.) The external tagger is loaded through the
package's own :class:`~anime_tools.tagger.dbv4_backend.Dbv4Backend` — the two
are the same timm-over-``animetimm/*.dbv4-full`` load, and the copy this file
inlined from the archived ``run_bench.py`` (curation split Phase 3b) had drifted
into its own square-pad, its own normalisation and its own state-dict check.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from anime_tools._device import resolve_device
from anime_tools._env import resolve_path
from anime_tools.captions.vocab_io import names_by_category, resolved_groups
from anime_tools.tagger.data import TaggerCheckpoint
from anime_tools.tagger.dbv4_backend import Dbv4Backend
from bench._common import make_run_dir, write_result

log = logging.getLogger("probe_position_rescore")


def load_external(args, device: torch.device) -> Dbv4Backend:
    """The external tagger, through the package's own dbv4 loader.

    bf16 only on CUDA — the probe scores a few hundred crops, and bf16 on CPU
    buys nothing but noise in the probabilities it is comparing.
    """
    return Dbv4Backend(
        repo=args.external_repo,
        arch=args.external_arch,
        img_size=args.external_img_size,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )


def collect_external(
    backend: Dbv4Backend, images: list, batch_size: int
) -> torch.Tensor:
    """``[N, n_classes]`` sigmoid probs; ``images`` are paths or PIL images.

    Paths are opened a batch at a time, so a long crop list never sits decoded
    in memory at once.
    """
    probs: list[torch.Tensor] = []
    for i in range(0, len(images), batch_size):
        chunk = [
            im if isinstance(im, Image.Image) else Image.open(im).convert("RGB")
            for im in images[i : i + batch_size]
        ]
        probs.append(backend.forward(chunk).probs)
        if (i // batch_size) % 20 == 0:
            log.info("external: %d/%d", i + len(chunk), len(images))
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
    device = torch.device(resolve_device(args.device))
    ckpt = TaggerCheckpoint.from_dir(resolve_path(args.model_dir))
    hair_group = next(g for g in resolved_groups(ckpt.vocab) if g.name == "hair_color")
    hair_names = set(hair_group.tag_names)
    char_names = names_by_category(ckpt.vocab, ("character",))["character"]

    backend = load_external(args, device)
    card = backend.card
    col = card.name_to_col
    thr = {n: float(card.rows[j]["best_threshold"]) for n, j in col.items()}
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
    probs = collect_external(backend, crop_paths, args.external_batch_size)

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
    # The halves are already in memory — no PNG round-trip through the run dir.
    bprobs = collect_external(backend, halves, args.external_batch_size)
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
