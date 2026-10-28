#!/usr/bin/env python
"""PixAI Tagger v1.0 vs the live anima-tagger-dbv4 on OUR held-out split.

Ground truth = the checkpoint's ``dataset.json`` val split (the caption sidecars
the manifest was built from). Both taggers are scored on the tags BOTH can emit:
ours' supported mask (dbv4-matched + sidecar rows) ∩ PixAI's 30,877-tag vocab,
joined by name (``_`` → space, ``rules.yaml`` renames recovered).

Arms:
  * ``ours@product`` — ``AnimaTagger`` kept set (per-tag thresholds, group
    argmax, character floor). Sidecar thresholds were F1-calibrated ON this val
    split, so its thresholded F1 is optimistic for us.
  * ``pixai@card`` — PixAI's per-category card thresholds.
  * ``*@0.4`` — flat 0.4 for both.
  Read mAP first (threshold-free).

``@artist`` rows (ours never fires) are scored for PixAI alone, its ``style``
category joined on the bare name.

READOUT: bench/tagger_external/results/<ts>[-label]/{summary.md, result.json, per_tag.csv}

Run through the daemon from anima_lora/::

    make daemon-run ARGS="../anime_tools/bench/tagger_external/pixai_vs_dbv4.py --label pixai-v1"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from anime_tools.tagger.dbv4_backend import align_vocab, rename_recovery_from_rules
from anime_tools.tagger.tagger import AnimaTagger
from bench._common import make_run_dir, start_heartbeat, write_result

log = logging.getLogger("pixai_vs_dbv4")

PIXAI_REPO = "pixai-labs/pixai-tagger-v1.0"
PIXAI_RATING = {
    "safe": "rating:g",
    "sensitive": "rating:s",
    "nsfw": "rating:q",
    "explicit": "rating:e",
}
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model_dir", default="models/captioners/anima-tagger-dbv4")
    p.add_argument(
        "--pixai_ckpt",
        default=None,
        help="also score a pixai-backed AnimaTagger checkpoint as shipped "
        "(thresholds / groups / sidecar) — arm newckpt@product",
    )
    p.add_argument("--pixai_repo", default=PIXAI_REPO)
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--flat_thr", type=float, default=0.4)
    p.add_argument("--freq_head_min", type=int, default=1000)
    p.add_argument("--freq_mid_min", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--label", default=None)
    return p.parse_args()


def load_images(paths: list[str]) -> list[Image.Image]:
    def _one(p: str) -> Image.Image:
        im = Image.open(p)
        im.load()
        return im

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(_one, paths))


def batches(xs: list, n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


@torch.no_grad()
def run_ours(tagger: AnimaTagger, paths: list[str], bs: int):
    probs, kept, rating = [], [], []
    t_model = 0.0
    for chunk in batches(paths, bs):
        ims = [im.convert("RGB") for im in load_images(chunk)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, rate, people = tagger._heads_forward_batch(ims)
        torch.cuda.synchronize()
        t_model += time.perf_counter() - t0
        probs.append(logits.float().sigmoid().cpu())
        rating.append(rate.float().cpu())
        for i in range(len(ims)):
            out = tagger._predict_row(
                logits[i], rate[i], None if people is None else people[i]
            )
            kept.append(set(out["kept"]))
        print(f"ours {sum(len(p) for p in probs)}/{len(paths)}", flush=True)
    return torch.cat(probs), kept, torch.cat(rating), t_model


@torch.no_grad()
def run_pixai(paths: list[str], bs: int, repo: str):
    from transformers import AutoImageProcessor, AutoModel

    model = (
        AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    proc = AutoImageProcessor.from_pretrained(repo, trust_remote_code=True)
    probs = []
    t_model = 0.0
    for chunk in batches(paths, bs):
        x = proc(load_images(chunk))["pixel_values"].to("cuda", torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model(x).float().sigmoid()
        torch.cuda.synchronize()
        t_model += time.perf_counter() - t0
        probs.append(y.cpu())
        print(f"pixai {sum(len(p) for p in probs)}/{len(paths)}", flush=True)
    cfg = model.config
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    return torch.cat(probs), cfg, t_model, peak


def pixai_card_thresholds(cfg) -> tuple[torch.Tensor, list[str]]:
    thr, cats = [], []
    for cat, n in cfg.tags_split:
        thr += [cfg.category_best_threshold[cat]] * n
        cats += [cat] * n
    return torch.tensor(thr), cats


def average_precision(scores: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-column AP for ``[N, T]`` scores / {0,1} gt; NaN where a column has no positives."""
    order = scores.argsort(dim=0, descending=True)
    g = gt.gather(0, order)
    tp = g.cumsum(0)
    prec = tp / torch.arange(1, g.shape[0] + 1, dtype=torch.float32).unsqueeze(1)
    npos = g.sum(0)
    ap = (prec * g).sum(0) / npos.clamp(min=1)
    ap[npos == 0] = float("nan")
    return ap


def prf(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    tp = (pred * gt).sum(0)
    fp = (pred * (1 - gt)).sum(0)
    fn = ((1 - pred) * gt).sum(0)
    micro = float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp(min=1))
    has = gt.sum(0) > 0
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    p = float(tp.sum() / (tp.sum() + fp.sum()).clamp(min=1))
    r = float(tp.sum() / (tp.sum() + fn.sum()).clamp(min=1))
    return {
        "micro_f1": micro,
        "macro_f1": float(f1[has].mean()),
        "precision": p,
        "recall": r,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    start_heartbeat()
    args = parse_args()
    model_dir = Path(args.model_dir)
    dataset = json.loads((model_dir / "dataset.json").read_text())
    stem_set = set(dataset["split"][args.split])
    rows = [
        (p, ti, ri)
        for s, p, ti, ri in zip(
            dataset["stems"],
            dataset["image_paths"],
            dataset["tag_indices"],
            dataset["rating_indices"],
        )
        if s in stem_set
    ]
    n_split = len(rows)
    rows = [r for r in rows if Path(r[0]).is_file()]
    n_missing = n_split - len(rows)
    if args.limit:
        rows = rows[: args.limit]
    paths = [p for p, _, _ in rows]
    n_tags = int(dataset["n_tags"])
    gt = torch.zeros(len(rows), n_tags)
    for n, (_, ti, _) in enumerate(rows):
        gt[n, ti] = 1.0
    gt_rating = torch.tensor([ri for _, _, ri in rows])
    log.info(
        "split=%s N=%d (missing on disk %d) n_tags=%d",
        args.split,
        len(rows),
        n_missing,
        n_tags,
    )

    tagger = AnimaTagger(model_dir, device="cuda")
    ours_p, ours_kept, ours_rate, t_ours = run_ours(tagger, paths, args.batch_size)
    entries = tagger.tag_entries
    names = [e.name for e in entries]
    cats = [e.category for e in entries]
    freq = {
        t["name"]: int(t.get("freq", 0))
        for t in json.loads((model_dir / "vocab.json").read_text())["tags"]
    }
    supported_ours = tagger._supported.clone()
    ratings_ours = list(tagger.ratings)
    rename = rename_recovery_from_rules(tagger.rules)
    del tagger
    torch.cuda.empty_cache()

    new = None
    if args.pixai_ckpt:
        nt = AnimaTagger(Path(args.pixai_ckpt), device="cuda")
        if [e.name for e in nt.tag_entries] != names:
            raise SystemExit(f"{args.pixai_ckpt} vocab differs from {model_dir}")
        new_p, new_kept, new_rate, t_new = run_ours(nt, paths, args.batch_size)
        new = {"p": new_p, "kept": new_kept, "rate": new_rate, "t": t_new}
        del nt
        torch.cuda.empty_cache()

    px_p, px_cfg, t_px, px_peak = run_pixai(paths, args.batch_size, args.pixai_repo)
    px_thr, _ = pixai_card_thresholds(px_cfg)
    px_tags = list(px_cfg.tags)
    card = SimpleNamespace(
        name_to_col={t.replace("_", " "): j for j, t in enumerate(px_tags)}
    )
    vocab_tags = [
        {"name": n.lstrip("@"), "index": i, "category": c}
        for i, (n, c) in enumerate(zip(names, cats))
    ]
    al = align_vocab(vocab_tags, card, rename)
    px_on_ours = torch.full((len(rows), n_tags), float("nan"))
    px_on_ours[:, al.ours_idx] = px_p[:, al.ext_idx]
    px_thr_ours = torch.full((n_tags,), float("nan"))
    px_thr_ours[al.ours_idx] = px_thr[al.ext_idx]
    supported_px = torch.zeros(n_tags, dtype=torch.bool)
    supported_px[al.ours_idx] = True

    ours_pred = torch.zeros_like(gt)
    name_idx = {n: i for i, n in enumerate(names)}
    for r, kept in enumerate(ours_kept):
        for n in kept:
            if n in name_idx:
                ours_pred[r, name_idx[n]] = 1.0

    def tier(n: str) -> str:
        f = freq.get(n, 0)
        return (
            "head"
            if f >= args.freq_head_min
            else "mid"
            if f >= args.freq_mid_min
            else "tail"
        )

    is_artist = torch.tensor([c == "artist" for c in cats])
    both = supported_ours & supported_px & ~is_artist
    has_pos = gt.sum(0) > 0
    ap_ours = average_precision(ours_p, gt)
    ap_px = average_precision(torch.nan_to_num(px_on_ours, nan=0.0), gt)

    slices: dict[str, torch.Tensor] = {"all": both}
    for c in sorted(set(cats) - {"artist"}):
        slices[f"cat:{c}"] = both & torch.tensor([x == c for x in cats])
    for t in ("head", "mid", "tail"):
        slices[f"freq:{t}"] = both & torch.tensor([tier(n) == t for n in names])

    px_filled = torch.nan_to_num(px_on_ours, nan=0.0)
    aps = {"ours": ap_ours, "pixai": ap_px}
    arms = {
        "ours@product": ours_pred,
        f"ours@{args.flat_thr}": (ours_p >= args.flat_thr).float(),
        "pixai@card": (px_filled > torch.nan_to_num(px_thr_ours, nan=2.0)).float(),
        f"pixai@{args.flat_thr}": (px_filled >= args.flat_thr).float(),
    }
    if new is not None:
        new_pred = torch.zeros_like(gt)
        for r, kept in enumerate(new["kept"]):
            for n in kept:
                if n in name_idx:
                    new_pred[r, name_idx[n]] = 1.0
        aps["newckpt"] = average_precision(new["p"], gt)
        arms["newckpt@product"] = new_pred
    metrics_slices: dict[str, dict] = {}
    table = []
    for sname, mask in slices.items():
        m = mask & has_pos
        if m.sum() == 0:
            continue
        entry = {"n_tags": int(m.sum()), "n_pos": int(gt[:, m].sum())}
        for arm, pred in arms.items():
            ap = aps[arm.split("@")[0]][m]
            res = prf(pred[:, m], gt[:, m]) | {"mAP": float(ap.nanmean())}
            entry[arm] = res
            table.append((sname, entry["n_tags"], entry["n_pos"], arm, res))
        metrics_slices[sname] = entry

    # Artists: PixAI only (ours never fires @artist).
    art = supported_px & is_artist & has_pos
    artist_metrics = None
    if art.sum() > 0:
        artist_metrics = {
            "n_tags": int(art.sum()),
            "n_artist_vocab": int(is_artist.sum()),
            "n_pos": int(gt[:, art].sum()),
            "pixai@card": prf(arms["pixai@card"][:, art], gt[:, art])
            | {"mAP": float(ap_px[art].nanmean())},
        }

    # Rating (4-way argmax).
    px_rate_cols = [px_tags.index(PIXAI_RATING[r]) for r in ratings_ours]
    px_rate = px_p[:, px_rate_cols]
    rating_acc = {
        "ours": float((ours_rate.argmax(1) == gt_rating).float().mean()),
        "pixai": float((px_rate.argmax(1) == gt_rating).float().mean()),
    }
    if new is not None:
        rating_acc["newckpt"] = float(
            (new["rate"].argmax(1) == gt_rating).float().mean()
        )

    # Coverage: how much of our vocab each side can emit.
    gt_pos = gt.sum(0)
    coverage = {
        "ours_supported": int(supported_ours.sum()),
        "pixai_matched": int(supported_px.sum()),
        "both_non_artist": int(both.sum()),
        "pixai_unmatched_by_category": al.unmatched_by_category,
        "gt_pos_frac_both": float(gt_pos[both].sum() / gt_pos[~is_artist].sum()),
        "pixai_only_non_artist": int(
            (supported_px & ~supported_ours & ~is_artist).sum()
        ),
        "ours_only": int((supported_ours & ~supported_px & ~is_artist).sum()),
    }
    if new is not None:
        speed_new = len(rows) / new["t"]
    speed = {
        "newckpt_img_per_s": speed_new if new is not None else None,
        "ours_img_per_s": len(rows) / t_ours,
        "pixai_img_per_s": len(rows) / t_px,
        "pixai_peak_vram_gib": px_peak,
        "batch_size": args.batch_size,
    }

    run_dir = make_run_dir("tagger_external", label=args.label, root=RESULTS_ROOT)
    with open(run_dir / "per_tag.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["tag", "category", "tier", "n_pos", *(f"ap_{k}" for k in aps), "both"]
        )
        for i, n in enumerate(names):
            if gt_pos[i] > 0:
                w.writerow(
                    [
                        n,
                        cats[i],
                        tier(n),
                        int(gt_pos[i]),
                        *(f"{a[i]:.4f}" for a in aps.values()),
                        bool(both[i]),
                    ]
                )

    lines = [
        f"# {args.pixai_repo} vs {model_dir.name} — split={args.split}, N={len(rows)} ({n_missing} split images missing on disk)",
        "",
        (
            f"- coverage: ours {coverage['ours_supported']} / pixai {coverage['pixai_matched']} of {n_tags} vocab tags; "
            f"scored on {coverage['both_non_artist']} shared non-artist tags ({coverage['gt_pos_frac_both']:.1%} of GT positives)"
        ),
        f"- pixai unmatched by category: {coverage['pixai_unmatched_by_category']}; pixai-only (non-artist) {coverage['pixai_only_non_artist']}, ours-only {coverage['ours_only']}",
        "- ours@product thresholds: dbv4 card best_threshold + sidecar calibrated ON this split (optimistic for us); pixai@card: its category thresholds",
        *(
            [
                "- newckpt@product: pixai-backed checkpoint as shipped — matched rows refit on the TRAIN split (val held out), sidecar rows calibrated on this split"
            ]
            if new is not None
            else []
        ),
        "- rating acc (4-way): "
        + " / ".join(f"{k} {v:.3f}" for k, v in rating_acc.items()),
        f"- speed (model only, bs={args.batch_size}, bf16): ours {speed['ours_img_per_s']:.1f} img/s, pixai {speed['pixai_img_per_s']:.1f} img/s (peak {px_peak:.1f} GiB)",
        "",
        "| slice | n_tags | n_pos | arm | micro-F1 | macro-F1 | P | R | mAP |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for sname, nt, npos, arm, r in table:
        lines.append(
            f"| {sname} | {nt} | {npos} | {arm} | {r['micro_f1']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['mAP']:.4f} |"
        )
    if artist_metrics:
        a = artist_metrics["pixai@card"]
        lines += [
            "",
            (
                f"## @artist (pixai only): {artist_metrics['n_tags']} of {artist_metrics['n_artist_vocab']} vocab artists matched, "
                f"n_pos={artist_metrics['n_pos']} — micro-F1 {a['micro_f1']:.4f}, macro-F1 {a['macro_f1']:.4f}, mAP {a['mAP']:.4f}"
            ),
        ]
    swing = [
        (i, float(ap_px[i] - ap_ours[i]))
        for i in range(n_tags)
        if both[i] and gt_pos[i] >= 5
    ]
    swing.sort(key=lambda x: x[1])
    lines += ["", "## Largest per-tag AP swings (support >= 5)", "", "pixai >> ours:"]
    for i, d in swing[::-1][:15]:
        lines.append(
            f"- {names[i]} ({cats[i]}, n={int(gt_pos[i])}): pixai {ap_px[i]:.4f} vs ours {ap_ours[i]:.4f}"
        )
    lines += ["", "ours >> pixai:"]
    for i, d in swing[:15]:
        lines.append(
            f"- {names[i]} ({cats[i]}, n={int(gt_pos[i])}): ours {ap_ours[i]:.4f} vs pixai {ap_px[i]:.4f}"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "n_images": len(rows),
            "n_missing_on_disk": n_missing,
            "coverage": coverage,
            "rating_acc": rating_acc,
            "speed": speed,
            "slices": metrics_slices,
            "artist_pixai_only": artist_metrics,
        },
        artifacts=["summary.md", "per_tag.csv"],
    )


if __name__ == "__main__":
    main()
