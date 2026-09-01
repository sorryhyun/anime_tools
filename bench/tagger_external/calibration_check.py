#!/usr/bin/env python
"""Does the dbv4 backend's confidence mean anything on our images, or only rank?

CPU-only: reads the projected our-vocab probs the sidecar trainer cached per
``dataset.json`` stem against the val split's ground truth, and reports ECE
(15 equal-width bins) plus a reliability curve per frequency tier, and how far
the card's per-tag ``best_threshold`` sits from the val-optimal one. Writes
``summary.md``, ``result.json`` and ``reliability.csv`` into the run dir.

    uv run python bench/tagger_external/calibration_check.py --label dbv4-calib
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from safetensors.torch import load_file as st_load

from anime_tools.tagger.cli.calibrate import DEFAULT_SWEEP, calibrate_thresholds
from anime_tools.tagger.data import TaggerCheckpoint
from anime_tools.tagger.dbv4_backend import SidecarHead
from anime_tools.tagger.feature_cache import (
    dbv4_cache_path,
    load_dbv4_cache,
    multi_hot_from_manifest,
)
from bench._common import make_run_dir, write_result

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--model_dir", default="models/captioners/anima-tagger-dbv4")
    p.add_argument("--feature_cache", default=None)
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--freq_head_min", type=int, default=1000)
    p.add_argument("--freq_mid_min", type=int, default=200)
    p.add_argument("--min_support", type=int, default=5)
    p.add_argument("--agree_tol", type=float, default=0.10)
    p.add_argument("--gate_ece", type=float, default=0.05)
    p.add_argument("--gate_agree", type=float, default=0.80)
    p.add_argument("--label", default=None)
    return p.parse_args()


def ece_curve(
    conf: torch.Tensor, hit: torch.Tensor, bins: int
) -> tuple[float, list[dict]]:
    """Pooled ECE + reliability rows over flat (conf, hit) cells."""
    edges = torch.linspace(0, 1, bins + 1)
    n = conf.numel()
    ece = 0.0
    rows = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf >= lo) & (conf < hi) if b < bins - 1 else (conf >= lo) & (conf <= hi)
        k = int(m.sum())
        if k == 0:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0})
            continue
        c, a = float(conf[m].mean()), float(hit[m].mean())
        ece += (k / n) * abs(a - c)
        rows.append(
            {"bin_lo": float(lo), "bin_hi": float(hi), "n": k, "conf": c, "acc": a}
        )
    return ece, rows


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    ckpt = TaggerCheckpoint.from_dir(
        model_dir, require=("vocab", "dataset"), backend="dbv4"
    )
    cfg, vocab, dataset = ckpt.config, ckpt.vocab, ckpt.dataset
    n_tags = ckpt.n_tags
    arch = cfg["dbv4"]["arch"]
    cache_path = dbv4_cache_path(arch, args.feature_cache)
    cache, cached_stems = load_dbv4_cache(cache_path)
    pos = {s: i for i, s in enumerate(cached_stems)}
    stems = [
        s for s in dataset["split"][args.split] if s in pos and cache["ok"][pos[s]]
    ]
    rows = torch.tensor([pos[s] for s in stems])
    probs = cache["probs"][rows].float()
    tag_idx = dict(zip(dataset["stems"], dataset["tag_indices"]))
    gt = multi_hot_from_manifest([tag_idx[s] for s in stems], n_tags)

    # supported (dbv4-native) vs sidecar rows
    unmatched = {u["index"] for u in cfg["alignment"]["unmatched"]}
    native = torch.tensor([i not in unmatched for i in range(n_tags)])
    head = SidecarHead.load(model_dir)
    sidecar = torch.zeros(n_tags, dtype=torch.bool)
    if head is not None:
        with torch.no_grad():
            bce, _ = head(cache["hidden"][rows].float())
        idx = torch.tensor(head.bce_indices)
        probs[:, idx] = bce.sigmoid()
        sidecar[idx] = True
    card_thr = st_load(str(model_dir / "thresholds.safetensors"))["thresholds"].float()

    freqs = torch.tensor([t["freq"] for t in vocab["tags"]])
    tier = torch.full((n_tags,), 2)  # 0 head / 1 mid / 2 tail
    tier[freqs >= args.freq_mid_min] = 1
    tier[freqs >= args.freq_head_min] = 0
    support = gt.sum(0)
    scored = support >= args.min_support

    # val-optimal thresholds for the transfer test
    val_thr, _ = calibrate_thresholds(
        probs, gt, DEFAULT_SWEEP, default=float("nan"), min_support=args.min_support
    )

    slices: dict[str, torch.Tensor] = {
        "native:head": native & (tier == 0),
        "native:mid": native & (tier == 1),
        "native:tail": native & (tier == 2),
        "native:all": native.clone(),
        "sidecar:all": sidecar.clone(),
    }
    for c in ("copyright", "character", "general"):
        m = torch.tensor([t["category"] == c for t in vocab["tags"]]) & sidecar
        if m.any():
            slices[f"sidecar:{c}"] = m

    metrics: dict[str, object] = {
        "n_images": len(stems),
        "split": args.split,
        "slices": {},
    }
    rel_rows = []
    for name, m in slices.items():
        m = m & scored
        if not m.any():
            continue
        conf = probs[:, m].flatten()
        hit = gt[:, m].flatten()
        ece, curve = ece_curve(conf, hit, args.bins)
        for r in curve:
            rel_rows.append({"slice": name, **r})
        # threshold transfer on tags that have a val-optimal threshold
        has = m & ~torch.isnan(val_thr) & (card_thr <= 1.0)
        d = (card_thr[has] - val_thr[has]).abs()
        # overconfidence direction: mean(conf) − mean(hit) over cells ≥ 0.5
        hi = conf >= 0.5
        metrics["slices"][name] = {
            "n_tags": int(m.sum()),
            "n_cells": int(conf.numel()),
            "ece": ece,
            "mean_conf_minus_acc_above_0.5": float(conf[hi].mean() - hit[hi].mean())
            if hi.any()
            else float("nan"),
            "thr_n_compared": int(has.sum()),
            "thr_agree_within_tol": float((d <= args.agree_tol).float().mean())
            if has.any()
            else float("nan"),
            "thr_median_abs_delta": float(d.median()) if has.any() else float("nan"),
            "thr_mean_card_minus_val": float((card_thr[has] - val_thr[has]).mean())
            if has.any()
            else float("nan"),
        }
    head_s = metrics["slices"].get("native:head", {})
    metrics["gate"] = {
        "ece_head": head_s.get("ece"),
        "ece_head_pass": bool(head_s.get("ece", 1.0) <= args.gate_ece),
        "thr_agree_head": head_s.get("thr_agree_within_tol"),
        "thr_agree_head_pass": bool(
            head_s.get("thr_agree_within_tol", 0.0) >= args.gate_agree
        ),
    }

    run_dir = make_run_dir("tagger_external", args.label)
    with open(run_dir / "reliability.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["slice", "bin_lo", "bin_hi", "n", "conf", "acc"]
        )
        w.writeheader()
        w.writerows(rel_rows)
    lines = [
        f"# dbv4 calibration — {model_dir.name}, split={args.split}, N={len(stems)}",
        "",
        (
            f"gate: head ECE {head_s.get('ece', float('nan')):.4f} (≤ {args.gate_ece}) → "
            f"{'PASS' if metrics['gate']['ece_head_pass'] else 'FAIL'}; head threshold "
            f"agreement ±{args.agree_tol} {head_s.get('thr_agree_within_tol', float('nan')):.3f} "
            f"(≥ {args.gate_agree}) → {'PASS' if metrics['gate']['thr_agree_head_pass'] else 'FAIL'}"
        ),
        "",
        "| slice | tags | cells | ECE | conf−acc (≥0.5) | thr agree ±tol | median |Δthr| | mean card−val |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in metrics["slices"].items():
        lines.append(
            f"| {name} | {s['n_tags']} | {s['n_cells']} | {s['ece']:.4f} | "
            f"{s['mean_conf_minus_acc_above_0.5']:+.3f} | {s['thr_agree_within_tol']:.3f} "
            f"(n={s['thr_n_compared']}) | {s['thr_median_abs_delta']:.3f} | "
            f"{s['thr_mean_card_minus_val']:+.3f} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["summary.md", "reliability.csv"],
        device=torch.device("cpu"),
    )
    log.info("wrote %s", run_dir)


if __name__ == "__main__":
    main()
