#!/usr/bin/env python
"""Train the dbv4 sidecar head — copyright / dataset-only characters /
renamed general tags / people-count — on cached backend features.

dbv4 (``animetimm/*.dbv4-full``) has no copyright category, none of our
dataset OCs, and a 2025 danbooru namespace that renamed a few of our general
tags (``black shoes`` → ``black footwear`` …). Those rows are what this head
emits; ``@artist`` is deliberately excluded (not a tagger goal any more).

Stages (all resumable — each is skipped when its output exists):

1. **cache** — one dbv4 forward per ``dataset.json`` image → the MLP-head
   hidden feature (3072-d on caformer_b36, fp16) plus the projected our-vocab
   probs (for the count-tag people rule baseline). ~minutes on a GPU.
2. **train** — a linear head, BCE over the sidecar tag rows + CE over the
   8-way people-count bucket, AdamW + cosine, best epoch by val (mean AP of
   the BCE rows + people accuracy).
3. **calibrate** — per-tag F1-optimal thresholds on val for the BCE rows
   (``calibrate.calibrate_thresholds``, same rule as the PE checkpoints),
   written into the checkpoint's ``thresholds.safetensors`` at those indices.
4. **write** — ``sidecar.safetensors`` + ``sidecar.json`` + ``sidecar_metrics.json``
   into ``--ckpt_dir``; ``AnimaTagger`` picks them up on next load.

::

    make daemon-run ARGS="anime_tools/tagger/cli/train_sidecar.py --ckpt_dir models/captioners/anima-tagger-dbv4"

Gate (proposal ``docs/proposal/tagger_caformer_backend.md`` Phase 2, artist
dropped): copyright macro-F1 ≥ v5's 0.638 on the same val split; people-count
acc ≥ v5's 0.885 — both printed at the end next to the count-tag-rule baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save
from torch.utils.data import DataLoader, Dataset

from anime_tools.captions import tag_rules as tr
from anime_tools.captions.taxonomy import classify_people
from anime_tools.tagger.cli.calibrate import calibrate_thresholds
from anime_tools.tagger.cli.eval_metrics import (
    per_tag_average_precision,
    per_tag_prf,
)
from anime_tools.tagger.dbv4_backend import (
    Dbv4Backend,
    SidecarHead,
    align_vocab,
    preprocess_dbv4,
    rename_recovery_from_rules,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Sidecar candidates = vocab tags dbv4 cannot emit, restricted to these categories.
DEFAULT_CATEGORIES = ("copyright", "character", "general")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--ckpt_dir", default="models/captioners/anima-tagger-dbv4")
    p.add_argument(
        "--feature_cache",
        default=None,
        help="safetensors with cached hidden features; default "
        "post_image_dataset/anima_tagger/dbv4/<arch>_hidden.safetensors",
    )
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    p.add_argument(
        "--no_people", action="store_true", help="skip the people-count head"
    )
    p.add_argument("--batch_size", type=int, default=32, help="backend forward batch")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--train_batch", type=int, default=256)
    p.add_argument("--min_support", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--cache_only", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug: first N images")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Stage 1 — feature cache
# --------------------------------------------------------------------------- #


class _ImageDS(Dataset):
    def __init__(self, paths: Sequence[str], size: int):
        self.paths, self.size = list(paths), size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            im = Image.open(self.paths[i])
            return i, preprocess_dbv4(im, self.size), True
        except (OSError, ValueError):
            return i, torch.zeros(3, self.size, self.size), False


def _collate(b):
    idx, x, ok = zip(*b)
    return torch.tensor(idx), torch.stack(x), torch.tensor(ok)


def build_cache(
    backend: Dbv4Backend,
    image_paths: Sequence[str],
    stems: Sequence[str],
    align_ours: torch.Tensor,
    align_ext: torch.Tensor,
    n_tags: int,
    out_path: Path,
    batch_size: int,
    workers: int,
) -> None:
    ds = _ImageDS(image_paths, backend.img_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers, collate_fn=_collate)
    d_hidden = backend.d_hidden
    hidden = torch.zeros(len(ds), d_hidden, dtype=torch.float16)
    probs = torch.zeros(len(ds), n_tags, dtype=torch.float16)
    ok_all = torch.zeros(len(ds), dtype=torch.bool)
    t0 = time.time()
    for bi, (idx, x, ok) in enumerate(dl):
        out = backend.forward_tensor(x)
        hidden[idx] = out.hidden.to(torch.float16)
        pr = torch.zeros(len(idx), n_tags)
        pr[:, align_ours] = out.probs[:, align_ext]
        probs[idx] = pr.to(torch.float16)
        ok_all[idx] = ok
        if bi % 50 == 0:
            done = min((bi + 1) * batch_size, len(ds))
            log.info(
                "cache: %d/%d (%.1f img/s)", done, len(ds), done / (time.time() - t0)
            )
    n_bad = int((~ok_all).sum())
    if n_bad:
        log.warning("cache: %d unreadable images (zero features, dropped)", n_bad)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st_save(
        {"hidden": hidden, "probs": probs, "ok": ok_all},
        str(out_path),
        metadata={
            "stems": json.dumps(list(stems)),
            "repo": backend.repo,
            "arch": backend.arch,
            "img_size": str(backend.img_size),
            "feature": "mlp_hidden",
        },
    )
    log.info("wrote %s (%d × %d)", out_path, len(ds), d_hidden)


# --------------------------------------------------------------------------- #
# Stage 2 — train
# --------------------------------------------------------------------------- #


def _eval_head(
    head: SidecarHead,
    hidden: torch.Tensor,
    y_bce: torch.Tensor,
    y_people: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    head.eval()
    outs_b, outs_p = [], []
    with torch.no_grad():
        for i in range(0, hidden.shape[0], 1024):
            b, p = head(hidden[i : i + 1024].to(device).float())
            outs_b.append(b.float().cpu())
            if p is not None:
                outs_p.append(p.float().cpu())
    return torch.cat(outs_b), (torch.cat(outs_p) if outs_p else None)


def train_head(
    args,
    hidden: torch.Tensor,
    y_bce: torch.Tensor,
    y_people: torch.Tensor | None,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    bce_indices: list[int],
    people_labels: Sequence[str],
    device: torch.device,
) -> tuple[SidecarHead, dict[str, object]]:
    torch.manual_seed(args.seed)
    head = SidecarHead(
        d_in=hidden.shape[1], bce_indices=bce_indices, people_count_labels=people_labels
    ).to(device)
    opt = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_idx) / args.train_batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch
    )
    h_tr, h_va = hidden[train_idx], hidden[val_idx]
    yb_tr, yb_va = y_bce[train_idx], y_bce[val_idx]
    yp_tr = y_people[train_idx] if y_people is not None else None
    yp_va = y_people[val_idx] if y_people is not None else None
    best, best_state, history = -1.0, None, []
    g = torch.Generator().manual_seed(args.seed)
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(train_idx), generator=g)
        tot = 0.0
        for i in range(0, len(perm), args.train_batch):
            sel = perm[i : i + args.train_batch]
            xb = h_tr[sel].to(device).float()
            b, p = head(xb)
            loss = F.binary_cross_entropy_with_logits(b, yb_tr[sel].to(device))
            if p is not None:
                loss = loss + F.cross_entropy(p, yp_tr[sel].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(sel)
        b_va, p_va = _eval_head(head, h_va, yb_va, yp_va, device)
        ap = per_tag_average_precision(b_va.sigmoid(), yb_va)
        sup = yb_va.sum(0) > 0
        mean_ap = float(ap[sup][~torch.isnan(ap[sup])].mean()) if sup.any() else 0.0
        people_acc = (
            float((p_va.argmax(1) == yp_va).float().mean()) if p_va is not None else 0.0
        )
        score = mean_ap + people_acc
        row = {
            "epoch": ep,
            "train_loss": tot / len(train_idx),
            "val_mean_ap": mean_ap,
            "val_people_acc": people_acc,
        }
        history.append(row)
        log.info(
            "epoch %d loss %.4f | val mAP %.4f people_acc %.4f",
            ep,
            row["train_loss"],
            mean_ap,
            people_acc,
        )
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    return head, {"history": history, "best_score": best}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("backend") != "dbv4":
        raise SystemExit(f"{ckpt_dir} is not a dbv4-backed checkpoint")
    vocab = json.loads((ckpt_dir / "vocab.json").read_text(encoding="utf-8"))
    dataset = json.loads((ckpt_dir / "dataset.json").read_text(encoding="utf-8"))
    rules = tr.load_rules(ckpt_dir / "rules.yaml")
    n_tags = len(vocab["tags"])
    d = cfg["dbv4"]
    backend = Dbv4Backend(
        repo=d["repo"],
        arch=d["arch"],
        img_size=int(d["img_size"]),
        device=device,
        revision=d.get("revision"),
    )
    align = align_vocab(vocab["tags"], backend.card, rename_recovery_from_rules(rules))

    stems: list[str] = list(dataset["stems"])
    image_paths: list[str] = list(dataset["image_paths"])
    if args.limit:
        stems, image_paths = stems[: args.limit], image_paths[: args.limit]
    cache_path = Path(
        args.feature_cache
        or f"post_image_dataset/anima_tagger/dbv4/{d['arch']}_hidden.safetensors"
    )

    # ---- stage 1: cache ----
    if cache_path.exists():
        meta_stems = json.loads(
            __import__("safetensors")
            .safe_open(str(cache_path), "pt")
            .metadata()["stems"]
        )
        if meta_stems != stems:
            raise SystemExit(
                f"{cache_path} was built for a different stem list — delete it or "
                f"pass --feature_cache"
            )
        log.info("cache hit: %s", cache_path)
    else:
        build_cache(
            backend,
            image_paths,
            stems,
            align.ours_idx,
            align.ext_idx,
            n_tags,
            cache_path,
            args.batch_size,
            args.workers,
        )
    if args.cache_only:
        return
    cache = st_load(str(cache_path))
    hidden, probs, ok = cache["hidden"], cache["probs"], cache["ok"]

    # ---- labels ----
    cats = {c.strip() for c in args.categories.split(",") if c.strip()}
    by_index = {int(t["index"]): t for t in vocab["tags"]}
    bce_indices = sorted(int(i) for i, _n, c in align.unmatched if c in cats)
    log.info(
        "sidecar rows: %d (%s)",
        len(bce_indices),
        {
            c: sum(1 for i in bce_indices if by_index[i]["category"] == c)
            for c in sorted(cats)
        },
    )
    stem_pos = {s: i for i, s in enumerate(stems)}
    col_of = {vi: j for j, vi in enumerate(bce_indices)}
    y_bce = torch.zeros(len(stems), len(bce_indices))
    for n, ti in enumerate(dataset["tag_indices"][: len(stems)]):
        for t in ti:
            j = col_of.get(int(t))
            if j is not None:
                y_bce[n, j] = 1.0
    people_labels = (
        [] if args.no_people else list(vocab.get("people_count_labels") or [])
    )
    y_people = (
        torch.tensor(dataset["people_count_indices"][: len(stems)], dtype=torch.long)
        if people_labels
        else None
    )
    split = dataset["split"]
    train_idx = torch.tensor(
        [stem_pos[s] for s in split["train"] if s in stem_pos and ok[stem_pos[s]]]
    )
    val_idx = torch.tensor(
        [stem_pos[s] for s in split["val"] if s in stem_pos and ok[stem_pos[s]]]
    )
    log.info("train %d / val %d", len(train_idx), len(val_idx))

    # ---- stage 2: train ----
    head, train_info = train_head(
        args,
        hidden,
        y_bce,
        y_people,
        train_idx,
        val_idx,
        bce_indices,
        people_labels,
        device,
    )

    # ---- stage 3: calibrate + metrics on val ----
    b_va, p_va = _eval_head(head, hidden[val_idx], None, None, device)
    s_va = b_va.sigmoid()
    yb_va = y_bce[val_idx]
    sweep = torch.arange(0.05, 0.951, 0.05)
    thr_rows, _ = calibrate_thresholds(
        s_va, yb_va, sweep, default=0.5, min_support=args.min_support
    )
    pred = s_va >= thr_rows
    _, _, f1, sup = per_tag_prf(pred, yb_va)
    ap = per_tag_average_precision(s_va, yb_va)
    metrics: dict[str, object] = {"n_val": len(val_idx), "by_category": {}}
    for c in sorted(cats):
        m = torch.tensor([by_index[i]["category"] == c for i in bce_indices]) & (
            sup > 0
        )
        if not m.any():
            continue
        metrics["by_category"][c] = {
            "n_tags_with_val_support": int(m.sum()),
            "macro_f1": float(f1[m].mean()),
            "mean_ap": float(ap[m][~torch.isnan(ap[m])].mean()),
        }
    if p_va is not None:
        yp_va = y_people[val_idx]
        metrics["people_acc_sidecar"] = float((p_va.argmax(1) == yp_va).float().mean())
        # count-tag rule baseline: dbv4 count tags above card thresholds → bucket
        thr_all = st_load(str(ckpt_dir / "thresholds.safetensors"))[
            "thresholds"
        ].float()
        names = [t["name"] for t in vocab["tags"]]
        count_cols = [
            i for i, t in enumerate(vocab["tags"]) if t["category"] == "count"
        ]
        pv = probs[val_idx].float()
        rule = []
        for r in range(len(val_idx)):
            fired = [
                names[i].replace(" ", "_") for i in count_cols if pv[r, i] >= thr_all[i]
            ]
            rule.append(classify_people(fired))
        metrics["people_acc_count_rule"] = float(
            (torch.tensor(rule) == yp_va).float().mean()
        )
    metrics["train"] = train_info
    metrics["v5_reference"] = {"copyright_macro_f1": 0.638, "people_acc": 0.885}

    # ---- stage 4: write ----
    thr_path = ckpt_dir / "thresholds.safetensors"
    thr_all = st_load(str(thr_path))["thresholds"].float()
    thr_all[torch.tensor(bce_indices)] = thr_rows
    st_save({"thresholds": thr_all.contiguous()}, str(thr_path))
    head.cpu().save(
        ckpt_dir,
        extra_meta={
            "repo": d["repo"],
            "arch": d["arch"],
            "categories": sorted(cats),
            "feature_cache": str(cache_path),
            "seed": args.seed,
        },
    )
    with open(ckpt_dir / "sidecar_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("=== sidecar val metrics ===")
    log.info(json.dumps({k: v for k, v in metrics.items() if k != "train"}, indent=2))
    log.info("wrote sidecar into %s", ckpt_dir)


if __name__ == "__main__":
    main()
