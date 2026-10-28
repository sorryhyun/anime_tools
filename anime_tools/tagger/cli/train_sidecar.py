"""Train the sidecar head on cached backbone features; refit backbone thresholds.

A backbone misses some of our vocab: dbv4 (``animetimm/*.dbv4-full``) has no
copyright category, lacks newer and minor characters and renamed a few of our
general tags (``black shoes`` → ``black footwear``); pixai lacks fewer, but
some. Those rows plus the 8-way people-count bucket are what this head emits;
``@artist`` and artist OCs (``shiro (mignon)``, see :func:`select_bce_rows`)
are excluded.

Stages: **cache** (one backbone forward per ``dataset.json`` image → the fp16
hidden feature plus the projected our-vocab probs; skipped when the cache
exists), **train** (linear head, BCE + CE, best epoch by val mean-AP + people
accuracy), **calibrate** (per-tag F1-optimal thresholds for the sidecar rows on
val; with ``--backbone_thresholds train``, the backbone-matched rows are refit
on the *train* split from the cached probs, so val stays held out for them)
and **write**. See ``docs/anima_tagger.md``.
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

from anime_tools._device import add_device_arg, resolve_device
from anime_tools._json import write_json
from anime_tools.captions import tag_rules as tr
from anime_tools.captions.taxonomy import (
    artist_handles_of,
    artist_oc_handle,
    classify_people,
)
from anime_tools.tagger.backends import build_backend
from anime_tools.tagger.cli.build_ckpt import NEVER_FIRE
from anime_tools.tagger.cli.calibrate import DEFAULT_SWEEP, calibrate_thresholds
from anime_tools.tagger.cli.eval_metrics import (
    per_tag_average_precision,
    per_tag_prf,
)
from anime_tools.tagger.data import TaggerCheckpoint
from anime_tools.tagger.dbv4_backend import (
    SidecarHead,
    align_vocab,
    preprocess_dbv4,
    rename_recovery_from_rules,
)
from anime_tools.tagger.dbv4_meta import BACKBONES
from anime_tools.tagger.feature_cache import (
    backbone_cache_path,
    backbone_cache_stems,
    load_backbone_cache,
    multi_hot_from_manifest,
)
from anime_tools.tagger.pixai_backend import preprocess_pixai

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Sidecar candidates = vocab tags the backbone cannot emit, in these categories.
# Not ``deprecated`` (booru's retired-tag type — ``silver hair``, ``black
# footwear``): a head over them was measured at F1 ≈ 0.2 because their
# positives look exactly like the live tag's (``grey hair``); tag_rules.yaml
# ``aliases:`` folds them onto the live name instead.
DEFAULT_CATEGORIES = ("copyright", "character", "general")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--ckpt_dir", default="models/captioners/anima-tagger-pixai")
    p.add_argument(
        "--feature_cache",
        default=None,
        help="safetensors with cached hidden features; default "
        "workspace/anima_tagger/<backend>/<arch>_hidden.safetensors",
    )
    p.add_argument(
        "--backbone_thresholds",
        choices=["keep", "train"],
        default="train",
        help="backbone-matched rows: keep the thresholds in the checkpoint "
        "(the card's), or refit them per tag on the train split",
    )
    p.add_argument(
        "--backbone_min_support",
        type=int,
        default=10,
        help="train positives a matched row needs before it is refit",
    )
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    p.add_argument(
        "--keep_artist_oc",
        action="store_true",
        help="also train rows for artist OCs (`name (handle)` where handle is a "
        "vocab @artist). Off by default: those names mean nothing outside the "
        "dataset they came from and only ever fire as false positives on it.",
    )
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
    add_device_arg(p)
    p.add_argument("--cache_only", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="debug: first N images")
    return p.parse_args()


def select_bce_rows(
    unmatched: Sequence[tuple[int, str, str]],
    cats: set[str],
    vocab_tags: Sequence[dict],
    *,
    keep_artist_oc: bool = False,
) -> tuple[list[int], list[str]]:
    """Sidecar BCE rows: the vocab tags the backbone cannot emit, in ``cats``.

    Returns ``(bce_indices, dropped_artist_oc)``. An artist OC — a tag
    qualified by a vocab ``@artist`` handle, ``shiro (mignon)``, whatever
    category booru filed it under — is left out unless ``keep_artist_oc``: the head would learn it from a few dozen
    positives and then fire it on any look-alike in a stranger's dataset,
    where the name means nothing. Franchise characters the backbone lacks
    (``cartethyia (wuthering waves)``) are exactly what the head is for and
    stay. A dropped row keeps its vocab slot and simply never fires.
    """
    handles = artist_handles_of(t["name"] for t in vocab_tags)
    keep, dropped = [], []
    for i, name, c in unmatched:
        if c not in cats:
            continue
        if not keep_artist_oc and artist_oc_handle(name, handles) is not None:
            dropped.append(name)
            continue
        keep.append(int(i))
    return sorted(keep), dropped


# Stage 1 — feature cache.


PREPROCESS = {"dbv4": preprocess_dbv4, "pixai": preprocess_pixai}


class _ImageDS(Dataset):
    def __init__(self, paths: Sequence[str], kind: str, size: int):
        self.paths, self.kind, self.size = list(paths), kind, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            im = Image.open(self.paths[i])
            return i, PREPROCESS[self.kind](im, self.size), True
        except (OSError, ValueError):
            return i, torch.zeros(3, self.size, self.size), False


def _collate(b):
    idx, x, ok = zip(*b)
    return torch.tensor(idx), torch.stack(x), torch.tensor(ok)


def build_cache(
    backend,
    image_paths: Sequence[str],
    stems: Sequence[str],
    align_ours: torch.Tensor,
    align_ext: torch.Tensor,
    n_tags: int,
    out_path: Path,
    batch_size: int,
    workers: int,
) -> None:
    ds = _ImageDS(image_paths, backend.kind, backend.img_size)
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
            "backend": backend.kind,
            "repo": backend.repo,
            "revision": str(backend.revision),
            "arch": backend.arch,
            "img_size": str(backend.img_size),
            "feature": FEATURE[backend.kind],
        },
    )
    log.info("wrote %s (%d × %d)", out_path, len(ds), d_hidden)


FEATURE = {"dbv4": "mlp_hidden", "pixai": "attn_pool"}


# Stage 2 — train.


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


def main() -> None:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    device = torch.device(resolve_device(args.device))
    ckpt = TaggerCheckpoint.from_dir(ckpt_dir, require=("vocab", "dataset", "config"))
    cfg, vocab, dataset = ckpt.config, ckpt.vocab, ckpt.dataset
    if cfg.get("backend") not in BACKBONES:
        raise SystemExit(f"{ckpt_dir} names no known backbone ({sorted(BACKBONES)})")
    rules = tr.load_rules(ckpt_dir / "rules.yaml")
    n_tags = ckpt.n_tags
    backend = build_backend(cfg, device=device)
    align = align_vocab(vocab["tags"], backend.card, rename_recovery_from_rules(rules))

    stems: list[str] = list(dataset["stems"])
    image_paths: list[str] = list(dataset["image_paths"])
    if args.limit:
        stems, image_paths = stems[: args.limit], image_paths[: args.limit]
    cache_path = backbone_cache_path(backend.kind, backend.arch, args.feature_cache)

    # ---- stage 1: cache ----
    if cache_path.exists():
        if backbone_cache_stems(cache_path) != stems:
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
    cache, _ = load_backbone_cache(cache_path)
    hidden, probs, ok = cache["hidden"], cache["probs"], cache["ok"]

    # ---- labels ----
    cats = {c.strip() for c in args.categories.split(",") if c.strip()}
    by_index = {int(t["index"]): t for t in vocab["tags"]}
    bce_indices, dropped_oc = select_bce_rows(
        align.unmatched, cats, vocab["tags"], keep_artist_oc=args.keep_artist_oc
    )
    log.info(
        "sidecar rows: %d (%s); artist OCs left out: %d %s",
        len(bce_indices),
        {
            c: sum(1 for i in bce_indices if by_index[i]["category"] == c)
            for c in sorted(cats)
        },
        len(dropped_oc),
        dropped_oc,
    )
    stem_pos = {s: i for i, s in enumerate(stems)}
    y_bce = multi_hot_from_manifest(
        dataset["tag_indices"][: len(stems)],
        len(bce_indices),
        col_of={vi: j for j, vi in enumerate(bce_indices)},
    )
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
    thr_rows, _ = calibrate_thresholds(
        s_va, yb_va, DEFAULT_SWEEP, default=0.5, min_support=args.min_support
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
        # count-tag rule baseline: backbone count tags above their thresholds → bucket
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

    thr_path = ckpt_dir / "thresholds.safetensors"
    thr_all = st_load(str(thr_path))["thresholds"].float()

    # ---- stage 3b: backbone-matched rows, refit on train ----
    if args.backbone_thresholds == "train":
        matched = align.ours_idx
        y_tr = multi_hot_from_manifest(
            [dataset["tag_indices"][i] for i in train_idx.tolist()], n_tags
        )[:, matched]
        p_tr = probs[train_idx][:, matched].float()
        thr_fit, f1_fit = calibrate_thresholds(
            p_tr,
            y_tr,
            DEFAULT_SWEEP,
            default=-1.0,
            min_support=args.backbone_min_support,
        )
        refit = thr_fit >= 0
        thr_all[matched[refit]] = thr_fit[refit]
        metrics["backbone_thresholds"] = {
            "source": "train-split per-tag F1 sweep",
            "n_matched": int(matched.numel()),
            "n_refit": int(refit.sum()),
            "min_support": args.backbone_min_support,
            "n_train": len(train_idx),
            "mean_train_f1_refit": float(f1_fit[refit].mean()) if refit.any() else 0.0,
        }
        log.info("backbone thresholds: %s", metrics["backbone_thresholds"])

    # ---- stage 4: write ----
    # Every row the backbone cannot emit and this head does not carry must
    # never fire — including a row a previous head carried (a dropped OC).
    unmatched_idx = [int(i) for i, _n, _c in align.unmatched]
    thr_all[torch.tensor(unmatched_idx)] = NEVER_FIRE
    thr_all[torch.tensor(bce_indices)] = thr_rows
    st_save({"thresholds": thr_all.contiguous()}, str(thr_path))
    head.feature = FEATURE[backend.kind]
    head.cpu().save(
        ckpt_dir,
        extra_meta={
            "backend": backend.kind,
            "repo": backend.repo,
            "arch": backend.arch,
            "categories": sorted(cats),
            "dropped_artist_oc": dropped_oc,
            "feature_cache": str(cache_path),
            "seed": args.seed,
        },
    )
    write_json(ckpt_dir / "sidecar_metrics.json", metrics)
    log.info("=== sidecar val metrics ===")
    log.info(json.dumps({k: v for k, v in metrics.items() if k != "train"}, indent=2))
    log.info("wrote sidecar into %s", ckpt_dir)


if __name__ == "__main__":
    main()
