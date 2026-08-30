#!/usr/bin/env python
"""Build a dbv4-backed AnimaTagger checkpoint dir from an existing PE checkpoint.

The new dir carries **only our data** — vocab / rules / groups / dataset split
copied from the source checkpoint, ``config.json`` naming the upstream dbv4
repo, and a ``thresholds.safetensors`` seeded from the card's per-tag
``best_threshold`` (tags dbv4 cannot emit get a never-fire threshold until a
sidecar head is trained for them; see ``train_sidecar.py``). No weights are
copied or downloaded here: the GPL-3.0 dbv4 weights come from HF under the
user's own token at first ``AnimaTagger`` use.

::

    uv run python -m anime_tools.tagger.cli.build_dbv4_ckpt \\
        --src _archive/anima_tagger_training/checkpoints/anima-tagger-v5 \\
        --out models/captioners/anima-tagger-dbv4

Then ``AnimaTagger("models/captioners/anima-tagger-dbv4")`` works anywhere a
``--tagger_dir`` is accepted.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from anime_tools.captions import tag_rules as tr
from anime_tools.tagger.dbv4_backend import (
    DEFAULT_DBV4_ARCH,
    DEFAULT_DBV4_IMG_SIZE,
    DEFAULT_DBV4_REPO,
    align_vocab,
    load_dbv4_card,
    rename_recovery_from_rules,
)

logger = logging.getLogger(__name__)

# Threshold that a sigmoid can never reach — tags without a score source.
NEVER_FIRE = 1.01
COPIED = ("vocab.json", "rules.yaml", "groups.yaml", "dataset.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--src", default="_archive/anima_tagger_training/checkpoints/anima-tagger-v5"
    )
    p.add_argument("--out", default="models/captioners/anima-tagger-dbv4")
    p.add_argument("--repo", default=DEFAULT_DBV4_REPO)
    p.add_argument("--arch", default=DEFAULT_DBV4_ARCH)
    p.add_argument("--img_size", type=int, default=DEFAULT_DBV4_IMG_SIZE)
    p.add_argument("--revision", default=None, help="pin the upstream repo revision")
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    src, out = Path(args.src), Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} exists — pass --force to rebuild")
    out.mkdir(parents=True, exist_ok=True)
    for f in COPIED:
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
        elif f in ("vocab.json", "rules.yaml"):
            raise SystemExit(f"{src / f} missing")

    with open(out / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    rules = tr.load_rules(out / "rules.yaml")
    card = load_dbv4_card(args.repo, revision=args.revision)
    align = align_vocab(vocab["tags"], card, rename_recovery_from_rules(rules))
    n_tags = len(vocab["tags"])

    thresholds = torch.full((n_tags,), NEVER_FIRE)
    thresholds[align.ours_idx] = card.best_thresholds()[align.ext_idx]
    st_save(
        {"thresholds": thresholds.contiguous()}, str(out / "thresholds.safetensors")
    )

    with open(src / "config.json", encoding="utf-8") as f:
        src_cfg = json.load(f)
    cfg = {
        "backend": "dbv4",
        "dbv4": {
            "repo": args.repo,
            "arch": args.arch,
            "img_size": args.img_size,
            "revision": args.revision,
            "n_classes": card.n_classes,
            "license": "gpl-3.0 (upstream weights; fetched, never vendored)",
        },
        "source_checkpoint": str(src),
        "source_vocab_meta": {
            k: src_cfg.get(k)
            for k in ("ratings", "people_count_labels", "min_freq")
            if k in src_cfg
        },
        "alignment": {
            "n_supported": int(align.ours_idx.numel()),
            "n_tags": n_tags,
            "unmatched_by_category": align.unmatched_by_category,
            "unmatched": [
                {"index": i, "name": n, "category": c} for i, n, c in align.unmatched
            ],
        },
        "thresholds": {
            "source": "dbv4 card best_threshold",
            "never_fire": NEVER_FIRE,
        },
    }
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    logger.info(
        "wrote %s: %d/%d tags supported by %s; unmatched by category %s",
        out,
        align.ours_idx.numel(),
        n_tags,
        args.repo,
        align.unmatched_by_category,
    )


if __name__ == "__main__":
    main()
