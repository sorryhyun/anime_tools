"""Build a backbone-backed AnimaTagger checkpoint dir from an existing checkpoint.

The new dir carries only our data — vocab / rules / groups / dataset split
copied from ``--src``, a ``config.json`` naming the upstream backbone
(``--backend pixai`` or ``dbv4``), and a ``thresholds.safetensors`` seeded from
the backbone card: per-tag ``best_threshold`` for dbv4, per-category for pixai.
Tags the backbone cannot emit get a never-fire threshold until
``train_sidecar`` gives them a score source (and, with
``--backbone_thresholds train``, refits the matched rows on the train split).
No weights are copied; only the card (tag list) is downloaded.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file as st_save

from anime_tools._json import read_json, write_json
from anime_tools.captions import tag_rules as tr
from anime_tools.tagger.backends import build_backend
from anime_tools.tagger.dbv4_backend import align_vocab, rename_recovery_from_rules
from anime_tools.tagger.dbv4_meta import (
    BACKBONES,
    DEFAULT_BACKEND,
    DEFAULT_DBV4_ARCH,
    DEFAULT_DBV4_IMG_SIZE,
    DEFAULT_PIXAI_IMG_SIZE,
)

logger = logging.getLogger(__name__)

# Threshold that a sigmoid can never reach — tags without a score source.
NEVER_FIRE = 1.01
COPIED = ("vocab.json", "rules.yaml", "groups.yaml", "dataset.json")
DEFAULT_OUT = {
    "pixai": "models/captioners/anima-tagger-pixai",
    "dbv4": "models/captioners/anima-tagger-dbv4",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--backend", choices=sorted(BACKBONES), default=DEFAULT_BACKEND)
    p.add_argument("--src", default="models/captioners/anima-tagger-dbv4")
    p.add_argument("--out", default=None, help="default: per backend")
    p.add_argument("--repo", default=None, help="default: the backend's")
    p.add_argument("--arch", default=DEFAULT_DBV4_ARCH, help="dbv4 only")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument(
        "--revision",
        default=None,
        help="pin the upstream repo revision (pixai: pinned)",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing --out")
    return p.parse_args()


def backbone_block(args) -> dict:
    spec = BACKBONES[args.backend]
    block: dict[str, object] = {
        "repo": args.repo or spec.default_repo,
        "revision": args.revision or spec.default_revision,
        "license": spec.license
        + (" (upstream weights; fetched, never vendored)" if spec.gated else ""),
    }
    if args.backend == "dbv4":
        block |= {"arch": args.arch, "img_size": args.img_size or DEFAULT_DBV4_IMG_SIZE}
    else:
        block |= {
            "arch": "pixai-vitdet",
            "img_size": args.img_size or DEFAULT_PIXAI_IMG_SIZE,
        }
    return block


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out or DEFAULT_OUT[args.backend])
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} exists — pass --force to rebuild")
    out.mkdir(parents=True, exist_ok=True)
    for f in COPIED:
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
        elif f in ("vocab.json", "rules.yaml"):
            raise SystemExit(f"{src / f} missing")

    vocab = read_json(out / "vocab.json")
    rules = tr.load_rules(out / "rules.yaml")
    block = backbone_block(args)
    card = build_backend({"backend": args.backend, args.backend: block}).card
    block["n_classes"] = card.n_classes
    align = align_vocab(vocab["tags"], card, rename_recovery_from_rules(rules))
    n_tags = len(vocab["tags"])

    thresholds = torch.full((n_tags,), NEVER_FIRE)
    thresholds[align.ours_idx] = card.best_thresholds()[align.ext_idx]
    st_save(
        {"thresholds": thresholds.contiguous()}, str(out / "thresholds.safetensors")
    )

    src_cfg = read_json(src / "config.json")
    cfg = {
        "backend": args.backend,
        args.backend: block,
        "source_checkpoint": str(src),
        "source_vocab_meta": src_cfg.get("source_vocab_meta")
        or {
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
            "source": f"{args.backend} card best_threshold",
            "never_fire": NEVER_FIRE,
        },
    }
    write_json(out / "config.json", cfg)
    logger.info(
        "wrote %s: %d/%d tags supported by %s; unmatched by category %s",
        out,
        align.ours_idx.numel(),
        n_tags,
        block["repo"],
        align.unmatched_by_category,
    )


if __name__ == "__main__":
    main()
