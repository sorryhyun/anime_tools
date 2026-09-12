"""AnimaTagger — multi-label tagger over the Anima caption vocabulary.

Public surface: ``predict`` / ``predict_caption``. Checkpoint dir is
``config.json`` + ``vocab.json`` + ``rules.yaml`` + ``thresholds.safetensors``,
optionally ``groups.yaml`` (group-aware prediction: one argmax winner per
``softmax``/``softmax_when_solo`` group instead of the sigmoid threshold) and
``sidecar.safetensors``. See ``docs/anima_tagger.md``.

Everything after the score vector — thresholds, softmax groups, count dedupe,
character floor, top-1 copyright, slot order — is post-processing of
``{tag: prob}``. Captions come out in ``SLOT_ORDER`` with underscores replaced
by spaces (Anima's training-time T5 input).

The two halves that need no torch live beside this one and are re-exported from
it, so the vocab build and the ComfyUI node can reach them without loading a
model: :mod:`anime_tools.tagger.schema` (what a tag means to a checkpoint) and
:mod:`anime_tools.tagger.fetch` (getting one onto disk).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file as st_load

from anime_tools._device import resolve_device
from anime_tools._json import read_json
from anime_tools.captions import tag_groups as tg
from anime_tools.captions import tag_rules as tr
from anime_tools.captions.taxonomy import classify_people, is_solo_names
from anime_tools.captions.vocab_io import load_vocab
from anime_tools.tagger.dbv4_backend import (
    UNSUPPORTED_LOGIT,
    Dbv4Backend,
    SidecarHead,
    align_vocab,
    default_dtype,
    probs_to_logits,
    rename_recovery_from_rules,
)

# Re-exported from the three torch-free leaves; callers import them from here.
from anime_tools.tagger.dbv4_meta import (
    DEFAULT_DBV4_ARCH,
    DEFAULT_DBV4_IMG_SIZE,
    DEFAULT_DBV4_REPO,
    DEFAULT_TAGGER_DIR,
)
from anime_tools.tagger.fetch import (
    ensure_tagger_backbone,
    ensure_tagger_checkpoint,
    is_dbv4_dir,
)
from anime_tools.tagger.schema import (
    GIRLS_COUNT_RE,
    PEOPLE_COUNT_LABELS,
    RATINGS,
    SLOT_ORDER,
    TAG_TYPE_NAMES,
    TagEntry,
    dedupe_count_tags,
    fix_artist_category,
    underscore_to_space,
)

__all__ = [
    "PEOPLE_COUNT_LABELS",
    "RATINGS",
    "SLOT_ORDER",
    "TAG_TYPE_NAMES",
    "AnimaTagger",
    "dedupe_count_tags",
    "ensure_tagger_backbone",
    "ensure_tagger_checkpoint",
    "is_dbv4_dir",
]

logger = logging.getLogger(__name__)


def _load_thresholds(path: Path, n_tags: int, default: float = 0.5) -> torch.Tensor:
    if not path.exists():
        logger.warning(
            "no thresholds.safetensors at %s - using default=%.2f", path, default
        )
        return torch.full((n_tags,), default)
    d = st_load(str(path))
    t = d["thresholds"]
    if t.shape != (n_tags,):
        raise ValueError(f"thresholds shape {tuple(t.shape)} != ({n_tags},)")
    return t


class AnimaTagger:
    """Multi-label tagger over the Anima-distribution vocabulary."""

    def __init__(
        self,
        ckpt_dir: str | Path = DEFAULT_TAGGER_DIR,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        pe_ckpt: str | Path | None = None,
        character_floor: float = 0.5,
        pe_lora_path: str | Path | None = None,
        pe_lora_disabled: bool = False,
        pe_aux_ckpt: str | Path | None = None,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(resolve_device(device))
        self.dtype = dtype if dtype is not None else default_dtype(self.device)
        # No-ops, accepted for call-site compat (ComfyUI workflows, old scripts).
        del pe_ckpt, pe_aux_ckpt, pe_lora_path, pe_lora_disabled
        # Absolute confidence floor for characters, above the per-tag F1
        # threshold (some F1 thresholds are ~0.05, too permissive on its own).
        self._character_floor = float(character_floor)

        cfg_d = read_json(self.ckpt_dir / "config.json")
        self._cfg_d = cfg_d
        self.backend_kind: str = str(cfg_d.get("backend", "pe"))
        if self.backend_kind != "dbv4":
            raise ValueError(
                f"unsupported tagger backend {self.backend_kind!r} in "
                f"{self.ckpt_dir / 'config.json'}: the legacy in-house 'pe' "
                "dual-encoder head was removed 2026-08-30. Use a dbv4-backed "
                f"checkpoint (default {DEFAULT_TAGGER_DIR}; "
                "`python -m anime_tools.downloads tagger` or "
                "`python -m anime_tools.tagger.cli.build_dbv4_ckpt`)."
            )
        self._dbv4: Dbv4Backend | None = None
        self._sidecar: SidecarHead | None = None

        vocab = load_vocab(self.ckpt_dir)
        self.tag_entries: list[TagEntry] = [
            TagEntry(
                name=t["name"],
                index=int(t["index"]),
                category=fix_artist_category(str(t["category"]), t["name"]),
                median_pos=float(t.get("median_pos", 0.0)),
            )
            for t in vocab["tags"]
        ]
        self.ratings: list[str] = list(vocab["ratings"])
        # empty = legacy/disabled checkpoint (no people-count head, n_people_counts == 0)
        self.people_count_labels: list[str] = list(
            vocab.get("people_count_labels") or []
        )
        # "original" copyright index; the uncertainty-fallback in predict()
        # when a character misses _character_floor.
        self._original_idx: int | None = next(
            (
                e.index
                for e in self.tag_entries
                if e.name == "original" and e.category == "copyright"
            ),
            None,
        )
        self._by_cat: dict[str, list[tuple[int, float, str]]] = {}
        for e in self.tag_entries:
            cat = e.category if e.category in SLOT_ORDER else "general"
            self._by_cat.setdefault(cat, []).append((e.index, e.median_pos, e.name))
        for cat in self._by_cat:
            self._by_cat[cat].sort(key=lambda triple: (triple[1], triple[2]))

        self.n_tags = len(self.tag_entries)
        self.rules = tr.load_rules(self.ckpt_dir / "rules.yaml")
        self._init_dbv4_backend(cfg_d)
        if int(self.cfg.n_tags) != self.n_tags:
            raise ValueError(
                f"vocab.json has {self.n_tags} tags but the head expects {self.cfg.n_tags}"
            )

        self.thresholds = _load_thresholds(
            self.ckpt_dir / "thresholds.safetensors", n_tags=self.n_tags
        )
        self.thresholds_dev = self.thresholds.to(self.device)
        # Attached to every ``predict`` output so a consumer can tell how far a
        # score fell short (the position-clause bag relaxation reads it).
        self.threshold_map: dict[str, float] = {
            e.name: float(t) for e, t in zip(self.tag_entries, self.thresholds)
        }

        # Optional groups snapshot; None when missing (older/flat-vocab).
        groups_path = self.ckpt_dir / "groups.yaml"
        self._groups: tg.TagGroups | None = None
        self._group_lookup: dict[str, dict] = {}
        if groups_path.exists():
            self._groups = tg.load_groups(groups_path)
            tag_to_idx = {e.name: e.index for e in self.tag_entries}
            for g in self._groups.groups:
                if g.mode not in ("softmax", "softmax_when_solo"):
                    continue
                tag_idx = [tag_to_idx[t] for t in g.tags if t in tag_to_idx]
                if not tag_idx:
                    continue
                # sentinel groups carry a synthetic "<none:group>" slot, appended
                # so argmax can pick "none of these" and emit nothing.
                sentinel_local: int | None = None
                if g.sentinel:
                    s_idx = tag_to_idx.get(tg.sentinel_tag_name(g.name))
                    if s_idx is not None:
                        sentinel_local = len(tag_idx)
                        tag_idx.append(s_idx)
                self._group_lookup[g.name] = {
                    "mode": g.mode,
                    "tag_idx": torch.tensor(
                        tag_idx, dtype=torch.long, device=self.device
                    ),
                    "tag_names": tuple(g.tags),
                    "escape_names": tuple(g.escape),
                    "sentinel_local": sentinel_local,
                }

    def _build_dbv4(self, d: dict) -> Dbv4Backend:
        """The backbone this checkpoint's ``config.json['dbv4']`` block describes.

        There is one runtime. The exported ONNX graph this used to choose between
        was retired 2026-09-09: it existed because timm on an Apple CPU was 3.7x
        slower than onnxruntime, and on the GPU every such machine actually has
        (MPS) torch is 5.7x *faster* than the graph was — 87 ms an image against
        491, at 2.3e-06 on the scores.
        """
        return Dbv4Backend(
            repo=d.get("repo", DEFAULT_DBV4_REPO),
            arch=d.get("arch", DEFAULT_DBV4_ARCH),
            img_size=int(d.get("img_size", DEFAULT_DBV4_IMG_SIZE)),
            device=self.device,
            dtype=self.dtype,
            revision=d.get("revision"),
        )

    def _init_dbv4_backend(self, cfg_d: dict) -> None:
        """External dbv4 tagger projected onto our vocab (+ optional sidecar)."""
        from types import SimpleNamespace

        d = dict(cfg_d.get("dbv4") or {})
        self._dbv4 = self._build_dbv4(d)
        # readback / bench read ``cfg.n_tags``; the pool kinds are vestigial.
        self.cfg = SimpleNamespace(
            n_tags=self.n_tags, pool_kind=None, pool_kind_aux=None
        )
        vocab_tags = [
            {"name": e.name, "index": e.index, "category": e.category}
            for e in self.tag_entries
        ]
        self._align = align_vocab(
            vocab_tags, self._dbv4.card, rename_recovery_from_rules(self.rules)
        )
        self._align_ours = self._align.ours_idx
        self._align_ext = self._align.ext_idx
        self._supported = self._align.supported_mask(self.n_tags)
        self._rating_cols = [self._dbv4.card.rating_cols.get(r) for r in self.ratings]
        self._sidecar = SidecarHead.load(self.ckpt_dir)
        if self._sidecar is not None:
            self._sidecar.to(self.device)
            self._sidecar_bce_idx = torch.tensor(
                self._sidecar.bce_indices, dtype=torch.long
            )
            self._supported[self._sidecar_bce_idx] = True
            if (
                self._sidecar.people_count_labels
                and list(self._sidecar.people_count_labels) != self.people_count_labels
            ):
                raise ValueError("sidecar people_count_labels disagree with vocab.json")
        logger.info(
            "AnimaTagger[dbv4]: %s → %d/%d vocab tags supported (unmatched by "
            "category: %s); sidecar=%s",
            self._dbv4.repo,
            int(self._supported.sum()),
            self.n_tags,
            self._align.unmatched_by_category,
            "none"
            if self._sidecar is None
            else f"{self._sidecar.n_bce} bce + "
            f"{len(self._sidecar.people_count_labels)} people",
        )

    @torch.no_grad()
    def _heads_forward(
        self, pil_img: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Head pass: dbv4 backend + optional sidecar → ``(tag_logits,
        rating_probs, people_probs | None)`` on ``self.device``.

        Tag logits are the projected sigmoid probs mapped back through logit();
        tags the backend cannot emit sit at ``UNSUPPORTED_LOGIT``, so they never
        clear a threshold or win a group argmax.
        """
        out = self._dbv4.forward([pil_img])
        probs = torch.zeros(self.n_tags)
        probs[self._align_ours] = out.probs[0, self._align_ext]
        people: torch.Tensor | None = None
        if self._sidecar is not None:
            bce, people_logits = self._sidecar(out.hidden.to(self.device))
            probs[self._sidecar_bce_idx] = bce[0].float().sigmoid().cpu()
            if people_logits is not None:
                people = people_logits[0].float().softmax(dim=-1)
        tag_logits = probs_to_logits(probs)
        tag_logits[~self._supported] = UNSUPPORTED_LOGIT
        # dbv4 ratings are 4 independent sigmoids; normalise to a distribution.
        rating = torch.tensor(
            [
                float(out.probs[0, c]) if c is not None else 0.0
                for c in self._rating_cols
            ]
        )
        rating = rating / rating.sum().clamp(min=1e-6)
        return tag_logits.to(self.device), rating.to(self.device), people

    @torch.no_grad()
    def tag_logits(self, pil_img: Image.Image) -> torch.Tensor:
        """Image → raw ``[n_tags]`` tag logits (float32, cpu)."""
        return self._heads_forward(pil_img)[0].float().cpu()

    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> dict[str, object]:
        """Run one image through the head; return raw + thresholded outputs.

        Returns a dict with ``rating`` / ``rating_scores``; ``people_count`` /
        ``people_count_scores`` (absent on legacy checkpoints without the
        people head); ``scores`` (all in-vocab tag probs); ``kept`` (emitted
        positives — softmax-group winners are picked by argmax, not sigmoid
        threshold, when typed groups are loaded); and ``groups``
        (``{group_name: predicted_tag_or_None}``, only when groups loaded).
        """
        tag_logits_row, rating_probs, people_probs = self._heads_forward(pil_img)
        tag_probs = tag_logits_row.sigmoid()
        kept_mask = (tag_probs >= self.thresholds_dev).cpu()
        tag_probs_cpu = tag_probs.cpu()
        scores = {
            self.tag_entries[i].name: float(tag_probs_cpu[i])
            for i in range(self.n_tags)
        }
        # sentinel slots ("<none:group>") stay in `scores` but are never emitted
        kept = {
            self.tag_entries[i].name: float(tag_probs_cpu[i])
            for i in range(self.n_tags)
            if kept_mask[i] and not tg.is_sentinel_name(self.tag_entries[i].name)
        }
        rating_idx = int(rating_probs.argmax().item())
        out: dict[str, object] = {
            "rating": self.ratings[rating_idx],
            "rating_scores": {
                r: float(rating_probs[i].cpu()) for i, r in enumerate(self.ratings)
            },
            "scores": scores,
            "kept": kept,
            # A shared reference, not a copy — treat as read-only.
            "thresholds": self.threshold_map,
        }
        if people_probs is not None and self.people_count_labels:
            people_idx = int(people_probs.argmax().item())
            out["people_count"] = self.people_count_labels[people_idx]
            out["people_count_scores"] = {
                lbl: float(people_probs[i].cpu())
                for i, lbl in enumerate(self.people_count_labels)
            }

        # Group-aware refinement: one argmax winner per applicable group.
        if self._group_lookup:
            kept_names = set(kept.keys())
            is_solo = is_solo_names(kept_names)
            group_preds: dict[str, str | None] = {}
            for name, info in self._group_lookup.items():
                mode = info["mode"]
                escape_fired = bool(kept_names & set(info["escape_names"]))
                if mode == "softmax_when_solo":
                    applicable = is_solo and not escape_fired
                else:  # "softmax"
                    applicable = not escape_fired
                if not applicable:
                    # Leave the group's tags as the per-tag threshold decided.
                    group_preds[name] = None
                    continue
                idx_t = info["tag_idx"]
                group_logits = tag_logits_row.index_select(0, idx_t)
                winner_local = int(group_logits.argmax().item())
                for t in info["tag_names"]:
                    kept.pop(t, None)
                if winner_local == info.get("sentinel_local"):
                    # "None of these" won — the group emits nothing.
                    group_preds[name] = None
                    continue
                winner_idx = int(idx_t[winner_local].item())
                # dbv4 was never CE-trained on our groups, so the winner must
                # also clear its own threshold ("at most one"): otherwise an
                # all-unsupported group emits its first member off a -30 logit.
                if (
                    self.backend_kind == "dbv4"
                    and tag_probs_cpu[winner_idx] < self.thresholds[winner_idx]
                ):
                    group_preds[name] = None
                    continue
                winner_name = self.tag_entries[winner_idx].name
                kept[winner_name] = float(tag_probs_cpu[winner_idx])
                group_preds[name] = winner_name
            out["kept"] = kept
            out["groups"] = group_preds

        dedupe_count_tags(kept)

        # People-count from the emitted count tags, bucketed by the rule the
        # vocab build labels by, so it agrees with the ``Ngirls`` tags the
        # position-clause pipeline reads. The sidecar distribution stays in
        # ``people_count_scores``.
        if self.backend_kind == "dbv4" and self.people_count_labels:
            people_idx = classify_people(n.replace(" ", "_") for n in kept)
            out["people_count"] = self.people_count_labels[people_idx]
            out["people_count_source"] = "count-tag-rule"

        # cap characters to the largest digit-prefixed girls-count in `kept`
        girl_caps = [
            int(m.group(1)) for name in kept if (m := GIRLS_COUNT_RE.match(name))
        ]
        if girl_caps:
            cap = max(girl_caps)
            char_scored = sorted(
                (
                    (kept[e.name], e.name)
                    for e in self.tag_entries
                    if e.category == "character" and e.name in kept
                ),
                reverse=True,
            )
            for _, name in char_scored[cap:]:
                kept.pop(name, None)
            out["kept"] = kept

        # drop characters below the floor; if that empties both character and
        # copyright, add "original" as a slot-filler (booru non-IP convention)
        dropped_any = False
        for e in self.tag_entries:
            if e.category != "character" or e.name not in kept:
                continue
            if kept[e.name] < self._character_floor:
                kept.pop(e.name, None)
                dropped_any = True
        if dropped_any and self._original_idx is not None:
            has_char = any(
                e.category == "character" and e.name in kept for e in self.tag_entries
            )
            has_copy = any(
                e.category == "copyright" and e.name in kept for e in self.tag_entries
            )
            if not has_char and not has_copy:
                kept["original"] = float(tag_probs_cpu[self._original_idx])

        # cap artist/copyright to top-1 by score (booru convention is one each)
        for cat in ("artist", "copyright"):
            cat_scored = sorted(
                (
                    (kept[e.name], e.name)
                    for e in self.tag_entries
                    if e.category == cat and e.name in kept
                ),
                reverse=True,
            )
            for _, name in cat_scored[1:]:
                kept.pop(name, None)

        out["kept"] = kept
        return out

    def predict_caption(self, pil_img: Image.Image, min_confidence: float = 0.0) -> str:
        """Image → canonical Anima caption string (rating + slotted tags).

        ``min_confidence`` (0-1) is an extra probability floor on top of the
        per-tag F1 thresholds; 0.0 (default) leaves them untouched. The
        rating slot is always emitted regardless of this floor.
        """
        out = self.predict(pil_img)
        kept = out["kept"]
        if min_confidence > 0.0:
            kept = {name: p for name, p in kept.items() if p >= min_confidence}
        kept_idxs = {
            self.tag_entries[i].index
            for i, name in enumerate([e.name for e in self.tag_entries])
            if name in kept
        }
        slotted: dict[str, list[str]] = {cat: [] for cat in SLOT_ORDER}
        slotted["rating"].append(out["rating"])
        for cat, entries in self._by_cat.items():
            for idx, _, name in entries:
                if idx in kept_idxs:
                    slotted.setdefault(cat, []).append(name)
        # Re-apply tag rules at emit time: the model can predict both `bra` and
        # `black bra`; apply_rules drops `bra` in that case.
        flat: list[str] = []
        for cat in SLOT_ORDER:
            flat.extend(slotted.get(cat, []))
        rating_held = flat[:1]
        rest = tr.apply_rules(flat[1:], self.rules)
        out_tags = rating_held + rest
        return ", ".join(underscore_to_space(t) for t in out_tags)
