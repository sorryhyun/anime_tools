"""AnimaTagger — multi-label tagger trained on the Anima caption distribution.

The ψ_src provider for DirectEdit. Public surface: ``predict``,
``predict_caption``.

Checkpoint layout (produced by ``python -m anime_tools.tagger.cli.main``):

::

    ckpt_dir/
      config.json              # model config + training metadata
      sidecar.safetensors      # optional linear sidecar head (dbv4 lacks)
      thresholds.safetensors   # per-tag F1-optimal thresholds
      vocab.json               # tag list with category + median_pos + group info
      rules.yaml               # caption-normalization rules snapshot
      groups.yaml              # tag-group taxonomy (optional)

When ``groups.yaml`` is present, prediction is group-aware: ``softmax`` and
``softmax_when_solo`` groups emit exactly one tag per group (argmax over
group logits) instead of the sigmoid threshold. Ungrouped/multi-label tags
use the standard threshold path.

One backend, ``config.json["backend"] == "dbv4"`` (the legacy in-house
``"pe"`` dual-encoder head was removed 2026-08-30 — curation split Phase 0;
its trainer + data builders live in ``_archive/anima_tagger_training/``):

* ``"dbv4"`` — an off-the-shelf danbooru tagger (``animetimm/*.dbv4-full``,
  GPL-3.0 + gated, **never vendored** — fetched under the user's HF token)
  projected onto our vocab (``anime_tools/tagger/dbv4_backend.py``), plus an
  optional ``sidecar.safetensors`` linear head for what dbv4 lacks
  (copyright / dataset-only characters / people-count). ``model.safetensors``
  is absent; ``config.json["dbv4"]`` carries ``repo`` / ``arch`` / ``img_size``.
  Build one with ``python -m anime_tools.tagger.cli.build_dbv4_ckpt``.

Everything after the score vector — thresholds, softmax groups, count dedupe,
character floor, top-1 copyright, slot order — is backend-agnostic
post-processing of ``{tag: prob}``.

Captions are emitted in Anima's canonical slot order:
``rating, count_tags, characters, copyrights, @artists, generals``, with
underscores replaced by spaces (matching Anima's training-time T5 input).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file as st_load

from anime_tools._device import resolve_device
from anime_tools.captions import tag_groups as tg
from anime_tools.captions import tag_rules as tr
from anime_tools.captions.taxonomy import classify_people, is_solo_names
from anime_tools.captions.vocab_io import load_vocab
from anime_tools.tagger.dbv4_backend import (
    UNSUPPORTED_LOGIT,
    Dbv4Backend,
    SidecarHead,
    align_vocab,
    probs_to_logits,
    rename_recovery_from_rules,
)

# Checkpoint layout / repo facts live in the torch-free dbv4_meta so the
# download catalog and the ComfyUI loader can read them without torch; they are
# re-exported here because every caller imports them from this module.
from anime_tools.tagger.dbv4_meta import (
    DBV4_OPTIONAL_FILES,
    DBV4_REQUIRED_FILES,
    DEFAULT_TAGGER_DIR,
    TAGGER_HF_REPO,
    TAGGER_HF_SUBFOLDER,
    TAGGER_OPTIONAL_FILES,
    TAGGER_REQUIRED_FILES,
)

logger = logging.getLogger(__name__)


def ensure_tagger_checkpoint(
    ckpt_dir: str | Path,
    repo: str = TAGGER_HF_REPO,
    subfolder: str = TAGGER_HF_SUBFOLDER,
    *,
    backbone: bool = True,
) -> Path:
    """Fetch the tagger checkpoint into ``ckpt_dir`` if any required file is missing.

    Files are flattened into ``ckpt_dir`` regardless of source layout so the
    loader's directory contract stays uniform. Optional files (thresholds /
    groups) are best-effort — a 404 just means the checkpoint doesn't ship it.

    With ``backbone=True`` (default) a dbv4 checkpoint also runs
    :func:`ensure_tagger_backbone`, so the gated upstream weights are verified
    / fetched **here**, before any caller loads SAM3 or builds the tagger —
    not lazily on the first crop halfway through a daemon job.
    """
    ckpt_dir = Path(ckpt_dir)
    if all((ckpt_dir / f).exists() for f in TAGGER_REQUIRED_FILES):
        return ckpt_dir
    if all((ckpt_dir / f).exists() for f in DBV4_REQUIRED_FILES) and _is_dbv4_dir(
        ckpt_dir
    ):
        if backbone:
            ensure_tagger_backbone(ckpt_dir)
        return ckpt_dir
    from huggingface_hub.utils import EntryNotFoundError

    from anime_tools._hf import hf_download

    logger.info(
        "AnimaTagger: %s missing required files — fetching %s%s (one-time).",
        ckpt_dir,
        repo,
        f"/{subfolder}" if subfolder else "",
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_flat(fname: str) -> Path:
        repo_path = f"{subfolder}/{fname}" if subfolder else fname
        downloaded = Path(
            hf_download(
                what="AnimaTagger weights",
                repo_id=repo,
                filename=repo_path,
                local_dir=str(ckpt_dir),
            )
        )
        dest = ckpt_dir / fname
        if downloaded.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), str(dest))
        return dest

    # config.json first: it decides whether this is a PE checkpoint (needs
    # model.safetensors) or a dbv4 one (no weights of ours — sidecar optional).
    _fetch_flat("config.json")
    if _is_dbv4_dir(ckpt_dir):
        required, optional = DBV4_REQUIRED_FILES, DBV4_OPTIONAL_FILES
    else:
        required, optional = TAGGER_REQUIRED_FILES, TAGGER_OPTIONAL_FILES
    for fname in required:
        if fname != "config.json":
            _fetch_flat(fname)
    for fname in optional:
        try:
            _fetch_flat(fname)
        except EntryNotFoundError:
            logger.debug("optional tagger file %s not present on %s", fname, repo)
    if backbone and _is_dbv4_dir(ckpt_dir):
        ensure_tagger_backbone(ckpt_dir)
    return ckpt_dir


def ensure_tagger_backbone(ckpt_dir: str | Path) -> str:
    """Preflight the gated dbv4 backbone for the checkpoint at ``ckpt_dir``.

    Our half of the tagger is public; the backbone
    (``config.json["dbv4"]["repo"]``, GPL-3.0) is gated and only ever lands in
    the HF hub cache under the user's own token. Returns the repo id. Order:

    1. offline cache probe (``hf_file_cached``) — no network when installed;
    2. otherwise fetch every backbone file through ``hf_download``, which turns
       a gated 401/403 into a ``FileNotFoundError`` naming the accept-terms
       recovery instead of a raw hub traceback.

    Non-dbv4 (legacy PE) checkpoints return their default repo without touching
    anything. Set ``ANIMA_TAGGER_NO_AUTOFETCH=1`` to fail instead of fetching
    (offline hosts, CI).
    """
    from anime_tools.tagger.dbv4_meta import (
        DBV4_BACKBONE_FILES,
        backbone_cached,
        backbone_repo_for,
        gated_hint,
    )

    ckpt_dir = Path(ckpt_dir)
    repo = backbone_repo_for(ckpt_dir)
    if not _is_dbv4_dir(ckpt_dir) or backbone_cached(repo):
        return repo
    if os.environ.get("ANIMA_TAGGER_NO_AUTOFETCH"):
        raise FileNotFoundError(
            f"AnimaTagger backbone {repo} is not in the HF cache and "
            f"ANIMA_TAGGER_NO_AUTOFETCH is set. Run "
            f"`python -m anime_tools.downloads tagger_backbone` "
            f"({gated_hint(repo)})."
        )
    from anime_tools._hf import hf_download

    logger.info(
        "AnimaTagger: backbone %s not cached — fetching under your HF token "
        "(gated, GPL-3.0; one-time).",
        repo,
    )
    for fname in DBV4_BACKBONE_FILES:
        hf_download(
            what=f"AnimaTagger backbone ({repo})",
            hint=gated_hint(repo),
            repo_id=repo,
            filename=fname,
        )
    return repo


def _is_dbv4_dir(ckpt_dir: Path) -> bool:
    try:
        with open(ckpt_dir / "config.json", encoding="utf-8") as f:
            return json.load(f).get("backend") == "dbv4"
    except (OSError, ValueError):
        return False


# Digit-prefixed girls counts ("1girl"…"6+girls"). "multiple girls" is
# intentionally not matched — no exact count, so leave the character head alone.
_GIRLS_COUNT_RE = re.compile(r"^(\d+)\+?girls?$")

# Exact people-count families ("3girls" / "2boys" / "1other", open "6+girls"
# included). "multiple_girls"/"multiple_boys" are booru implication co-tags,
# not exact counts — they legitimately ride alongside a digit count and are
# deliberately not matched here.
_EXACT_COUNT_RES = tuple(
    re.compile(rf"^\d+\+?{noun}s?$") for noun in ("girl", "boy", "other")
)


def dedupe_count_tags(kept: dict[str, float]) -> None:
    """Drop all but the highest-scoring exact count per family, in place.

    Training captions never carry two exact counts of one family
    (``3girls`` + ``4girls``), but the sigmoid head has no mutual exclusion,
    so a near-threshold image can clear both — which also inflates the
    girls-count-driven character cap and trips the position-clause pipeline's
    count-mismatch gate downstream.
    """
    for cre in _EXACT_COUNT_RES:
        hits = sorted((n for n in kept if cre.match(n)), key=lambda n: -kept[n])
        for name in hits[1:]:
            kept.pop(name)


# Canonical caption-format slot order (matches Anima training captions).
SLOT_ORDER: tuple[str, ...] = (
    "rating",
    "count",
    "character",
    "copyright",
    "artist",
    "general",
)

# Booru tag-type integer → category name. Written into vocab.json and read back
# at inference, so changes here invalidate existing checkpoints.
TAG_TYPE_NAMES: dict[int, str] = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "metadata",
    6: "deprecated",
}

# Anima's 4-class rating set, in canonical class-index order (least -> most
# restrictive; see anime_tools.captions.taxonomy.CAPTION_RATINGS for the legacy
# booru aliases). Do not reorder without rebuilding vocab.json/dataset.json —
# but existing checkpoints are unaffected since AnimaTagger reads ratings/
# n_ratings from the checkpoint itself, so a 3-class checkpoint still works.
RATINGS: tuple[str, ...] = ("safe", "sensitive", "nsfw", "explicit")

# 8-class people-count bucket from parsed count tags
# (``anime_tools.tagger.cli.constants.classify_people``); dedicated softmax head.
# Order is the canonical class index — do not reorder without rebuilding vocab.
PEOPLE_COUNT_LABELS: tuple[str, ...] = (
    "no_people",  # 0 — no count tag at all
    "1girl",  # 1 — 1girl, no boy
    "1girl_1boy",  # 2 — exactly one of each
    "2girls",  # 3 — 2girls, no boy
    "2girls_1boy",  # 4 — 2girls + 1boy
    "2boys_1girl",  # 5 — 2boys + 1girl  (mirror of 2girls_1boy)
    "1boy",  # 6 — 1boy, no girl (solo male)
    "multi",  # 7 — 3+girls / 3+boys / 2g-2b+ / multiple_* / Nothers
)


@dataclass
class _TagEntry:
    name: str
    index: int
    category: str
    median_pos: float


def _underscore_to_space(s: str) -> str:
    """Anima caption format: tags with spaces, not underscores.

    The cache key uses underscores; the canonical caption uses spaces.
    Apply at emit time (not vocab-build) so tag indexing stays stable.
    """
    return s.replace("_", " ")


def _fix_artist_category(category: str, name: str) -> str:
    """Retype legacy mis-categorized "artist" entries shipped in vocab.json.

    Older vocab builds typed any ``@``-prefixed tag as ``artist``, sweeping
    up booru emoticons like ``@_@``. The corrected rule requires ``@``
    followed by non-whitespace; patched here so existing checkpoints don't
    need rebuilding.
    """
    if category != "artist":
        return category
    if len(name) >= 2 and name[0] == "@" and not name[1].isspace():
        return "artist"
    return "general"


def _load_thresholds(path: Path, n_tags: int, default: float = 0.5) -> torch.Tensor:
    """Load per-tag thresholds; missing → uniform default."""
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
        dtype: torch.dtype = torch.bfloat16,
        pe_ckpt: str | Path | None = None,
        character_floor: float = 0.5,
        pe_lora_path: str | Path | None = None,
        pe_lora_disabled: bool = False,
        pe_aux_ckpt: str | Path | None = None,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        self.device = torch.device(resolve_device(device))
        self.dtype = dtype
        # Accepted for call-site compat (ComfyUI workflows / old scripts) but
        # no-ops: the PE dual-encoder backend and PE-LoRA are gone.
        del pe_ckpt, pe_aux_ckpt, pe_lora_path, pe_lora_disabled
        # Absolute confidence floor for characters, above the per-tag F1
        # threshold (some F1 thresholds are ~0.05, too permissive on its own).
        self._character_floor = float(character_floor)

        with open(self.ckpt_dir / "config.json", encoding="utf-8") as f:
            cfg_d = json.load(f)
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
        self.tag_entries: list[_TagEntry] = [
            _TagEntry(
                name=t["name"],
                index=int(t["index"]),
                category=_fix_artist_category(str(t["category"]), t["name"]),
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
        # Per-tag keep thresholds by name, in ``tag_entries`` order (the same
        # alignment ``predict`` keys ``scores``/``kept`` by). Built once and
        # attached to every ``predict`` output so a downstream consumer can
        # reason about how far a score fell short of the tagger's own decision
        # (the position-clause bag relaxation reads it).
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

    # ------------------------------------------------------------------ #
    # Backend construction
    # ------------------------------------------------------------------ #

    def _init_dbv4_backend(self, cfg_d: dict) -> None:
        """External dbv4 tagger projected onto our vocab (+ optional sidecar)."""
        from types import SimpleNamespace

        d = dict(cfg_d.get("dbv4") or {})
        self._dbv4 = Dbv4Backend(
            repo=d.get("repo", "animetimm/caformer_b36.dbv4-full"),
            arch=d.get("arch", "caformer_b36"),
            img_size=int(d.get("img_size", 384)),
            device=self.device,
            dtype=self.dtype,
            revision=d.get("revision"),
        )
        # readback / bench read ``cfg.n_tags``; pool kinds are PE-only concepts.
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
        """Head pass: dbv4 backend + optional sidecar.

        Returns ``(tag_logits[n_tags], rating_probs[n_ratings], people_probs
        [n_people] | None)`` on ``self.device``. The tag logits are the
        projected sigmoid probs mapped back through logit(); tags the backend
        cannot emit sit at ``UNSUPPORTED_LOGIT`` (never clear any threshold,
        never win a group argmax).
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
        """Image → raw ``[n_tags]`` tag logits (float32, cpu). The readback path."""
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

        # Group-aware refinement: replace softmax-group sigmoid-threshold output
        # with one argmax winner per applicable group.
        if self._group_lookup:
            kept_names = set(kept.keys())
            # ``softmax_when_solo`` applies only to single-subject images; the
            # names in ``kept`` are vocab names, so the shared name-side
            # predicate answers exactly what the trainer's index-side one does.
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
                # drop sigmoid-admitted tags, re-add the argmax winner w/ its sigmoid prob
                for t in info["tag_names"]:
                    kept.pop(t, None)
                if winner_local == info.get("sentinel_local"):
                    # "None of these" won — the group emits nothing.
                    group_preds[name] = None
                    continue
                winner_idx = int(idx_t[winner_local].item())
                # dbv4 was never CE-trained on our groups, so "exactly one"
                # is not a calibrated contract there: the winner must also
                # clear its own threshold ("at most one"). This is what stops
                # an all-unsupported group (e.g. @artist) from emitting its
                # first member off a -30 logit.
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

        # dbv4: people-count comes from the emitted count tags, bucketed by the
        # same rule the vocab build labels people-count by. Measured on v5's
        # val split (2026-08-27) the rule beats the sidecar softmax head
        # (0.943 vs 0.929) and is consistent with the ``Ngirls`` tags the
        # position-clause pipeline reads; the sidecar's distribution stays
        # available as ``people_count_scores``.
        if self.backend_kind == "dbv4" and self.people_count_labels:
            people_idx = classify_people(n.replace(" ", "_") for n in kept)
            out["people_count"] = self.people_count_labels[people_idx]
            out["people_count_source"] = "count-tag-rule"

        # cap characters to the largest digit-prefixed girls-count in `kept`
        # (trim borderline sigmoid admits to top-N by score, N = parsed count)
        girl_caps = [
            int(m.group(1)) for name in kept if (m := _GIRLS_COUNT_RE.match(name))
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

        # (The former OC-suffix rule — keep a character under `original`/meta
        # copyright only when its parens-suffix matched the surviving @artist —
        # was PE-only: dbv4 never emits @artist, so there is nothing to compare
        # against and the rule dropped real OCs. Removed with the PE backend.)

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
        # Re-apply tag rules at emit time as a safety net: the model can predict
        # both `bra` and `black bra`; apply_rules drops `bra` in that case.
        flat: list[str] = []
        for cat in SLOT_ORDER:
            flat.extend(slotted.get(cat, []))
        rating_held = flat[:1]
        rest = tr.apply_rules(flat[1:], self.rules)
        out_tags = rating_held + rest
        return ", ".join(_underscore_to_space(t) for t in out_tags)
