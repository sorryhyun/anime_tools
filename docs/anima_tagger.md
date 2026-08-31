# Anima Tagger — multi-label tagger trained on Anima's caption distribution

A small classifier that maps an image to a comma-separated tag string in
exactly the format Anima's training-time T5 saw. Used as the case-1 ψ_src
provider for DirectEdit, and as a standalone captioner for LoRA dataset
prep / prompt scaffolding via the `comfyui-anima-tagger` ComfyUI node.

Status: **shipped**. The live checkpoint is **dbv4** (2026-08-27) —
`models/captioners/anima-tagger-dbv4/`: the external
`animetimm/caformer_b36.dbv4-full` backbone (GPL-3.0, gated, fetched under
your HF token — **never vendored**) projected onto the same 2,532-tag vocab /
4-class rating, plus a small sidecar head for copyright / OC characters /
renamed generals (see *dbv4 backend* under Inference). Our part of the
checkpoint (vocab, rules, groups, thresholds, sidecar) is auto-fetched from
the `dbv4/` subfolder of
[`sorryhyun/anima-tagger`](https://huggingface.co/sorryhyun/anima-tagger) when
missing; `v5/` is the last in-house PE dual-encoder head (val macro-F1
~0.236). `DEFAULT_TAGGER_DIR` / `TAGGER_HF_SUBFOLDER` in the torch-free
`anime_tools/tagger/dbv4_meta.py` are the single source of truth for both
(`tagger.py` re-exports them); the ComfyUI node and the download catalog
track them. Requires `timm`. `python -m anime_tools.downloads tagger
tagger_backbone` — the GUI's ⚙ Settings → **Models** rows run exactly this —
pre-fetches both halves: our checkpoint dir *and* the gated backbone.
Every runtime entry point (`caption-position`, `caption-autotag`, the GUI
autotag server, DirectEdit) goes through `ensure_tagger_checkpoint`, which
now also runs `ensure_tagger_backbone`: an offline hub-cache probe, then a
token fetch on miss — **before** SAM3 / the tagger are loaded, so a missing
token or unaccepted terms fails fast with the `hf auth login` + accept-terms
hint (a gated 401/403 is translated in `library/runtime/hf_download.py`)
instead of a raw traceback halfway through a daemon job. Set
`ANIMA_TAGGER_NO_AUTOFETCH=1` to refuse the fetch (offline hosts / CI). Repo
+ file set live in the torch-free `anime_tools/tagger/dbv4_meta.py`, shared
by the loader and by `anime_tools/downloads.py` (the catalog behind the GUI's
Models rows); the repo follows the installed checkpoint's
`config.json["dbv4"]["repo"]`.

> **Doc drift.** Sections below still describe the pre-v3 single-encoder +
> PE-LoRA stack, which no longer loads — the tagger is dual-encoder
> (PE-Core + PE-Spatial) and hard-routed. Treat the training-flag prose as
> historical; the config/paths above are current.

## Why this exists

DirectEdit's invert/edit primitive is robust to ψ_src corruption — even
shuffled or tag-dropped source captions reconstruct the source image at
~99% pixel fidelity. But edit *leverage* (whether ψ_tar = ψ_src + edit-tag
actually applies the change) collapses when ψ_src is structurally far
from Anima's training-time embedding manifold. Generic booru taggers
were bad enough at this to be the live blocker; this tagger replaces
that role with an Anima-distribution head.

## Architecture

```
PIL image → PIL LANCZOS-resize to PE-Core bucket size
         → IMAGE_TRANSFORMS (= [-1, 1])
         → frozen PE-Core-L14-336 → patch tokens [T, 1024]
         → mean-pool over T → feature [1024]
         ──────────────────────────────────────────────────  trunk (frozen)
         → LayerNorm + Linear(1024, 1024) + GELU + Dropout
         ────────────────────────────────────────────────── shared trunk_h
         ├→ Linear(1024, n_tags)       → tag_logits     ──── multi-label head
         └→ Linear(1024, n_ratings)    → rating_logits  ──── rating head

Per-tag F1-calibrated threshold sweep at the end of training picks the
inference threshold for each output dimension. Tags belonging to a
softmax group are excluded from the sweep — they're argmax-only at
inference.
```

Total trained params at default `n_tags=4937, d_hidden=1024`: **~6.1M**
(frozen path). The end-to-end `--pe_lora_rank > 0` path adds a low-rank
delta over the trailing PE-Core blocks; alpha/rank/layers configurable.

The shipped checkpoint runs the frozen path (`pe_lora: false`), trained
for 100 epochs at `lr=2e-4`, `batch_size=64`, `lambda_rating=0.1`.

### Rating band

Anima's rating band is 4-class — `safe, sensitive, nsfw, explicit`.
`anime_tools.tagger.tagger.RATINGS` fixes the class *order* (it's the
rating head's class index); `anime_tools.captions.taxonomy.CAPTION_RATINGS` is
the unordered set the caption-side consumers test against. Danbooru's own
literals are accepted as aliases and folded onto the band at vocab-build time
(`general`→`safe`, `questionable`→`nsfw`), so a raw booru caption still
classifies as a rating instead of falling through to the `general`
*category*.

The rename does not invalidate checkpoints: `AnimaTagger` reads
`vocab["ratings"]` from the checkpoint and `n_ratings` flows from the
manifest, so the shipped 3-class checkpoint keeps loading and predicting its
own labels. Rebuilding vocab against the 4-class band widens the head — that
is a retrain.

### Why a shared trunk for both heads

Rating prediction and tag prediction look at the same kinds of visual
content — lots of the rating signal is also expressible as tag
co-occurrence. A shared trunk gives the rating gradient a path into the
same representation the tag head reads from, at the cost of one extra
Linear at the head split. Empirically this is what gelcrawl's quality
classifier does too (`gelcrawl/classify.py`); we reuse that pattern.

### Why mean-pool over patch tokens

PE-Core's CLS token is contrastive-image-text trained — useful for
retrieval, not optimized for multi-label classification. Mean-pool over
the patch tokens gives a content-weighted summary; head capacity is
enough that the pooling choice doesn't bottleneck.

### Why a sqrt(neg/pos) BCE pos-weight

Anima's tag distribution has a heavy long-tail. Default BCE-with-logits
treats every tag-output identically, so common tags (1girl) dominate the
gradient. Inverse-frequency weights (`n_neg/n_pos`) over-correct and
explode rare-tag gradients. `sqrt(n_neg/n_pos)` is the standard middle
ground — softens the long-tail without overshoot.

## Code layout

`anime_tools/captions/` (inference + shared schema):

| File | Role |
|---|---|
| `anima_tagger.py` | `AnimaTagger` — public inference class. Exposes `predict`/`predict_caption`. Requires `config.json["backend"] == "dbv4"` (the legacy PE dual-encoder head was removed 2026-08-30 — curation split Phase 0; archived under `_archive/anima_tagger_training/pe_backend_removed_2026_08_30/`). Implements all post-prediction refinements (group argmax, character floor, original-fallback, girls-count cap, top-1 artist/copyright). |
| `dbv4_backend.py` | `animetimm/caformer_b36.dbv4-full` loader + `align_vocab` (the single vocab join point) + `SidecarHead` (our linear head over the backbone's hidden state). |
| `group_router.py` | `GroupRouter` + `compute_grouped_loss` — typed tag-group routing (softmax / softmax_when_solo / multilabel, sentinel + escape semantics). Promoted out of the archived trainer because the inference rule, calibrator and benches still resolve groups through it. |
| `feature_cache.py` | The dbv4 hidden-state cache: `dbv4_cache_path` / `dbv4_cache_stems` / `load_dbv4_cache` (the stem list rides in the safetensors metadata — a cache built for another manifest is misaligned row-for-row, so every reader checks it) + `multi_hot_from_manifest`. The archived PE token-cache path helpers went with their caches. |
| `data.py` | `TaggerCheckpoint.from_dir(path, require=…, backend=…)` — the one read of a checkpoint dir (`config.json` / `vocab.json` / `dataset.json`, the "run build_vocab first" exit, `idx_to_name`) — and `TaggerManifest`. |
| `tag_rules.py` | `tag_rules.yaml` loader/applier (replacements, always-remove, clothing dedup, `category_overrides`, `coverage_ignore`). |
| `tag_groups.py` | `tag_groups.yaml` loader; `TagGroup`/`TagGroups`/`ResolvedGroup`; modes `softmax`, `softmax_when_solo`, `multilabel`. |
| `taxonomy.py` / `readback.py` / `correction.py` | Danbooru category taxonomy + count-tag regex; Read-It-Back tag-adherence instrument; caption correction helpers. |

`scripts/anima_tagger/` (CLI — invoke as `python -m anime_tools.tagger.cli.main`):

| File | Role |
|---|---|
| `cli.py` | Argparse + 4-mode dispatcher (`build_vocab`, `predict`, `scan_role_markers`, `derive_groups`). Loads `.env` so `CAPTION_CORPUS_DIR` resolves before defaults are computed. |
| `vocab.py` | Caption discovery, tag categorization (rating literal → `@` artist → count regex → `category_overrides` → tag cache → `general` fallback), `min_freq` cut, train/val split, manifest build, group resolution against the kept vocab, coverage scan. |
| `derive_groups.py` | Taxonomy-driven tag-group candidates (folded into `build_vocab` by default). |
| `build_dbv4_ckpt.py` | Assemble the dbv4 checkpoint dir (`make tagger-dbv4`): copies vocab / rules / groups / split next to the backend descriptor. |
| `train_sidecar.py` | Sidecar linear-head trainer on cached dbv4 hidden states (`make daemon-run ARGS="anime_tools/tagger/cli/train_sidecar.py"`); calls `calibrate.calibrate_thresholds`. |
| `calibrate.py` | `calibrate_thresholds` — per-tag F1-optimal threshold sweep on val (skips softmax-group tags). Library function only; the old `--mode calibrate` driver is archived. |
| `eval_metrics.py` | Shared eval + `predict_with_inference_rule` (group argmax / count-tag rule). |
| `predict.py` / `autotag.py` / `autotag_server.py` | Single-image debug entry; CLI one-shot autotag; resident GUI worker. |
| `role_markers.py` | Read-only curator helper — scans the vocab + manifest for character-typed tags that behave like affiliation markers and emits a YAML stub ready to paste into `tag_rules.yaml`. |
| `constants.py` | `find_image_for_caption`, image extensions; re-exports the taxonomy count-tag regex. |

**Archived 2026-08-27** (`_archive/anima_tagger_training/`, untracked): the
PE dual-encoder training pipeline — `train_cached.py`, `train_common.py`
(minus `GroupRouter`), `caches.py` (`build_features`), `embed_tags.py`, the
`build_features` / `train` / `calibrate` / `embed_tags` CLI modes, `make
preprocess-tagger`, the v2–v6 + ab94 checkpoints, and the training tests.
The PE feature caches (`post_image_dataset/anima_tagger/tokens-*`,
`anima_tagger_stroked/`, 158 GB) were reclaimed; only `anima_tagger/dbv4/`
(the sidecar hidden-state cache, ≈170 MB) remains.

## Configuration via `.env`

External corpus paths are routed via `CAPTION_CORPUS_DIR`. Add to
`.env`:

```
CAPTION_CORPUS_DIR=/path/to/external/caption/corpus
```

Expected layout:

| Path | What it is | Consumer |
|---|---|---|
| `<corpus>/retrieved/{artist}/{stem}.{webp,jpg,png,jpeg}` | Source images, paired with `.txt` captions | Training input + label. ~12k images. |
| `<corpus>/retrieved/{artist}/{stem}.txt` | Booru-style caption per image, in Anima format (`rating, count, characters, copyrights, @artists, generals`) | Multi-hot label after `tag_rules` normalization. |
| `<corpus>/retrieved/.tag_cache.json` | `tag → integer type id` (0=general, 1=artist, 3=copyright, 4=character, 5=metadata, 6=deprecated) | Vocab categorization + canonical-emit-slot routing. |
| `<corpus>/tag_rules.yaml` | Replacements + always-remove + clothing dedup + `category_overrides` + `coverage_ignore` | Vocab-build time and inference safety net. Snapshotted into the checkpoint. |
| `<corpus>/tag_groups.yaml` | Typed groupings (`eye_color`, `hair_color`, `hair_length`, `rating`, `top_garment`, …) | Group routing during training + inference. Snapshotted into the checkpoint. |
| `<corpus>/selected/` (optional) | Curated subset (already deduped) | Additional caption source. |

`image_dataset/` (Anima's training set) is also scanned by default.

`CAPTION_CORPUS_DIR` is **not committed** — it's per-user. The trained
checkpoint snapshots `rules.yaml` + `groups.yaml` so inference has zero
runtime dependency on the corpus dir.

## Training pipeline

Since 2026-08-27 the shipped tagger is **not trained end-to-end here**: the
backbone is the external `caformer_b36.dbv4-full`, and the only trained
piece is the sidecar linear head. The vocab build is still ours.

```bash
# 1. Build the vocabulary + train/val split + per-stem manifest +
#    resolved typed groups (derive_groups folded in).
make tagger ARGS="--min_freq 5"          # == --mode build_vocab

# 2. Assemble the dbv4 checkpoint dir (vocab / rules / groups / split +
#    backend descriptor).
make tagger-dbv4

# 3. Cache dbv4 hidden states + train / calibrate the sidecar head.
make daemon-run ARGS="anime_tools/tagger/cli/train_sidecar.py --ckpt_dir models/captioners/anima-tagger-dbv4"

# 4. Single-image sanity check.
make test-tagger ARGS="--image foo.png --show_scores"

# 5. Curator helper — find character tags that behave like affiliation markers.
python -m anime_tools.tagger.cli.main --mode scan_role_markers --out_yaml stub.yaml
```

All artifacts go to `--out_dir` (default `models/captioners/anima-tagger-dbv4/`):

```
models/captioners/anima-tagger-dbv4/
├── vocab.json              # tag list + category + median emit pos + groups
├── rules.yaml              # snapshot of tag_rules.yaml at vocab-build time
├── groups.yaml             # snapshot of tag_groups.yaml
├── dataset.json            # per-stem (image_path, multi_hot, rating) manifest
├── config.json             # backend="dbv4" descriptor (repo / arch / vocab map)
├── sidecar.safetensors     # our linear head (the only trained weights shipped)
└── thresholds.safetensors  # per-tag F1-calibrated thresholds (sidecar tags)
```

The dbv4 weights are fetched at runtime under the user's HF token (GPL-3.0,
gated) and never bundled; the checkpoint dir holds only our files, so it
stays moveable across machines. The gate is *auto-approve*: `hf auth login`
(or the GUI Settings dialog's token field) plus one click on
[the repo page](https://huggingface.co/animetimm/caformer_b36.dbv4-full) is
all it takes, and `python -m anime_tools.downloads tagger_backbone` then
pulls them eagerly. They
land in the **HF hub cache**, not under `models/` — that is where
`Dbv4Backend._load_model` looks. The dbv4 hidden-state cache lives at
`post_image_dataset/anima_tagger/dbv4/<arch>_hidden.safetensors`
(`feature_cache.py`), read by `train_sidecar.py` and
`bench/tagger_external/calibration_check.py`.

### Legacy PE dual-encoder path (archived)

The frozen PE-Core + PE-Spatial hard-routed head (`train_cached.py`,
`build_features`, stroke augmentation, label-embed / spatial-headroom levers)
is archived at `_archive/anima_tagger_training/` together with its
checkpoints (v2 … v6-spatialL, ab94 ablations) and the pre-archive version of
this doc (`_archive/anima_tagger_training/docs/anima_tagger.pre_archive.md`). `AnimaTagger` still **loads**
those checkpoints (`anima_tagger_model.py` / `anima_tagger_data.py` are
load-only), and `bench/tagger_external` defaults to the archived v5 for the
ours-vs-external comparison. Why it was retired: the external dbv4 backbone
crushes it (mAP 0.72 vs 0.30, position crops hair 8–10/10 vs 3/10), and the
spatial-headroom line was **superseded**, not refuted — its premise (the PE
trunk is the ceiling) is confirmed from the other direction.

### Group routing (`GroupRouter`)

`tag_groups.yaml` declares typed groups with one of three modes:

* **`softmax_when_solo`** — K-way CE over the group's logits when the
  sample is single-subject (`solo`/`1girl`/`1boy`/`1other` fires AND no
  multi-count tag fires) AND no `escape:` tag fires; falls back to BCE
  per-tag otherwise. Used for groups that are mutually exclusive
  on a single subject (eye color, hair color, hair length, primary
  garment) but irrelevant when an explicit escape applies (e.g.
  `heterochromia` for eye_color, `multicolored hair` for hair_color).
* **`softmax`** — always K-way CE (modulo `escape:`). Used for genuinely
  exclusive groups like rating.
* **`multilabel`** — left in BCE; the group only exists for
  introspection / UI grouping.

`anime_tools/captions/group_router.py` holds the router and
`compute_grouped_loss` (BCE on every (sample, tag), masking the positions CE
supervises so each cell has exactly one term). At inference the same router
drives group argmax via `eval_metrics.predict_with_inference_rule`; the
sidecar trainer does not use the grouped loss (its tags are plain BCE), but
the semantics stay pinned by `tests/test_tagger_sentinel_groups.py` and
`tests/test_grouped_loss_negweight.py`.

### Calibration

`calibrate.calibrate_thresholds` sweeps thresholds in `[0.05, 0.95]` step
`0.05` per tag and picks the F1-maximizing one on val. Tags with fewer than
`min_support` val positives, zero achievable F1, or membership in a softmax
group keep `default=0.5` (softmax-group tags are routed by argmax at
inference). Tag-block size of 256 caps memory. dbv4-native tags keep the
card thresholds (head-tier ECE 0.019 — do not recalibrate on 791 images);
only sidecar tags are swept.

### Role-marker scan

`role_markers.py` is a read-only curator helper. It reads `vocab.json`
+ `dataset.json` and ranks every `category=='character'` tag by its
conditional co-occurrence with another character tag on **solo**
training samples (using the same `solo`/`1girl`/`1boy`/`1other` predicate
the trainer applies). Each candidate is auto-bucketed:

* **A_costume** — candidate shares a name prefix with a top partner →
  variant of an existing base. Curate via `tag_rules.yaml` `dedup:`.
* **D_role** — broad partner pool (≥ `--min_role_partners` distinct
  partners) → affiliation marker mistyped as character (`sensei (blue
  archive)`, `producer (idolmaster)`, `doctor (arknights)`). Curate via
  `tag_rules.yaml` `remove:`.
* **C_pair** — narrow partner pool (top-1 partner ≥
  `--pair_dominance` of co-occurrences) → genuine couple/sibling pair.
  Leave alone.
* **B_review** — everything else; eyeball.

`--out_yaml stub.yaml` writes a YAML stub split into pasteable sections
(A as dedup blocks, D under `remove:`, B/C as commented hints). No files
in the checkpoint dir are mutated.

## Inference

```python
from anime_tools.captions import AnimaTagger
from PIL import Image

tagger = AnimaTagger("models/captioners/anima-tagger-dbv4")  # default
caption = tagger.predict_caption(Image.open("foo.png"))
# → "sensitive, 1girl, hatsune miku, vocaloid, @some_artist, blue eyes, ..."

debug = tagger.predict(Image.open("foo.png"))
# → {"rating": "...", "rating_scores": {...}, "scores": {...},
#    "kept": {...}, "groups": {"eye_color": "blue eyes", ...}}
```

`AnimaTagger.predict`:

1. PIL → bucket-resize → IMAGE_TRANSFORMS → frozen PE-Core (+ optional
   PE-LoRA delta loaded from `pe_lora.safetensors` when `config.pe_lora`
   is true) → mean-pool → trunk → tag_logits + rating_logits.
2. `sigmoid(tag_logits) ≥ thresholds` → `kept`; `argmax(rating_logits)`
   → rating.
3. **Group-aware refinement.** For each loaded `softmax`/`softmax_when_solo`
   group, when the gating predicate applies (single-subject for
   `softmax_when_solo`, always for `softmax`, both modulo escape tags),
   replace any sigmoid-admitted members with the single argmax winner
   over the group's logits.
4. **Girls-count cap.** When `kept` contains digit-prefixed `Ngirls`, trim
   character predictions to the top-`max(N)` by score — caps the
   independent-sigmoid leakage on gender-ambiguous art.
5. **Character floor + original fallback.** Any character below
   `character_floor` (default `0.5`, sits above some F1 thresholds as
   low as `0.05` for noisy long-tail characters) is dropped. When that
   empties the character slot AND no copyright tag survives, add
   `original` (booru convention for non-IP work) so the caption still
   has a slot-filling copyright.
6. **Top-1 artist + top-1 copyright.** Independent sigmoid heads can
   admit several borderline tags; collapse to the highest-scoring one
   (booru convention is one artist / one copyright per work).

### dbv4 backend (2026-08-27)

`config.json["backend"] = "dbv4"` swaps the whole PE stack for an external
danbooru tagger (`animetimm/caformer_b36.dbv4-full` by default — 134 M
params, 384², **GPL-3.0 + gated: fetched under the user's HF token, never
vendored**) projected onto our vocab. Steps 2–6 above are unchanged; only
step 1 differs (`anime_tools/tagger/dbv4_backend.py`):

- `align_vocab` joins dbv4's snake_case names onto our space-separated vocab
  (rules.yaml renames recovered). On v5's vocab 2,182 / 2,532 tags match;
  unmatched = 118 copyright + 36 dataset OCs + 84 renamed generals + 92
  `@artist` (+ deprecated/meta). Unsupported tags sit at logit −30 and a
  pure-`softmax` group only emits a winner that clears its own threshold
  ("at most one" — dbv4 was never CE-trained on our groups).
- dbv4's four rating sigmoids (`general`/`questionable` → `safe`/`nsfw`) are
  normalised to a distribution for `rating_scores`.
- A **sidecar** linear head (`sidecar.safetensors` + `sidecar.json`, trained
  by `anime_tools/tagger/cli/train_sidecar.py` on the backend's 3072-d MLP-head
  hidden feature) emits what dbv4 cannot: copyright, OC characters, renamed
  generals, and the 8-way people-count. **`@artist` is deliberately not
  covered** — artist attribution stopped being a tagger goal on 2026-08-27.
  `people_count` is always the count-tag rule on dbv4
  (`taxonomy.classify_people`, `people_count_source="count-tag-rule"`) — on
  v5's val split it scores 0.943 vs the sidecar head's 0.929 (v5 head 0.885);
  the sidecar softmax is exposed as `people_count_scores` only. Sidecar val
  (2026-08-27): copyright macro-F1 0.815 / mAP 0.92 (v5 0.638), OC characters
  0.889 / 0.98, renamed generals 0.40 / 0.61.
- Thresholds: dbv4 card `best_threshold` for matched tags, F1-calibrated on
  our val split for sidecar rows, `1.01` (never fires) for the rest.

Build with `make tagger-dbv4` (→ `models/captioners/anima-tagger-dbv4/`),
train the sidecar with `make daemon-run
ARGS="anime_tools/tagger/cli/train_sidecar.py"`, then pass the dir anywhere a
`--tagger_dir` is accepted. Bench evidence: `bench/tagger_external/` and
`docs/proposal/tagger_caformer_backend.md`.

`predict_caption` then slots tags by canonical category order
(`rating, count, character, copyright, artist, general`), within-slot by
median emit position from the training corpus, re-applies `tag_rules` as
a safety net (the dedup map already fired during training-data
normalization, but the model could in principle predict both `bra` and
`black bra`), replaces underscores with spaces, and joins with `, `.

## Wired-up touchpoints

### CLI driver

`scripts/experimental_tasks/inference.py::cmd_test_directedit` runs the
Anima Tagger on the source image to seed `--prompt_src`:

```bash
make exp-test-directedit PROMPT='glasses'
```

Requires `models/captioners/anima-tagger-dbv4/` (auto-downloaded on first
use; `make tagger-dbv4` rebuilds it locally). The driver exits with a clear
error if the checkpoint is missing.

`scripts/edit.py` itself doesn't tag — it takes `--prompt_src` directly.
Tagging only happens in the make-target driver (CLI) or the ComfyUI node
(see below).

### Batch auto-tagging (`make caption-autotag`)

`anime_tools/stages/cli/autotag_captions.py` (over `stages/autotag.py`) is the
dataset-wide counterpart to the Dataset tab's per-image button: it walks the
resized tree, tags each image, and writes the `.txt` sidecar into the caption
**master** under `image_dataset/`. `--mode missing` (default) is the only
non-destructive mode; `merge` appends only novel tags and round-trips position
clauses verbatim; `overwrite` replaces. Dry run by default, `--apply` writes,
and any apply must be followed by `make preprocess-te`.

#### `--from_report` — apply a dry run without re-loading the tagger

The dry run's `report.json` already records, per image, the destination
(`rows[].caption_path`) and the exact text (`rows[].proposed`), so the apply
pass has nothing left to compute:

```bash
make caption-autotag                                   # the model pass, once
make caption-autotag ARGS="--apply --from_report post_image_dataset/captions/autotag/report.json"
make preprocess-te                                     # still REQUIRED
```

The second line **loads no model** — it does not even import `torch`
(`tests/test_stage_replay.py` pins that in a subprocess). The same flag exists
on `caption-position` and `audit-multiview`; the shared implementation is
`anime_tools/stages/replay.py`, and
[`position_captions.md`](position_captions.md) carries the full staleness table.
In short:

- **Refused** if the report's recorded `src`/`dst` differ from this run's, if it
  records neither, or if its own `apply` flag is already true (its `existing`
  text describes the pre-apply world, so every row would read as drifted).
- **Skipped and counted**, never overwritten, if the caption on disk no longer
  matches the row's `existing` (`skip:drifted`) — the guard that makes a stale
  report safe to replay over hand edits. A file that already holds the proposal
  is `skip:already-applied`, so replays are idempotent.
- `--path_pattern` still filters, and without `--apply` it is a re-play dry run
  that prints what would be written.

The replay writes **`apply_report.json`** (never over the `report.json` it read),
shaped like the stage's own — top-level metadata plus `stats` and `rows` — with
`from_report`, per-row `{image, caption_path, before, after, status}`, and a
top-level **`written[]` of the relative image paths actually written**, which is
what a UI reads to reload exactly the affected dataset items.

### ComfyUI nodes (`comfyui/anima_tagger/`)

Two nodes share the `ANIMA_TAGGER` socket type:

| Node | Inputs | Outputs |
|------|--------|---------|
| `AnimaTaggerLoader` | `tagger_dir` (STRING) | `tagger` (ANIMA_TAGGER) |
| `AnimaTaggerCaption` | `tagger` (ANIMA_TAGGER), `image` (IMAGE) | `caption` (STRING) |

The node ships in this repo under [`comfyui/anima_tagger/`](../comfyui/anima_tagger/)
(moved from the standalone `ComfyUI-Anima-Tagger` repo, 2026-08-30). It vendors
nothing — it imports `anime_tools.tagger.AnimaTagger` from the installed package —
so install is `pip install ./anime_tools` + link/copy the directory into ComfyUI's
`custom_nodes/`. See its README for the exact commands.

`AnimaTaggerCaption` outputs a STRING that drops into any text input —
DirectEdit's `ANIMA_TAGGER` socket, `CLIPTextEncode` for prompt
scaffolding, or `Save Text File` for LoRA dataset pre-fill.

## Known limitations

1. **Rating-class imbalance.** Train-corpus rating mix is ~67% explicit
   / ~32% sensitive / ~0.6% safe. Class-weighted CE compensates
   partially. If `safe`-rating accuracy matters downstream, oversample
   at training time.
2. **Per-tag positives are thin for the long tail.** At `min_freq=5`
   each long-tail tag has 5–20 positives; calibrated thresholds for those
   tags are noisier than for high-frequency ones. `--min_freq 10` is a
   knob to revisit if F1 disappoints.
3. **No bench harness yet.** An `anima_tagger` bench per the standard
   envelope (cf. `bench/_common.py::write_result`) is the next thing to
   add — should report F1 on a held-out set plus a downstream
   "edit-success-rate" metric on a small DirectEdit set.
4. **Long-tail characters benefit from `character_floor`.** Some F1
   thresholds settle as low as `0.05` for noisy long-tail characters;
   the post-prediction floor (default `0.5`) is what stops borderline
   guesses from leaking into ψ_src on stylized / gender-ambiguous art.
   Lowering the floor recovers recall at the cost of precision.

## Open design questions

1. **DINOv3 trunk swap.** gelcrawl's `classify.py` uses DINOv3 ViT-L/16@224
   and works well in this domain. If F1 saturates and we suspect the
   trunk is the limit, swap encoders — `--encoder` flag already plumbs
   through the loader registry.
2. **Embedding output instead of tag string.** `predict_caption` emits a
   string that gets re-tokenized by T5. We could add a head producing
   `[K, D_t5]` continuous tokens directly — but that's the img2emb design
   (`_archive/proposals/img2emb_plan.md`) and hits the same structural
   challenges. Stick with tag-string output for now.
