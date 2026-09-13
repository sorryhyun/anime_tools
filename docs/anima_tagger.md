# Anima Tagger — multi-label tagger trained on Anima's caption distribution

A multi-label classifier that maps an image to a comma-separated tag string in
Anima's caption format (`rating, count, characters, copyrights, @artists,
generals`). It is what this package's own caption stages tag with: position
captions, batch autotag, and the GUI's autotag server.

The live checkpoint is `models/captioners/anima-tagger-dbv4/`: the external
`animetimm/caformer_b36.dbv4-full` backbone (134 M params, 384², GPL-3.0,
gated — fetched under your HF token, never vendored) projected onto a
2,532-tag vocab / 4-class rating, plus a small sidecar head for copyright /
characters dbv4 lacks / renamed generals. `config.json["backend"] == "dbv4"` is the only
backend `AnimaTagger` loads. Requires `timm`.

Our half of the checkpoint (vocab, rules, groups, thresholds, sidecar) is
auto-fetched from the root of
[`sorryhyun/anima-tagger`](https://huggingface.co/sorryhyun/anima-tagger) when
missing; the backbone repo follows the checkpoint's
`config.json["dbv4"]["repo"]`. `DEFAULT_TAGGER_DIR` / `TAGGER_HF_SUBFOLDER` in
the torch-free `anime_tools/tagger/dbv4_meta.py` are the single source of truth
for repo and file set (`tagger.py` re-exports them; the ComfyUI node and
`anime_tools/downloads/` — the catalog behind the GUI's Models rows — track
them). `python -m anime_tools.downloads tagger tagger_backbone`, what
⚙ Settings → Models runs, pre-fetches both halves.

Every runtime entry point (position captions, batch autotag, the GUI autotag
server) goes through `ensure_tagger_checkpoint`, which also runs
`ensure_tagger_backbone`: an offline hub-cache probe, then a token fetch on miss
— before SAM3 / the tagger load, so a missing token or unaccepted terms
fails fast with the `hf auth login` + accept-terms hint (a gated 401/403 is
translated in `anime_tools/_hf.py`) instead of a traceback halfway through a
job. `ANIMA_TAGGER_NO_AUTOFETCH=1` refuses the fetch (offline hosts / CI).

## Architecture

```
PIL image → 384² resize
         → caformer_b36.dbv4-full (frozen, external, gated)
         ├→ dbv4 tag sigmoids ─── align_vocab ──→ our 2,532-tag vocab
         ├→ dbv4 rating sigmoids ─ normalise ───→ 4-class rating distribution
         └→ 3072-d MLP-head hidden feature
                    └→ sidecar Linear ──────────→ copyright / missing characters /
                                                  renamed generals / people-count
         → per-tag thresholds → kept tags
         → group argmax → girls-count cap → character floor + original
           fallback → top-1 artist / top-1 copyright
```

The backbone is frozen and external; the only trained weights we ship are
the sidecar linear head (`sidecar.safetensors` + `sidecar.json`). The vocab
build, the rules/groups snapshots and the threshold calibration are ours.

`align_vocab` (`dbv4_backend.py`) is the single vocab join point: it joins
dbv4's snake_case names onto our space-separated vocab, recovering `rules.yaml`
renames. Tags dbv4 does not support sit at logit −30, and a pure-`softmax`
group only emits a winner that clears its own threshold ("at most one" — dbv4
was never CE-trained on our groups). What dbv4 cannot express is exactly what
the sidecar covers: copyright, characters dbv4 does not know, renamed generals,
and an 8-way people-count softmax. Booru-`deprecated` names (`silver hair`,
`light brown hair`, `black footwear` — dbv4's 2025 namespace dropped them) are
*not* head rows: trained, they reached F1 ≈ 0.2, since their positives look
exactly like the live tag's. `tag_rules.yaml` `aliases:` folds each onto its
live name (`silver hair → grey hair`) at vocab-build and emit time instead. Two kinds of character are left out on
purpose: `@artist` (attribution is not a tagger goal) and **artist OCs** —
a tag whose trailing qualifier is a vocab artist handle, `shiro (mignon)`,
whatever category booru filed it under.
Such a name means nothing outside the dataset it came from, and a head trained
on a few dozen positives fires it on any look-alike in a stranger's dataset;
`train_sidecar.py::select_bce_rows` drops them (`--keep_artist_oc` restores,
`sidecar.json["dropped_artist_oc"]` lists them, their thresholds are pinned
to never-fire). Franchise characters dbv4 lacks (`cartethyia (wuthering
waves)`) stay — they are what the head is for. `people_count` is nonetheless always
taken from the count-tag rule (`taxonomy.classify_people`,
`people_count_source="count-tag-rule"`), which beats the sidecar's own head;
the sidecar softmax is exposed as `people_count_scores` only. Thresholds come
from three places: the dbv4 card's `best_threshold` for matched tags, an F1
calibration on our val split for sidecar rows, and `1.01` (never fires) for
everything else.

### Rating band

Anima's rating band is 4-class — `safe, sensitive, nsfw, explicit`.
`anime_tools.tagger.schema.RATINGS` fixes the class order (it is the rating
head's class index); `anime_tools.captions.taxonomy.CAPTION_RATINGS` is the
unordered set the caption-side consumers test against. Danbooru's own literals
are accepted as aliases and folded onto the band at vocab-build time
(`general`→`safe`, `questionable`→`nsfw`), so a raw booru caption classifies as
a rating instead of falling through to the `general` category. `AnimaTagger`
reads `vocab["ratings"]` from the checkpoint and `n_ratings` flows from the
manifest, so the band is a property of the checkpoint, not a loader constant.

## Code layout

`anime_tools/tagger/` — inference + checkpoint schema:

| File | Role |
|---|---|
| `tagger.py` | `AnimaTagger` — public inference class (`predict` / `predict_caption`, and the batched `predict_batch` / `predict_caption_batch`), plus `ensure_tagger_checkpoint` / `ensure_tagger_backbone`. Implements every post-prediction refinement (group argmax, character floor, original-fallback, girls-count cap, top-1 artist/copyright). |
| `dbv4_meta.py` | Torch-free facts about the backbone and our checkpoint: repo ids, required/optional file sets, `DEFAULT_TAGGER_DIR`. Shared by the loader, the ComfyUI node and `downloads/`. |
| `dbv4_backend.py` | Backbone loader + `align_vocab` (the single vocab join point) + `SidecarHead` (our linear head over the backbone's hidden state) + `default_dtype` (bf16 on CUDA, fp32 elsewhere — MPS included). |
| `feature_cache.py` | The dbv4 hidden-state cache: `dbv4_cache_path` / `dbv4_cache_stems` / `load_dbv4_cache` (the stem list rides in the safetensors metadata — a cache built for another manifest is misaligned row-for-row, so every reader checks it) + `multi_hot_from_manifest`. |
| `data.py` | `TaggerCheckpoint.from_dir(path, require=…, backend=…)` — the one read of a checkpoint dir (`config.json` / `vocab.json` / `dataset.json`, the shared "run `--mode build_vocab` first" exit, `idx_to_name`) — and `TaggerManifest`. |
| `readback.py` | Read-It-Back tag-adherence instrument. |

`anime_tools/captions/` — the shared schema the tagger resolves against:
`group_router.py` (`GroupRouter` + `compute_grouped_loss`, typed tag-group
routing with sentinel + escape semantics — the inference rule, the calibrator
and the benches all resolve groups through it), `tag_rules.py` (`tag_rules.yaml`
loader/applier: replacements, tag aliases, always-remove, clothing dedup,
`category_overrides`, `coverage_ignore`), `tag_groups.py`
(`TagGroup`/`TagGroups`/`ResolvedGroup` and the three modes), `taxonomy.py`
(Danbooru category taxonomy + the one count-tag regex) and `correction.py`.

`anime_tools/tagger/cli/` — invoke as `python -m anime_tools.tagger.cli.main`:

| File | Role |
|---|---|
| `main.py` | Argparse + 4-mode dispatcher (`build_vocab`, `predict`, `scan_role_markers`, `derive_groups`). Loads `.env` so `CAPTION_CORPUS_DIR` resolves before defaults are computed. |
| `vocab.py` | Caption discovery, tag categorization (rating literal → `@` artist → count regex → `category_overrides` → tag cache → `general` fallback), `min_freq` cut, train/val split, manifest build, group resolution against the kept vocab, coverage scan. |
| `derive_groups.py` | Taxonomy-driven tag-group candidates (folded into `build_vocab` by default). |
| `build_dbv4_ckpt.py` | Assemble the checkpoint dir: copies vocab / rules / groups / split next to the backend descriptor. |
| `train_sidecar.py` | Sidecar linear-head trainer over cached dbv4 hidden states; calls `calibrate.calibrate_thresholds`. |
| `calibrate.py` | `calibrate_thresholds` — per-tag F1-optimal threshold sweep on val (skips softmax-group tags). Library function only. |
| `eval_metrics.py` | Shared eval + `predict_with_inference_rule` (group argmax / count-tag rule). |
| `predict.py` / `autotag.py` / `autotag_server.py` | Single-image debug entry; CLI one-shot autotag; resident GUI worker. |
| `role_markers.py` | Read-only curator helper (see below). |
| `constants.py` | `find_image_for_caption`, image extensions; re-exports the taxonomy count-tag regex. |

## Configuration via `.env`

External corpus paths are routed via one `.env` key —
`CAPTION_CORPUS_DIR=/path/to/external/caption/corpus`:

| Path | What it is | Consumer |
|---|---|---|
| `<corpus>/retrieved/{artist}/{stem}.{webp,jpg,png,jpeg}` | Source images, paired with `.txt` captions | Training input + label. ~12k images. |
| `<corpus>/retrieved/{artist}/{stem}.txt` | Booru-style caption per image, in Anima format (`rating, count, characters, copyrights, @artists, generals`) | Multi-hot label after `tag_rules` normalization. |
| `<corpus>/retrieved/.tag_cache.json` | `tag → integer type id` (0=general, 1=artist, 3=copyright, 4=character, 5=metadata, 6=deprecated) | Vocab categorization + canonical-emit-slot routing. |
| `<corpus>/tag_rules.yaml` | Replacements + `aliases` (retired name → live tag) + always-remove + clothing dedup + `category_overrides` + `coverage_ignore` | Vocab-build time and inference safety net. Snapshotted into the checkpoint. |
| `<corpus>/tag_groups.yaml` | Typed groupings (`eye_color`, `hair_color`, `hair_length`, `rating`, `top_garment`, …) | Group routing at train + inference. Snapshotted into the checkpoint. |
| `<corpus>/selected/` (optional) | Curated subset (already deduped) | Additional caption source. |

The `src` root — the caption master, `(source_dir)/` by default — is also
scanned.
`CAPTION_CORPUS_DIR` is not committed — it's per-user. The checkpoint
snapshots `rules.yaml` + `groups.yaml`, so inference has zero runtime
dependency on the corpus dir.

## Training pipeline

Every step is a `python -m` module in `anime_tools.tagger.cli`.

```bash
# 1. Vocab + train/val split + per-stem manifest + resolved typed groups.
python -m anime_tools.tagger.cli.main --mode build_vocab --min_freq 5
# 2. Assemble the checkpoint dir (vocab / rules / groups / split + descriptor).
python -m anime_tools.tagger.cli.build_dbv4_ckpt
# 3. Cache dbv4 hidden states + train / calibrate the sidecar head.
python -m anime_tools.tagger.cli.train_sidecar --ckpt_dir models/captioners/anima-tagger-dbv4
# 4. Single-image sanity check.
python -m anime_tools.tagger.cli.main --mode predict --image foo.png --show_scores
# 5. Curator helper — character tags that behave like affiliation markers.
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

The checkpoint dir holds only our files, so it stays moveable across machines;
the backbone weights land in the HF hub cache, not under `models/` — that is
where `Dbv4Backend._load_model` looks. Their gate is auto-approve: `hf auth
login` (or the GUI Settings dialog's token field) plus one click on
[the repo page](https://huggingface.co/animetimm/caformer_b36.dbv4-full). The
hidden-state cache lives at
`<export root>/anima_tagger/dbv4/<arch>_hidden.safetensors`
(`feature_cache.py`), read by `train_sidecar.py` and
`bench/tagger_external/calibration_check.py`.

### Group routing (`GroupRouter`)

`tag_groups.yaml` declares typed groups with one of three modes:

* `softmax_when_solo` — K-way CE over the group's logits when the sample is
  single-subject (`solo`/`1girl`/`1boy`/`1other` fires AND no multi-count tag
  fires) AND no `escape:` tag fires; per-tag BCE otherwise. Used for groups
  mutually exclusive on a single subject (eye color, hair color, hair length,
  primary garment) but irrelevant when an explicit escape applies (e.g.
  `heterochromia` for eye_color, `multicolored hair` for hair_color).
* `softmax` — always K-way CE (modulo `escape:`). For genuinely exclusive
  groups like rating.
* `multilabel` — left in BCE; the group exists only for introspection / UI
  grouping.

`captions/group_router.py` holds the router and `compute_grouped_loss` (BCE on
every (sample, tag), masking the positions CE supervises so each cell has
exactly one term). At inference the same router drives group argmax via
`eval_metrics.predict_with_inference_rule`; the sidecar trainer does not use the
grouped loss (its tags are plain BCE), but the semantics stay pinned by
`tests/test_tagger_sentinel_groups.py` and `tests/test_grouped_loss_negweight.py`.

### Calibration

`calibrate.calibrate_thresholds` sweeps thresholds in `[0.05, 0.95]` step `0.05`
per tag and picks the F1-maximizing one on val. Tags with fewer than
`min_support` val positives, zero achievable F1, or membership in a softmax
group keep `default=0.5` (softmax-group tags are routed by argmax at inference).
Tag-block size of 256 caps memory. dbv4-native tags keep their card thresholds
and are not recalibrated — our val split is far too small to improve on
them; only sidecar tags are swept.

### Role-marker scan

`role_markers.py` is a read-only curator helper. It reads `vocab.json` +
`dataset.json` and ranks every `category=='character'` tag by its conditional
co-occurrence with another character tag on solo training samples (the same
`solo`/`1girl`/`1boy`/`1other` predicate the router applies), auto-bucketing
each candidate: A_costume (shares a name prefix with a top partner → variant
of an existing base; curate via `tag_rules.yaml` `dedup:`), D_role (broad
partner pool, ≥ `--min_role_partners` distinct partners → affiliation marker
mistyped as character, e.g. `sensei (blue archive)`, `doctor (arknights)`;
curate via `remove:`), C_pair (top-1 partner ≥ `--pair_dominance` of
co-occurrences → genuine couple/sibling pair; leave alone) and B_review
(everything else; eyeball). `--out_yaml stub.yaml` writes a YAML stub split into
pasteable sections (A as dedup blocks, D under `remove:`, B/C as commented
hints). No file in the checkpoint dir is mutated.

## Inference

```python
from anime_tools.tagger import AnimaTagger
from PIL import Image

tagger = AnimaTagger("models/captioners/anima-tagger-dbv4")  # default
caption = tagger.predict_caption(Image.open("foo.png"))
# → "sensitive, 1girl, hatsune miku, vocaloid, @some_artist, blue eyes, ..."

debug = tagger.predict(Image.open("foo.png"))
# → {"rating": "...", "rating_scores": {...}, "scores": {...},
#    "kept": {...}, "groups": {"eye_color": "blue eyes", ...}}

# One forward for a whole batch; the post-processing is per image and the same,
# so a batch of one answers what the calls above do. What a walk over a tree
# should use -- the per-image call leaves the backbone waiting on the next decode.
captions = tagger.predict_caption_batch([Image.open(p) for p in paths])
outs = tagger.predict_batch([Image.open(p) for p in paths])
```

`AnimaTagger.predict`:

1. PIL → 384² resize → backbone → tag logits projected onto our vocab by
   `align_vocab`, rating sigmoids normalised to a distribution, and the 3072-d
   hidden feature run through the sidecar head.
2. `sigmoid(tag_logits) ≥ thresholds` → `kept`; `argmax(rating_logits)` → rating.
3. Group-aware refinement. For each loaded `softmax`/`softmax_when_solo`
   group, when the gating predicate applies (single-subject for
   `softmax_when_solo`, always for `softmax`, both modulo escape tags), replace
   any sigmoid-admitted members with the single argmax winner over the group's
   logits.
4. Girls-count cap. When `kept` contains digit-prefixed `Ngirls`, trim
   character predictions to the top-`max(N)` by score — caps independent-sigmoid
   leakage on gender-ambiguous art.
5. Character floor + original fallback. Any character below
   `character_floor` (default `0.5`, above some F1 thresholds as low as `0.05`
   for noisy long-tail characters) is dropped. When that empties the character
   slot AND no copyright survives, add `original` (booru convention for non-IP
   work) so the caption still has a slot-filling copyright.
6. Top-1 artist + top-1 copyright. Independent sigmoid heads can admit
   several borderline tags; collapse to the highest-scoring one (booru
   convention is one artist / one copyright per work).

`predict_caption` then slots tags by canonical category order (`rating, count,
character, copyright, artist, general`), within-slot by median emit position
from the training corpus, re-applies `tag_rules` as a safety net (the dedup map
already fired during training-data normalization, but the model could in
principle predict both `bra` and `black bra`), replaces underscores with
spaces, and joins with `, `.

### The runtime: timm on the best device torch has

The backbone runs on timm, on whatever `anime_tools/_device.py::resolve_device`
answers — `cuda`, else `mps`, else `cpu` — in the dtype `default_dtype` picks for
that device: bfloat16 on CUDA, float32 everywhere else.

Measured on caformer_b36 at 384px — end to end through `predict`, not just the
forward pass:

| Runtime | s/img | max \|Δscore\| vs float32 CPU torch |
|---|---|---|
| torch, bfloat16, CPU (the old default) | 2.39 | 2.1e-02 |
| torch, float32, CPU | 1.80 | — |
| torch, float32, MPS (batch 4) | 0.087 | 2.3e-06 |
| onnxruntime, CPU (retired 2026-09-09) | 0.49 | 7.6e-06 |

Torch emulates bfloat16 on a CPU, so it was both the larger error and the slower
path — hence `default_dtype` picking per device. MPS repeats the lesson: bfloat16
there is 0.102 s/img against float32's 0.087 and 1.3e-02 off, and float16 buys 12 %
(0.077) for 2.4e-03, which is not a trade this stage makes silently.

**The exported ONNX graph is gone.** Until 2026-09-09 an `<ckpt_dir>/dbv4.onnx`
beside the checkpoint switched the backbone onto onnxruntime, because on an Apple
CPU that was 3.7x faster than timm. `mps` is 5.7x faster than the graph was, on the
same machines that wanted it, so the graph, `dbv4_onnx.py`, `onnx_export.py`, the
`export_onnx` CLI, the `tagger_onnx` catalog row, `ANIMA_TAGGER_BACKEND` and
`AnimaTagger.dbv4_runtime` all went together — along with the `onnx`, `onnxscript`
and `onnxruntime` dependencies. An existing `dbv4.onnx` in a checkpoint dir is now
inert and can be deleted; nothing reads it.

Nothing downstream of the score vector ever depended on which runtime produced it:
`Dbv4Output`, the sidecar head, the thresholds, the groups and the slot order were
the same code either way, which is why removing one of the two changed no output.

## Wired-up touchpoints

### Batch auto-tagging (`make caption-autotag`)

`anime_tools/stages/cli/autotag_captions.py` (over `stages/autotag.py`) is the
dataset-wide counterpart to the Dataset tab's per-image button: it walks the
resized tree, tags each image, and writes the `.txt` sidecar beside the resized
image — the revised caption under `workspace/resized/`. The hand-written
master is the read-only fallback (`resolve_caption`), so `--mode missing`
(default) means "no caption speaks for this image"; `merge` appends only novel
tags and round-trips position clauses verbatim; `overwrite` replaces. Every
write keeps what it replaced as a `{stem}.history.txt` version.
Dry run by default, `--apply` writes,
and any apply must be followed by a downstream text-embedding re-encode.

### ComfyUI nodes (`comfyui/anima_tagger/`)

Two nodes share the `ANIMA_TAGGER` socket type:

| Node | Inputs | Outputs |
|------|--------|---------|
| `AnimaTaggerLoader` | `tagger_dir` (STRING) | `tagger` (ANIMA_TAGGER) |
| `AnimaTaggerCaption` | `tagger` (ANIMA_TAGGER), `image` (IMAGE) | `caption` (STRING) |

The node ships in this repo under
[`comfyui/anima_tagger/`](../comfyui/anima_tagger/) and vendors nothing — it
imports `anime_tools.tagger.AnimaTagger` from the installed package — so install
is `pip install ./anime_tools` + link/copy the directory into ComfyUI's
`custom_nodes/` (its README has the exact commands). `AnimaTaggerCaption`
outputs a STRING that drops into any text input.

## Known limitations

1. Rating-class imbalance. Train-corpus rating mix is ~67% explicit / ~32%
   sensitive / ~0.6% safe. Class-weighted CE compensates partially. If
   `safe`-rating accuracy matters downstream, oversample at training time.
2. Per-tag positives are thin for the long tail. At `min_freq=5` each
   long-tail tag has 5–20 positives; calibrated thresholds for those tags are
   noisier than for high-frequency ones. `--min_freq 10` is a knob to revisit if
   F1 disappoints.
3. Long-tail characters lean on `character_floor`. Some F1 thresholds settle
   as low as `0.05`; the post-prediction floor (default `0.5`) is what stops
   borderline guesses from leaking into the caption on stylized /
   gender-ambiguous art. Lowering it recovers recall at the cost of precision.
