# `anime_tools` ↔ `anima_lora` contract

The formats and seams both repos agree on. Tests on either side pin most of
these, so changing a row means changing both repos — worth checking before you
edit one.

## 1. Dependency direction

```
anima_lora (trainer)  ──depends on──▶  anime_tools            never the reverse
```

`tests/test_boundary.py` checks this: no module in this package imports
`library.*`, `networks`, `train`, `gui`, `scripts` or `bench`. It carries its own tiny copies of the
infrastructure it needs (`anime_tools/{_env,_walk,_json,_device,_hf}.py`,
`path_filter.py`) instead.

## 2. File artifacts (what the trainer reads)

The curation stages don't write these paths directly: they write `workspace/`,
and Export publishes from there to the paths below. `workspace/` itself is
curation-private — nothing in the trainer reads it, and its layout
(`anime_tools/workspace/__init__.py`) is ours to change.

Export is `python -m anime_tools.stages.cli.export_workspace` (dry-run by
default, `--apply` to publish; taking one back is `revert_export` over that
run's report, which is what the GUI's **Undo** calls) and the GUI's **Export**
stage. It always copies, skipping anything already identical at the
destination, so re-exporting an unchanged dataset is a walk and a stat apiece.
`python -m anime_tools.workspace.migrate` moves a pre-workspace tree into
`workspace/` in the first place.

| Artifact | Producer (curation) | Consumer (trainer) | Format |
|---|---|---|---|
| `image_dataset/**/{stem}.txt` — caption master | hand-written; Export and the GUI caption editor are its only writers | the caption mirror (`correct_captions --src`), read only for an image with no revised caption yet; caption index | Caption grammar (§3). Parse via `position_clauses.parse_caption`, not `split(",")`. |
| `post_image_dataset/resized/**/{stem}.txt` — revised caption | every caption stage: `correct_captions` corrects it in place (mirroring the master for an image that has none), autotag / position / audit rewrite it, Export publishes `workspace/resized/` here | `preprocess-te` (`library/preprocess/text.py`, beside the resized image) — this is the caption training encodes | Caption grammar (§3). **Revised first, master as read-only fallback** for every reader on either side; once an image has a revised caption a hand-edit of its master no longer reaches it (edit the revised one, or delete it to re-mirror). Every write pushes the replaced text onto `{stem}.history.txt`. |
| `{stem}.variants.txt` (next to the resized image / revised caption) | `anime_tools/captions/variants.py::write_variants_sidecar` | TE caching (`read_variants_sidecar`) → per-variant TE cache rows | UTF-8; `#` comment lines skipped; one `label<TAB>text` per line, order preserved; `v0` = pristine/corrected caption (== `{stem}.txt`), `v1…` shuffles/dropouts, `r1…` identity-randomized draws. Tab-delimited because captions never contain tabs. |
| `post_image_dataset/captions/caption_index.json` | `anime_tools/captions/index.py` (`make caption-index`) | IP-Adapter identity-pair sampler, artist balancing, analytics | `{"meta": {…provenance…}, "image_meta": {key: {path, character[], copyright[], artist[], count[]}}, "groups": {axis: {tag: [key…]}}}`. `key` = path relative to the source root, extension stripped, `/`-separated (`caption_key`). Sampling policy lives in the trainer, not in this file. |
| `masks/**/{stem}_mask.png` | `anime_tools/masking/cli/{generate_masks,generate_masks_mit,merge_masks}.py` (trainer `make mask`; `scripts/preprocess/*.py` are forwarding shells) | `library/datasets/image_utils.py::load_mask_from_dir` via subset `mask_dir` | 8-bit **L** PNG, `{stem}_mask.png`; nested layout mirrors the source subdir under `mask_dir` (flat `mask_dir/{stem}_mask.png` is the legacy fallback). Loader converts to L, NEAREST-resizes to the latent's pixel size, scales to `[0,1]`. |
| `post_image_dataset/groups/groups.json` | `anime_tools/grouping/cli/build_groups.py` (trainer `make curate-group`) | GUI Dataset tab group filter (`gui/tabs/image_tab.py`) | `MANIFEST_VERSION = 2`: `{version, source_dir, encoder, cell_match_min, match_frac_min, sim_min, grid, ratio, min_size, n_images, n_groups, n_grouped, n_singletons, groups: [{id, artist, size, mean_cosine, members: [rel-posix…]}]}`. |
| grouping feature cache `$NEAR_TWIN_CACHE/<dirhash>/{stem}.npz` (default `~/.cache/near_twin/`) | `anime_tools/grouping/features.py::embed_members` | grouping + near-twin miner only | curation-private; `cls` `[D]` f32 L2-normed + `grid16` `[16,16,D]` f16. Not the trainer's `{stem}_anima_pe.safetensors`. Switch embedders ⇒ new cache root. |
| decensor match tables (`anime_tools/grouping/cli/{match,apply}_decensored.py`) | curation | curation | internal to the curation side; not read by the trainer. |

Resized PNGs under `post_image_dataset/resized/` are produced by **both** sides:
`anime_tools/stages/resize.py` (`python -m anime_tools.stages.cli.resize_images`,
the GUI's Resize stage) and the trainer's `make preprocess-resize`, which since
the API-first migration (2026-09-03) builds the same `ResizeRequest` and runs
the same stage — in-process under its daemon, as a `python -m` child from a
shell. There is one geometry: `anime_tools/buckets.py` **owns** the free-fit
tiers, bands and solver (`EDGE_TOKEN_BANDS`, `choose_edge`,
`freefit_band_for_edge`, `freefit_bucket`) and the trainer's
`library/datasets/buckets.py` re-exports them, so whichever side runs first the
other finds every image already at its target bucket and skips it. The tiers
(`--target_res`) still have to match or each pass re-resizes the other's output.
`ResizeRequest.skip` (paths relative to `--src`) is how the trainer GUI's
per-image curation decisions (`post_image_dataset/curation_decisions.json`,
actions `skip` / `move`) reach the pass; the package has no notion of that file.

Not in the contract (trainer-owned caches, never produced by curation): VAE
latents `{stem}_{WxH}_anima.npz`, TE `{stem}_anima_te.safetensors`, PE
`{stem}_anima_pe.safetensors`, σ-demote siblings.

## 3. Caption grammar (the one shared parser)

`<flat tag bag>. <Position clause>. <Position clause>. … <Text clause>.`

- The period delimits clauses; commas separate tags inside one; the first
  clause is the flat bag. `@artist` handles, the `@no-artist` sentinel and
  `On the left, …` / `In the …` headers are grammar, not tags.
- A closed quote pair (`「…」`, `『…』`, `"…"` — `QUOTE_PAIRS`) is one opaque
  tag: a comma or `. On the` inside it is content. A **text clause**
  (`Japanese text reads as "…", "…"` / `Japanese SFX reads as …`,
  `TEXT_PREFIXES`) carries a page's OCR'd lines as quoted tags in reading
  order; it composes last, after every position clause, and the variants
  pass never shuffles, drops or randomizes it. `has_clauses` means
  *position* clauses only.
- One implementation, in `anime_tools/captions/position_clauses.py`
  (`parse_caption` / `compose_caption`), plus the shuffle grammar in
  `anime_tools/captions/shuffle.py` (`NO_ARTIST_SENTINEL`,
  `find_anima_prefix_end`, `strip_no_artist_sentinel`,
  `anima_smart_shuffle_caption`). The trainer imports both
  (`library.anima.training` re-exports the shuffle names today) — a second copy
  on either side is how the two drift apart.

## 4. Code-level seams

| Seam | Decision |
|---|---|
| **Tokenizers** (caption length / erasure pool) | Curation scripts take tokenizer directories only (`anime_tools/captions/tokenizers.py::load_{qwen3,t5}_tokenizer_from_dir`). The trainer wrapper resolves a `.safetensors` text-encoder path → bundled config dir (`library.anima.weights.qwen3_tokenizer_dir` / `t5_tokenizer_dir`) and passes `--qwen3 <dir> --t5_tokenizer_path <dir>`. Curation never learns the safetensors→config mapping. |
| **Grouping embedder** | `anime_tools.grouping.features.Embedder` protocol: `.device`, `.dtype`, `__call__(batch[B,3,512,512] in [-1,1]) -> (cls[B,D] f32 L2-normed, grid16[B,16,16,D] f16)`. PE-Spatial-B16-512 lives here: the vendored PE tower is `anime_tools.vision.pe` (`load_pe_spatial()`, Hub fetch, `ANIME_TOOLS_MODELS/pe/`), the default embedder is `anime_tools.grouping.embedder:pe_spatial_embedder` (bf16), and `build_groups` / the CLI use it when no `--embedder module:callable(device=…)` override is given. The trainer re-exports the tower as `library.models.pe` (REPA / CMMD / PE caching) and keeps its own encoder registry (`library.vision.{encoder,encoders,buckets}`), which the package never imports. |
| **Tagger backend** | dbv4 only (`config.json["backend"] == "dbv4"`). The tagger never imports `library.vision`. Checkpoint layout: `config.json`, `vocab.json`, `rules.yaml`, `groups.json`, `thresholds.safetensors`, optional `sidecar.safetensors`; GPL backbone fetched from the gated upstream repo at load. |
| **Home / model paths** | `anime_tools/_env.py`: `curation_home()` = `ANIME_TOOLS_HOME` → `ANIMA_HOME` → checkout root; `resolve_path()` anchors bare relatives there; `models_dir()` = `ANIME_TOOLS_MODELS` → `<home>/models`; `workspace_dir()` = `ANIME_TOOLS_WORKSPACE` → `<home>/workspace`, everything the curation stages write. In-tree all three coincide with `library.env.anima_home()`, so nothing changes for the trainer; a standalone `anime_tools` install sets `ANIME_TOOLS_HOME`/`ANIME_TOOLS_MODELS`. The trainer's `make` wrappers keep passing explicit dirs (`--tagger_dir`, `--src/--dst`). |
| **`path_pattern` glob** | One implementation, `anime_tools/path_filter.py::filter_paths_by_glob`; training subsets and every curation stage share it. |
| **HF fetch** | `anime_tools/_hf.py`; tests patch the canonical path. |
| **Shared constants** | `anime_tools/contract.py`, stdlib-only: the autotag worker's stdout sentinels, autotag `--mode` values, the tagger checkpoint file sets, the replay report name and shapes, the GUI drawer attribute, and `CONTRACT_VERSION`. The trainer reads these from here instead of spelling them; the surface is append-only within one version. |
| **Process boundary** | Curation stages are plain CLIs; the trainer's daemon wraps them (`make … --queue`). No daemon client in `anime_tools`. A stage run with `ANIMA_DAEMON_JOB_DIR` set appends to `<job_dir>/progress.jsonl` in the daemon's own line shape (`anime_tools/_progress.py`: `{"ev": "step", "global_step", "total_steps", "detail"}` per image, `phase` / `heartbeat` lines around a model load); without it, nothing. |


