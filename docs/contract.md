# `anime_tools` ↔ `anima_lora` contract

Status: **Phase 0 (2026-08-30)** — frozen in-tree before any code moves; lands in
`anime_tools/docs/contract.md` at Phase 1. Companion to
[`curation_repo_split.md`](curation_repo_split.md). Both repos' tests pin every
row below; a change to any row is a two-PR change by design.

## 1. Dependency direction

```
anima_lora (trainer)  ──depends on──▶  anime_tools            never the reverse
```

Enforced today by `tests/test_curation_boundary.py`: every file in the move
manifest must not import **any** `library.*` module outside the manifest, nor
`networks` / `train`. The curation set carries its own tiny copies of the
infrastructure it needs (`anime_tools/captions/{_env,_walk,_hf}.py`,
`path_filter.py`); `tests/test_curation_walk_parity.py` pins the copied walkers
to the trainer originals.

## 2. File artifacts (what the trainer reads)

| Artifact | Producer (curation) | Consumer (trainer) | Format |
|---|---|---|---|
| `image_dataset/**/{stem}.txt` — caption master | autotag / position / correction stages, GUI caption editor | `preprocess-te` (`library/preprocess/text.py`), caption index | **Caption grammar** (§3). Parsed only via `position_clauses.parse_caption` — never `split(",")`. |
| `{stem}.variants.txt` (next to the resized image / derived caption) | `anime_tools/captions/variants.py::write_variants_sidecar` | TE caching (`read_variants_sidecar`) → per-variant TE cache rows | UTF-8; `#` comment lines skipped; one `label<TAB>text` per line, order preserved; `v0` = pristine/corrected caption (== `{stem}.txt`), `v1…` shuffles/dropouts, `r1…` identity-randomized draws. Tab-delimited because captions never contain tabs. |
| `post_image_dataset/captions/caption_index.json` | `anime_tools/captions/index.py` (`make caption-index`) | IP-Adapter identity-pair sampler, artist balancing, analytics | `{"meta": {…provenance…}, "image_meta": {key: {path, character[], copyright[], artist[], count[]}}, "groups": {axis: {tag: [key…]}}}`. `key` = path relative to the source root, extension stripped, `/`-separated (`caption_key`). **No sampling policy inside.** |
| `masks/**/{stem}_mask.png` | `scripts/preprocess/{generate_masks,generate_masks_mit,merge_masks}.py` (`make mask`) | `library/datasets/image_utils.py::load_mask_from_dir` via subset `mask_dir` | 8-bit **L** PNG, `{stem}_mask.png`; nested layout mirrors the source subdir under `mask_dir` (flat `mask_dir/{stem}_mask.png` is the legacy fallback). Loader converts to L, NEAREST-resizes to the latent's pixel size, scales to `[0,1]`. |
| `post_image_dataset/groups/groups.json` | `scripts/curate/build_groups.py` (`make curate-group`) | GUI Dataset tab group filter (`gui/tabs/image_tab.py`) | `MANIFEST_VERSION = 2`: `{version, source_dir, encoder, cell_match_min, match_frac_min, sim_min, grid, ratio, min_size, n_images, n_groups, n_grouped, n_singletons, groups: [{id, artist, size, mean_cosine, members: [rel-posix…]}]}`. |
| grouping feature cache `$NEAR_TWIN_CACHE/<dirhash>/{stem}.npz` (default `~/.cache/near_twin/`) | `library/vision/pe_features.py::embed_members` | grouping + near-twin miner only | **curation-private**; `cls` `[D]` f32 L2-normed + `grid16` `[16,16,D]` f16. Not the trainer's `{stem}_anima_pe.safetensors`. Switch embedders ⇒ new cache root. |
| decensor match tables (`scripts/curate/{match,apply}_decensored.py`) | curation | curation | internal to the curation side; not read by the trainer. |

Not in the contract (trainer-owned caches, never produced by curation): VAE
latents `{stem}_{WxH}_anima.npz`, TE `{stem}_anima_te.safetensors`, PE
`{stem}_anima_pe.safetensors`, σ-demote siblings, resized PNGs.

## 3. Caption grammar (the one shared parser)

`<flat tag bag>. <Position clause>. <Position clause>. …`

- The **period** delimits clauses; **commas** separate tags inside one; the
  first clause is the flat bag. `@artist` handles, the `@no-artist` sentinel
  and `On the left, …` / `In the …` headers are grammar, not tags.
- Single implementation: `anime_tools/captions/position_clauses.py`
  (`parse_caption` / `compose_caption`) + the shuffle grammar in
  `anime_tools/captions/shuffle.py` (`NO_ARTIST_SENTINEL`, `find_anima_prefix_end`,
  `strip_no_artist_sentinel`, `anima_smart_shuffle_caption`). Both move to
  `anime_tools.captions`; the trainer imports them (`library.anima.training`
  re-exports the shuffle names today). **Never fork either.**

## 4. Code-level seams (Phase 0 decisions)

| Seam | Decision |
|---|---|
| **Tokenizers** (caption length / erasure pool) | Curation scripts take tokenizer **directories** only (`anime_tools/captions/tokenizers.py::load_{qwen3,t5}_tokenizer_from_dir`). The trainer wrapper resolves a `.safetensors` text-encoder path → bundled config dir (`library.anima.weights.qwen3_tokenizer_dir` / `t5_tokenizer_dir`) and passes `--qwen3 <dir> --t5_tokenizer_path <dir>`. Curation never learns the safetensors→config mapping. |
| **Grouping embedder** | `library.vision.pe_features.Embedder` protocol: `.device`, `.dtype`, `__call__(batch[B,3,512,512] in [-1,1]) -> (cls[B,D] f32 L2-normed, grid16[B,16,16,D] f16)`. `build_groups(embedder=…)` loads no encoder; the CLI takes `--embedder module:callable(device=…)`. The trainer's implementation is `library.vision.grouping_embedder:pe_spatial_embedder` (PE-Spatial-B16-512), injected by `make curate-group`. PE-Core/PE-Spatial loading **stays in the trainer**. |
| **Tagger backend** | dbv4 only (`config.json["backend"] == "dbv4"`); the in-house PE dual-encoder head was deleted 2026-08-30 (archived under `_archive/anima_tagger_training/pe_backend_removed_2026_08_30/`). The tagger never imports `library.vision`. Checkpoint layout: `config.json`, `vocab.json`, `rules.yaml`, `groups.json`, `thresholds.safetensors`, optional `sidecar.safetensors`; GPL backbone fetched from the gated upstream repo at load. |
| **Home / model paths** | `anime_tools/_env.py`: `curation_home()` = `ANIME_TOOLS_HOME` → `ANIMA_HOME` → checkout root; `resolve_path()` anchors bare relatives there; `models_dir()` = `ANIME_TOOLS_MODELS` → `<home>/models`. In-tree all three coincide with `library.env.anima_home()`, so nothing changes for the trainer; a standalone `anime_tools` install sets `ANIME_TOOLS_HOME`/`ANIME_TOOLS_MODELS`. The trainer's `make` wrappers keep passing explicit dirs (`--tagger_dir`, `--src/--dst`). |
| **`path_pattern` glob** | One implementation, `anime_tools/path_filter.py::filter_paths_by_glob` (moved from `library/datasets/`, shim left); training subsets and every curation stage share it. |
| **HF fetch** | `anime_tools/_hf.py` (moved from `library/runtime/hf_download.py`, shim left); tests patch the canonical path. |
| **Process boundary** | Curation stages are plain CLIs; the trainer's daemon wraps them (`make … --queue`). No daemon client in `anime_tools`. |

## 5. Guarantees during the move (Phases 1–2)

- Byte-identical `{stem}.txt` + `.variants.txt` + TE caches on the live dataset
  (hash the caption master before/after).
- Byte-identical masks on a fixed 20-image sample.
- Same `groups.json` for the same thresholds + embedder.
- Old import paths keep working one release (`anime_tools.captions`,
  `library.vision.{pe_features,pe_matching}`, `library.datasets.path_filter`,
  `library.runtime.hf_download`) as warning shims; deleted in Phase 3.
