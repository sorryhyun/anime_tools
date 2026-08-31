# `anime_tools` ↔ `anima_lora` contract

Status: **frozen**. Both repos' tests pin every row below; a change to any row
is a two-PR change by design.

## 1. Dependency direction

```
anima_lora (trainer)  ──depends on──▶  anime_tools            never the reverse
```

Enforced by `tests/test_boundary.py`: no module in this package may import
`library.*`, `networks`, `train`, `gui`, `scripts` or `bench`. The package
carries its own tiny copies of the infrastructure it needs
(`anime_tools/{_env,_walk,_json,_device,_hf}.py`, `path_filter.py`) rather than
importing the trainer's.

## 2. File artifacts (what the trainer reads)

The curation stages do not write these paths directly: they write `workspace/`,
and **Export** publishes from there to the paths in this table. `workspace/`
itself is curation-private in the same sense as the near-twin feature cache:
nothing in the trainer reads it, and its layout
(`anime_tools/workspace/__init__.py`) is not part of this contract.

Export is `python -m anime_tools.stages.cli.export_workspace` (dry-run by
default, `--apply` to publish, `--undo <report>` to take it back) and the GUI's
**Export** stage. It always copies, skipping anything already identical at the
destination, so re-exporting an unchanged dataset is a walk and a stat apiece.
`python -m anime_tools.workspace.migrate` moves a pre-workspace tree into
`workspace/` in the first place.

| Artifact | Producer (curation) | Consumer (trainer) | Format |
|---|---|---|---|
| `image_dataset/**/{stem}.txt` — caption master | autotag / position / correction stages, GUI caption editor | `preprocess-te` (`library/preprocess/text.py`), caption index | **Caption grammar** (§3). Parsed only via `position_clauses.parse_caption` — never `split(",")`. |
| `{stem}.variants.txt` (next to the resized image / derived caption) | `anime_tools/captions/variants.py::write_variants_sidecar` | TE caching (`read_variants_sidecar`) → per-variant TE cache rows | UTF-8; `#` comment lines skipped; one `label<TAB>text` per line, order preserved; `v0` = pristine/corrected caption (== `{stem}.txt`), `v1…` shuffles/dropouts, `r1…` identity-randomized draws. Tab-delimited because captions never contain tabs. |
| `post_image_dataset/captions/caption_index.json` | `anime_tools/captions/index.py` (`make caption-index`) | IP-Adapter identity-pair sampler, artist balancing, analytics | `{"meta": {…provenance…}, "image_meta": {key: {path, character[], copyright[], artist[], count[]}}, "groups": {axis: {tag: [key…]}}}`. `key` = path relative to the source root, extension stripped, `/`-separated (`caption_key`). **No sampling policy inside.** |
| `masks/**/{stem}_mask.png` | `anime_tools/masking/cli/{generate_masks,generate_masks_mit,merge_masks}.py` (trainer `make mask`; `scripts/preprocess/*.py` are forwarding shells) | `library/datasets/image_utils.py::load_mask_from_dir` via subset `mask_dir` | 8-bit **L** PNG, `{stem}_mask.png`; nested layout mirrors the source subdir under `mask_dir` (flat `mask_dir/{stem}_mask.png` is the legacy fallback). Loader converts to L, NEAREST-resizes to the latent's pixel size, scales to `[0,1]`. |
| `post_image_dataset/groups/groups.json` | `anime_tools/grouping/cli/build_groups.py` (trainer `make curate-group`) | GUI Dataset tab group filter (`gui/tabs/image_tab.py`) | `MANIFEST_VERSION = 2`: `{version, source_dir, encoder, cell_match_min, match_frac_min, sim_min, grid, ratio, min_size, n_images, n_groups, n_grouped, n_singletons, groups: [{id, artist, size, mean_cosine, members: [rel-posix…]}]}`. |
| grouping feature cache `$NEAR_TWIN_CACHE/<dirhash>/{stem}.npz` (default `~/.cache/near_twin/`) | `anime_tools/grouping/features.py::embed_members` | grouping + near-twin miner only | **curation-private**; `cls` `[D]` f32 L2-normed + `grid16` `[16,16,D]` f16. Not the trainer's `{stem}_anima_pe.safetensors`. Switch embedders ⇒ new cache root. |
| decensor match tables (`anime_tools/grouping/cli/{match,apply}_decensored.py`) | curation | curation | internal to the curation side; not read by the trainer. |

Resized PNGs under `post_image_dataset/resized/` are produced by **both** sides:
`anime_tools/stages/resize.py` (`python -m anime_tools.stages.cli.resize_images`,
the GUI's Resize stage) and the trainer's `make preprocess-resize`. They are
interchangeable by construction — same tier (`choose_edge`), same free-fit band
and solver, same `anima_resize_{crop_anchor,bucket_resos,crop_margins}` PNG text
keys — so whichever side runs first, the other finds every image already at its
target bucket and skips it. `anime_tools/buckets.py` is a copy of the trainer's
free-fit geometry, not an import; the two must move together, and the tiers
(`--target_res`) must match on both sides or each pass re-resizes the other's
output.

Not in the contract (trainer-owned caches, never produced by curation): VAE
latents `{stem}_{WxH}_anima.npz`, TE `{stem}_anima_te.safetensors`, PE
`{stem}_anima_pe.safetensors`, σ-demote siblings.

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

## 4. Code-level seams

| Seam | Decision |
|---|---|
| **Tokenizers** (caption length / erasure pool) | Curation scripts take tokenizer **directories** only (`anime_tools/captions/tokenizers.py::load_{qwen3,t5}_tokenizer_from_dir`). The trainer wrapper resolves a `.safetensors` text-encoder path → bundled config dir (`library.anima.weights.qwen3_tokenizer_dir` / `t5_tokenizer_dir`) and passes `--qwen3 <dir> --t5_tokenizer_path <dir>`. Curation never learns the safetensors→config mapping. |
| **Grouping embedder** | `anime_tools.grouping.features.Embedder` protocol: `.device`, `.dtype`, `__call__(batch[B,3,512,512] in [-1,1]) -> (cls[B,D] f32 L2-normed, grid16[B,16,16,D] f16)`. **PE-Spatial-B16-512 is owned here** — the vendored PE tower is `anime_tools.vision.pe` (`load_pe_spatial()`, Hub fetch, `ANIME_TOOLS_MODELS/pe/`), the default embedder is `anime_tools.grouping.embedder:pe_spatial_embedder` (bf16), and `build_groups` / the CLI use it when no `--embedder module:callable(device=…)` override is given. The trainer re-exports the tower as `library.models.pe` (REPA / CMMD / PE caching) and keeps its own encoder registry (`library.vision.{encoder,encoders,buckets}`), which the package never imports. |
| **Tagger backend** | dbv4 only (`config.json["backend"] == "dbv4"`). The tagger never imports `library.vision`. Checkpoint layout: `config.json`, `vocab.json`, `rules.yaml`, `groups.json`, `thresholds.safetensors`, optional `sidecar.safetensors`; GPL backbone fetched from the gated upstream repo at load. |
| **Home / model paths** | `anime_tools/_env.py`: `curation_home()` = `ANIME_TOOLS_HOME` → `ANIMA_HOME` → checkout root; `resolve_path()` anchors bare relatives there; `models_dir()` = `ANIME_TOOLS_MODELS` → `<home>/models`; `workspace_dir()` = `ANIME_TOOLS_WORKSPACE` → `<home>/workspace`, everything the curation stages write. In-tree all three coincide with `library.env.anima_home()`, so nothing changes for the trainer; a standalone `anime_tools` install sets `ANIME_TOOLS_HOME`/`ANIME_TOOLS_MODELS`. The trainer's `make` wrappers keep passing explicit dirs (`--tagger_dir`, `--src/--dst`). |
| **`path_pattern` glob** | One implementation, `anime_tools/path_filter.py::filter_paths_by_glob`; training subsets and every curation stage share it. |
| **HF fetch** | `anime_tools/_hf.py`; tests patch the canonical path. |
| **Process boundary** | Curation stages are plain CLIs; the trainer's daemon wraps them (`make … --queue`). No daemon client in `anime_tools`. |


