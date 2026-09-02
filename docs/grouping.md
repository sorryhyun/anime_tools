# Grouping — near-twin components over PE-Spatial features

Ships as the **Groups** stage: it embeds every image in the resized tree, connects the pairs that
are near-twins, and writes `groups.json`. The sidebar's *groups* ordering reads that manifest.
Nothing else does — grouping is a curation aid for thinning duplicates and balancing concepts, not
a preprocess or training step, and it never touches a caption.

## 1. What a near-twin is here

Two images are compared on two features from PE-Spatial-B16-512 (`anime_tools/vision/pe.py`),
computed once per image at the model's native 512×512 square bucket:

- `cls` — the global descriptor, L2-normed. Its cosine is a loose "same picture at all?" signal.
- `grid16` — the 32×32 patch grid pooled to 16×16 cells. This is what decides.

A pair goes through two stages (`grouping/groups.py::_grid_match_edges`):

1. **Stage A, prefilter.** `cls` cosine must reach `--sim-min`. This only prunes obviously unrelated
   pairs so the per-folder pass stays proportional to the candidate pairs; it is not the gate.
2. **Stage B, grid match** (`grouping/matching.py::match_fracs`). Each `grid16` is pooled again to
   `--grid`×`--grid` cells (7×7 = 49 by default). A cell of image A is an *inlier* when its nearest
   cell in B has cosine at least `--cell-match-min`, that cell's own nearest in A is the same cell
   (mutual nearest neighbour), and the match is distinctive under a ratio test: the cosine distance
   to the best cell must be at most `--ratio` times the distance to the second best. Flat colour
   fields in anime art give many cells a >0.9 neighbour, which is why distinctiveness matters. The
   pair is connected when the inlier fraction reaches `--match-frac-min`.

Connected pairs are unioned into components (`connected_components`, plain union-find), so a
group is transitive: A–B and B–C put A and C together even if they never matched directly.
Components are listed largest first; anything smaller than `--min-size` is dropped from the
manifest and stays a singleton.

**Scope is the top-level folder** under the source dir (`_artist_of`), so two folders never merge
and a folder with one image is skipped. A flat tree is one bucket.

Which way is stricter:

| Knob | Default | Stricter |
|---|---|---|
| `--cell-match-min` | 0.93 | higher |
| `--match-frac-min` | 0.25 | higher |
| `--sim-min` | 0.5 | higher (but it is only the prefilter) |
| `--ratio` | 0.8 | lower |
| `--grid` | 7 | finer cells make each cell easier to miss |

The default inlier fraction is deliberately low so a partial overlap — one edited region, a
variant with a different expression — still groups. `--match-frac-min 0.4` is a reasonable first
tightening.

The batched gate carries no translation check: the scalar `match_grids` has a RANSAC-lite
`_geom_filter` that rejects "same character, different pose", but `build_groups` runs the
vectorized `match_fracs`, which returns only the inlier fraction. The ratio test and the fraction
threshold are what stand between a pose change and a group.

## 2. What it produces

`--out` defaults to `workspace/groups/groups.json` (`MANIFEST_VERSION = 2`, the row in
`docs/contract.md` §2):

```json
{
  "version": 2, "source_dir": "…/workspace/resized", "encoder": "pe_spatial",
  "cell_match_min": 0.93, "match_frac_min": 0.25, "sim_min": 0.5, "grid": 7, "ratio": 0.8,
  "min_size": 2, "n_images": 3007, "n_groups": 41, "n_grouped": 97, "n_singletons": 2910,
  "groups": [
    {"id": 0, "artist": "ama_mitsuki", "size": 3, "mean_cosine": 0.9712,
     "members": ["ama_mitsuki/5847168.webp", "ama_mitsuki/5847169.webp", "…"]}
  ]
}
```

`members` are POSIX paths relative to `source_dir`, extension included; `mean_cosine` is the mean
pairwise `cls` cosine of the component, a readability score rather than a gate. `encoder` is the
embedder's `name` (its class name if it has none). An empty source tree still writes a manifest,
with zero groups.

The manifest is not one of Export's artifact kinds. The trainer's own `make curate-group` runs this
CLI with the trainer's output path, and its Dataset tab folds the members under green group
headers; see the trainer guidebook §7.4.

## 3. Running it

In the GUI, **Groups › Build groups**. The form shows `--sim-min` and `--min-size`; the rest
folds under *advanced*. `--source-dir` is bound to the dataset's `dst` root and `--out` to the
report root, so neither is asked, and `--device` is resolved in the child. Because the stage is
bound to `dst`, the GUI runs **Resize** in front of it — the resized tree is the pixel data every
other stage reads.

From a checkout:

```bash
python -m anime_tools.grouping.cli.build_groups                 # workspace/resized → workspace/groups/groups.json
python -m anime_tools.grouping.cli.build_groups --match-frac-min 0.4 --cell-match-min 0.9
python -m anime_tools.grouping.cli.build_groups --min-size 1     # keep singletons in the manifest
python -m anime_tools.downloads pe_spatial                       # fetch the weights up front (optional)
```

Flags are hyphenated; the underscore spelling is an alias. Bare relative paths resolve against
the curation home (`ANIME_TOOLS_HOME` → `ANIMA_HOME` → the current directory), not the shell's
cwd. The stage prints one tally line when it is done:

```
41 group(s) over 3007 image(s) (97 grouped, 2910 ungrouped) @ cell_match_min 0.93 / match_frac_min 0.25 → …/groups.json
```

Progress is two `tqdm` bars on stderr: `embedding` per image (absent on a fully cached re-run) and
`grouping` per top-level folder. The same object is `GroupRequest` from Python
(`examples/grouping.py`):

```python
from anime_tools.grouping import GroupRequest, run_groups

manifest = run_groups(GroupRequest(match_frac_min=0.4))  # also written to req.out
```

An image that fails to decode is warned about on stderr, skipped, and appears in no group. The
weights (`facebook/PE-Spatial-B16-512`, one `.pt`) are fetched into `<models>/pe/` on first use;
`ANIME_TOOLS_MODELS` moves that directory.

## 4. Reading it in the sidebar

The sidebar draws one listing in two orderings, *tree* and *groups*. In *groups* mode the rows
are folded as *folder → component → images*, each component headed by its mean cosine, with every
image the manifest does not place under a trailing **ungrouped** bucket, so switching modes cannot
lose an image. Filters and pending dots mean the same thing in both modes because the server
answers rels only (`gui/dataset.py::load_groups`) and the browser joins them onto the listing it
already has; a component the filter cuts down to one visible row is not shown as a group.

Three notes the view can carry:

- **no manifest** at `<report_root>/groups/groups.json` — build one from the Groups panel. Not an
  error; an unparseable file is.
- **older manifest** — the file's `version` is not 2. The components are still usable; rebuild to
  pick up the current gate.
- **nothing clustered**, with the `source_dir` the manifest was built from — a manifest built
  against another tree joins onto nothing, which is why that path rides along.

## 5. The feature cache

Embedding is the expensive half, so features are cached per image under `$NEAR_TWIN_CACHE`
(default `~/.cache/near_twin/`) as `<sha1 of the parent dir>[:16]/<stem>.npz`, holding `cls`,
`grid16`, the source's `(size, mtime_ns)` and `FEATURE_CACHE_VER`. A re-run at other thresholds is
only the matching pass.

The key addresses a *location*, so it does not move when the pixels underneath are rewritten. The
stamp is what makes that a miss: **Resize rewrites `workspace/resized/` under the same names**, and
the stamp check is what stops a regenerated tree from reading stale features
(`tests/test_grouping_features.py` pins it). Anything wrong with an entry — no file, a truncated
`.npz`, a pre-stamp or older-version entry, a moved stamp, an unreadable source — means recompute,
never an error.

The cache does not record which embedder wrote it. Switching embedders means a fresh
`$NEAR_TWIN_CACHE` root. The default embedder runs in bf16, which is what existing entries were
written with. This cache is curation-private and is not the trainer's `{stem}_anima_pe.safetensors`.

## 6. Custom embedders

`--embedder module:callable` names a factory that is imported and called with `device=`; the
default is `anime_tools.grouping.embedder:pe_spatial_embedder`. A spelling without the colon is
refused by the request. The result must satisfy `grouping/features.py::Embedder`:

- attributes `device` (`torch.device`) and `dtype`;
- `__call__(batch)` with `batch` `[B, 3, 512, 512]` in `[-1, 1]` on that device and dtype,
  returning `(cls [B, D] float32 L2-normed, grid16 [B, 16, 16, D] float16)` as numpy;
- optionally `name`, which becomes the manifest's `encoder`.

`D` is free; the thresholds were tuned on PE-Spatial's 768-wide features and will want retuning.

## 7. The decensor match tools

Two further CLIs under `grouping/cli/` solve a narrower problem with a different matcher: pairing
censored training images with an uncensored drop whose filenames and resolutions do not
correspond. They take **no flags for paths** — both read six fixed locations off the curation
home (`cli/_decensored.py`): `image_dataset/sincos/` (the censored originals), `sincos_decensored/`
(the drop), and `output/curate/sincos_decensored/` for everything they write.

`python -m anime_tools.grouping.cli.match_decensored` compares every original to every candidate
on PIL/numpy descriptors — a 64-bit DCT perceptual hash on a 32×32 grayscale (Hamming distance), a
z-normalized 16×16 thumbnail (cosine) and a 6% aspect-ratio gate — and tiers the best candidate:

| Tier | Condition |
|---|---|
| `auto` | Hamming ≤ 6 and thumbnail cosine ≥ 0.95 |
| `review` | Hamming ≤ 12 |
| `skip` | otherwise (`no_match`), or the caption carries no censor tag (`no_censor_tag`), or the match is animated (`animated_match`) |

The censor-tag check reads the caption's flat bag through `features.read_tags`, so a clause header
cannot vote and `convenient_hair` normalizes like `convenient hair`. It writes `matches.csv`
(best first) and `review.html`, a side-by-side contact sheet to eyeball the `review` tier in a
browser. Descriptors are cached one `.npz` per directory, stamped with the newest mtime, the file
count and `CACHE_VER` — deliberately not the near-twin cache, whose per-image `.npz` layout and
stamp answer a different question.

`python -m anime_tools.grouping.cli.apply_decensored` is dry-run by default. With `--apply` it
copies each selected original to `output/curate/sincos_decensored/backup_censored/`, overwrites it
with its match, leaves the caption alone, and deletes the image-derived caches for that stem
(`workspace/resized/sincos/<stem>.*` and `workspace/lora/sincos/<stem>_*` except the text-only
`_anima_te.safetensors`) so the next preprocess regenerates them. `--include-review` adds the
`review` tier; `--only a,b` and `--exclude a,b` restrict by stem. What it did lands in
`applied.csv`.

## 8. Limits

- Every image is decoded to a 512×512 **square**, so aspect is not preserved; the grid match is
  tolerant of the resulting stretch but a crop that changes the aspect a lot scatters cells.
- Scope is the top-level folder. A twin that straddles two folders is never found.
- The manifest is a snapshot: an image added after the run is simply ungrouped until the next one.
- There is no geometry check on the batched path (§1); tighten `--ratio` or `--match-frac-min`
  before reaching for the scalar matcher.

## 9. Code map

| File | Role |
|---|---|
| `anime_tools/grouping/requests.py` | `GroupRequest` — the flags, torch-free |
| `anime_tools/grouping/groups.py` | `run_groups` / `build_groups`, the two-stage gate, union-find, the manifest |
| `anime_tools/grouping/matching.py` | `pool_cells_batch`, `match_fracs`; the scalar `match_grids` + `_geom_filter` |
| `anime_tools/grouping/features.py` | `Embedder` protocol, `Feature`, the `$NEAR_TWIN_CACHE` read/write, `read_tags` |
| `anime_tools/grouping/embedder.py` | `pe_spatial_embedder`, the default factory |
| `anime_tools/vision/pe.py` | the vendored PE-Spatial tower, `load_pe_spatial` |
| `anime_tools/grouping/cli/build_groups.py` | the shell: `GroupRequest.parser()` → `run_groups` |
| `anime_tools/grouping/cli/{match,apply}_decensored.py`, `_decensored.py` | the decensor pass and its fixed paths |
| `anime_tools/gui/dataset.py::load_groups` | the manifest as the sidebar's group view |
| `frontend/src/components/DatasetTree.tsx` | the *groups* ordering and the ungrouped bucket |
| `tests/test_grouping_grid_match.py`, `tests/test_grouping_features.py` | the gate, the cache stamp, stem collisions |
