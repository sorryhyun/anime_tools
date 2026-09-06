# anime_tools/grouping/

Near-twin grouping over PE-Spatial features. Docs: `docs/grouping.md` (what a near-twin is,
`groups.json`, the sidebar's groups ordering, custom embedders, the decensor match tools).

`features.Embedder` protocol (`cls[B,D]` f32 L2-normed + `grid16[B,16,16,D]` f16); default
embedder is PE-Spatial-B16-512 from the vendored tower in `vision/pe.py`, whose weight path comes
from `downloads.py`. Feature cache at `$NEAR_TWIN_CACHE` (default `~/.cache/near_twin/`) is
curation-private, keyed by parent-dir hash + stem and stamped with `(size, mtime_ns)` +
`FEATURE_CACHE_VER`; anything wrong with an entry means recompute, never an error. The stamp is
load-bearing because `resize` rewrites files under a key that doesn't move.
`cli/match_decensored.py` has a different cache shape and they are deliberately not unified.

`features.read_tags` is the one caption read on this side (through `parse_caption`, keyed by
`normalize_tag`). The surface is `GroupRequest` (`requests.py`, torch-free, hyphenated flags,
`FLAG_SEP = "-"`) run by `groups.run_groups`, which resolves `--embedder`'s `module:callable` and
calls `build_groups`; `--source-dir` defaults to `workspace/resized/`. Output `groups.json` is
`MANIFEST_VERSION = 2`, read by the GUI's `dataset.load_groups` (rels only).

Tests: `test_grouping_features`, `test_grouping_grid_match`.
