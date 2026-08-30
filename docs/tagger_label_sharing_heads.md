# Anima Tagger — label-sharing heads REFUTED (issue #94), 2026-08-26

> Moved from the `anima_lora` trainer (`docs/`) in the curation split, Phase 3b (2026-08-30); `bench/…` paths below are this repo's `bench/`.

**Status: CLOSED / archived.** The factored head (`FactoredHead`, formerly
`tag_factors.py` under the then-`library.captioning` package (now `anime_tools.tagger`), `--tag_head_kind_core/_spatial`,
`--factored_rank`, `--factored_min_tags_per_word`) was removed from the tree;
the code, its unit test, and the full working-tree patch live in
`_archive/tagger_factored_head/tag_factors.py` and its siblings. The shipped tagger keeps the flat linear
head. What *did* stay in tree from this line is listed at the bottom.

Question: does sharing statistical strength across tags (label-embed cosine
head, or an explicit word/group **factor matrix**) lift the long tail of the
spatial (general-tag) vocab over the flat per-tag linear head that v5 ships?

**Answer: no.** Both sharing heads are worse; the best sharing variant
(factors + full free residual) exactly ties linear overall and is slightly
*worse* in the `<50`-positives bucket it was meant to win.

## Setup

Identical v5 recipe for every arm (32 ep joint + 15 ep spatial refit, bs 64,
lr 1.5e-4, MAP pooling both sides, `--select_metric spatial_ap`, stroked
feature cache, seed 42 → same split). Core side (identity tags) stays
linear in every arm; only the spatial head varies. Scoreboard = the new
**train-frequency-sliced** macro-F1 / spatial AP (`f1_<bin>`,
`spatial_ap_<bin>` in `config.json["freq_sliced"]`, bins on train-split
positives). Seed noise reference: this linear rerun vs shipped v5 =
0.231/0.272 vs 0.237/0.274, so |Δ| ≲ 0.01 is noise.

Checkpoints: `models/captioners/ab94-{linear,label_embed,factored,factored-full}`.

## Results (best-AP checkpoint)

| spatial head | macro-F1 | spatial AP | F1 <50 | AP <50 | F1 50–199 | AP 50–199 |
|---|---|---|---|---|---|---|
| **linear** (v5) | **0.231** | **0.272** | **0.248** | **0.264** | 0.245 | 0.244 |
| label_embed (Qwen prose, centered, scale 30) | 0.078 | 0.077 | 0.074 | 0.038 | 0.070 | 0.068 |
| label_embed + SVD-128 | killed @ep14 — AP plateau 0.049 | | | | | |
| factored, rank-64 residual | 0.199 | 0.221 | 0.213 | 0.200 | 0.203 | 0.204 |
| factored, full free residual | 0.231 | 0.269 | 0.242 | 0.250 | 0.251 | 0.253 |

n tags per bin (F1 / AP): <50 906/745, 50–199 679/685, 200–999 321/351, ≥1000 77/95.

## Why

- **Cosine head (label_embed)** never learns to rank. Before the fix it
  couldn't even train: the head's `F.normalize` promotes to fp32 under bf16
  autocast and the routed `index_copy_` rejected the dtype (every earlier
  label_embed run — `anima-tagger-v2-le` — died on step 1 for this reason).
  After the dtype fix, mean-centering (the Qwen matrix carries a 0.58-norm
  common component, all-pairs cos 0.34), `scale_init` 30 and a no-decay
  `logit_scale`, it trains but stays 3× under linear with a flat slope.
  SVD-truncating the matrix (k=128) sharpens the geometry (random-`z` logit
  spread 0.9 → 2.6) yet plateaus lower. The wall is structural: one unit-norm
  `z` must serve an image's ~30 true tags, so per-tag logit gaps are bounded
  by `scale × Δcos ≈ 1–3`. Frozen prose geometry is a rank bottleneck on top.
- **Factored head, low-rank residual**: trains cleanly but the trunk fights
  the sharing — in the checkpoint, *private* (single-tag) factor detectors
  end up with larger norms than *shared* ones (1.12 vs 0.87) and the rank-64
  residual is barely used (effective weight ratio 0.13). A shared detector's
  gradient is a compromise across every tag that uses it; the model prefers
  the columns it owns. The deficit is flat across frequency bins — an
  over-constrained head, not false sharing on specific tags.
- **Factored + full residual** (= linear + factor term, zero-init): the
  residual grows to the factor term's size (norms 0.92 vs 1.14, cos 0.37) and
  the result is linear ±noise. The factor columns are inert at best.

Take-away: at `min_freq=20` on frozen PE-Core + PE-Spatial features, a
`<50`-positive tag already reaches the same F1 as a 50–199 one under the
plain linear head (0.248 vs 0.245); there is no long-tail deficit for
sharing to fix, and imposing word structure only removes capacity.

## What shipped from this line

- Frequency-sliced val metrics (every tagger run; `train_common.freq_sliced_metrics`).
- `LabelEmbedHead` autocast dtype fix, `label_emb_center`, `label_emb_scale_init`,
  no-decay `logit_scale`, `--tag_emb_svd_k`.
- ~~`FactoredHead` + `tag_factors.py` + per-side head kinds~~ — **removed**
  (archived, see top). Reopen only with a different feature stack or a much
  lower `min_freq` where the tail actually under-performs; the archived patch
  reapplies cleanly on the 2026-08-26 tree.
- `--resident_backing mmap` (default) for the cached-feature loader — the
  ~42 GB resident set now lives in page cache instead of process RSS.
